import cv2
import numpy as np
from typing import Tuple, Dict, List, Any, Optional

import gymnasium as gym
from gymnasium import spaces

from utilities.utils import SDF_RT, polygon_SDF, SE2_kinematics
from envs.target.BrownWalker import BrownWalker
from utilities.KalmanFilter import KalmanFilter, cov_ellipse


class SimpleGymEnv(gym.Env):
    KMAX = 10
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, grid, fov_rad: float, radius: float | None = None, tau: float | None = None,
        num_targets: int = 3, render_mode: Optional[str] = None, use_env_kf: bool = True,
        obs_noise_std: float = 0.0,):
        super().__init__()

        self._global_map = grid
        self.height, self.width = self._global_map.shape

        self.num_targets = int(num_targets)
        self._fov = float(fov_rad)
        self._radius = float(radius) if radius is not None else 200.0
        self._tau = float(tau) if tau is not None else 0.8
        self.USE_ENV_KF = bool(use_env_kf)
        self.OBS_NOISE_STD = float(obs_noise_std)

        self._known = None
        self._visits = None

        self.rng_spawn = None
        self.rng_motion = None
        self.rng_obs = None

        self.EGO_CROP_P = 256
        self.EGO_OUT_N = 128

        self._tmask = np.ones(self.KMAX, dtype=np.uint8)

        self.render_mode = render_mode
        self.window_name = "sim_test"
        self._video_writer = None

        self._robot_width = 5.0
        self._robot_height = 5.0

        free = (self._global_map == 1).astype(np.uint8)
        self._dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)

        half_diag = int(np.hypot(self._robot_width / 2.0, self._robot_height / 2.0))
        self._keepout = max(half_diag, 3)

        self._planning_grid = (self._dist >= self._keepout).astype(np.uint8)
        self._plan_grid = self._planning_grid

        self._sigma_cap_px = 250.0
        self._obs_slot_dim = 4

        self.observation_space = spaces.Dict(
            {
                "robot": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "slots": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.KMAX, self._obs_slot_dim),
                    dtype=np.float32,
                ),
                "ego_map": spaces.Box(
                    low=0,
                    high=255,
                    shape=(4, self.EGO_OUT_N, self.EGO_OUT_N),
                    dtype=np.uint8,
                ),
            }
        )

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.2], dtype=np.float32),
            high=np.array([13.0, 1.2], dtype=np.float32),
            dtype=np.float32,
        )

        self._env_step_count = 0
        self.RECENT_H = 50
        self.RENDER_DRAW_MASKED_KF = True

    def _update_known_with_fov(self):
        if self._known is None:
            return
        poly = self.get_visible_region()
        H, W = self.height, self.width
        mask = np.zeros((H, W), dtype=np.uint8)
        poly_i32 = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        poly_i32[:, 0, 0] = np.clip(poly_i32[:, 0, 0], 0, W - 1)
        poly_i32[:, 0, 1] = np.clip(poly_i32[:, 0, 1], 0, H - 1)
        cv2.fillPoly(mask, [poly_i32], 1)

        seen_free = (mask == 1) & (self._global_map == 1)
        self._known[seen_free] = 1
        self._visits[mask == 1] += 1.0

    def _affine_world_to_ego(self, N: Optional[int] = None, P: Optional[int] = None):
        if N is None:
            N = self.EGO_OUT_N
        if P is None:
            P = self.EGO_CROP_P
        x, y, th = float(self._rbt[0]), float(self._rbt[1]), float(self._rbt[2])
        s = N / float(P)
        c, s_th = np.cos(-th), np.sin(-th)
        M = np.array([[c, -s_th, 0.0], [s_th, c, 0.0]], dtype=np.float64)
        M *= s
        tx = (N * 0.5) - (M[0, 0] * x + M[0, 1] * y)
        ty = (N * 0.5) - (M[1, 0] * x + M[1, 1] * y)
        M[:, 2] = [tx, ty]
        return M.astype(np.float32)

    def _build_ego_map(self) -> np.ndarray:
        N = int(self.EGO_OUT_N)
        P = int(self.EGO_CROP_P)
        H, W = self.height, self.width

        if self._known is None:
            self._known = np.zeros((H, W), dtype=np.int8)
            self._known[self._global_map == 0] = -1
        if self._visits is None:
            self._visits = np.zeros((H, W), dtype=np.float32)

        known_free = ((self._known == 1) & (self._global_map == 1)).astype(np.uint8)
        unknown = ((self._global_map == 1) & (self._known == 0)).astype(np.uint8)

        fov_mask = np.zeros((H, W), dtype=np.uint8)
        poly = self.get_visible_region()
        poly_i32 = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        poly_i32[:, 0, 0] = np.clip(poly_i32[:, 0, 0], 0, W - 1)
        poly_i32[:, 0, 1] = np.clip(poly_i32[:, 0, 1], 0, H - 1)
        cv2.fillPoly(fov_mask, [poly_i32], 1)

        v = self._visits
        if np.any(v > 0):
            vmax = float(np.percentile(v[v > 0], 99))
            vmax = max(vmax, 1.0)
        else:
            vmax = 1.0
        visits_norm = np.clip(v / vmax, 0.0, 1.0).astype(np.float32)

        M_crop = self._affine_world_to_ego(N=P, P=P)

        def _warp2stage(src: np.ndarray, is_binary: bool) -> np.ndarray:
            interp1 = cv2.INTER_NEAREST if is_binary else cv2.INTER_LINEAR
            crop = cv2.warpAffine(
                src.astype(np.float32),
                M_crop,
                (P, P),
                flags=interp1,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            crop = np.clip(crop, 0.0, 1.0).astype(np.float32)
            down = cv2.resize(crop, (N, N), interpolation=cv2.INTER_AREA)
            return np.clip(down, 0.0, 1.0).astype(np.float32)

        ch_known_free = (_warp2stage(known_free, is_binary=True) * 255.0).astype(
            np.uint8
        )
        ch_unknown = (_warp2stage(unknown, is_binary=True) * 255.0).astype(np.uint8)
        ch_fov = (_warp2stage(fov_mask, is_binary=True) * 255.0).astype(np.uint8)
        ch_visits = (
            np.clip(_warp2stage(visits_norm, is_binary=False), 0.0, 1.0) * 255.0
        ).astype(np.uint8)

        return np.stack([ch_known_free, ch_unknown, ch_visits, ch_fov], axis=0)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        self._update_known_with_fov()
        ego_map = self._build_ego_map().astype(np.uint8)

        H, W = self.height, self.width
        robot = np.asarray(self._rbt, dtype=np.float32).ravel()[:3]
        slots = np.zeros((self.KMAX, self._obs_slot_dim), dtype=np.float32)

        for i in range(self.KMAX):
            kf = self._kf_targets[i]

            if hasattr(kf, "x") and kf.x is not None and len(np.asarray(kf.x).ravel()) >= 2:
                mu = np.asarray(kf.x, dtype=np.float64).ravel()
                mx, my = float(mu[0]), float(mu[1])
            elif hasattr(kf, "mu") and kf.mu is not None and len(np.asarray(kf.mu).ravel()) >= 2:
                mu = np.asarray(kf.mu, dtype=np.float64).ravel()
                mx, my = float(mu[0]), float(mu[1])
            else:
                mx, my = 0.0, 0.0

            sx = sy = self._sigma_cap_px
            if hasattr(kf, "P") and kf.P is not None and np.asarray(kf.P).size >= 4:
                try:
                    diag = np.maximum(
                        np.diag(np.asarray(kf.P, dtype=np.float64)[:2, :2]), 0.0
                    )
                    sx = float(np.sqrt(diag[0]))
                    sy = float(np.sqrt(diag[1]))
                except Exception:
                    sx = sy = self._sigma_cap_px

            x_n = float((np.clip(mx, 0, W - 1) / max(W - 1, 1e-6) - 0.5) * 2.0)
            y_n = float((np.clip(my, 0, H - 1) / max(H - 1, 1e-6) - 0.5) * 2.0)
            sx_n = float(np.clip(sx / self._sigma_cap_px, 0.0, 1.0))
            sy_n = float(np.clip(sy / self._sigma_cap_px, 0.0, 1.0))
            slots[i, :] = np.array([x_n, y_n, sx_n, sy_n], dtype=np.float32)

        return {
            "robot": robot.astype(np.float32, copy=False),
            "slots": slots,
            "ego_map": ego_map,
        }

    def _get_info(self) -> Dict[str, Any]:
        eval_states: List[Dict[str, Any]] = []
        visible_flags = np.zeros(self.KMAX, dtype=np.uint8)
        exist_flags = np.zeros(self.KMAX, dtype=np.uint8)

        for i in range(self.KMAX):
            kf = self._kf_targets[i]
            tgt = self._tgts[i]
            exists = tgt is not None
            exist_flags[i] = 1 if exists else 0
            if exists:
                visible_flags[i] = 1 if self.is_target_visible(tgt) else 0

            state_dict = {
                "exists": exists,
                "x_true": tgt.get_pose()[:2].copy()
                if exists
                else np.array([np.nan, np.nan]),
                "mu": kf.x[:2].copy()
                if hasattr(kf, "x") and kf.x is not None
                else np.array([np.nan, np.nan]),
                "Sigma": kf.P[:2, :2].copy()
                if hasattr(kf, "P") and kf.P is not None
                else np.full((2, 2), np.nan),
                "last_update_step": getattr(kf, "_last_update_step", -1),
            }
            eval_states.append(state_dict)

        return {
            "robot_pose": self._rbt,
            "eval_states": eval_states,
            "visible_flags": visible_flags,
            "exist_flags": exist_flags,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
                    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        if seed is None:
            sseq = np.random.SeedSequence()
        else:
            sseq = np.random.SeedSequence(int(seed))
        spawn_ss, motion_ss, obs_ss = sseq.spawn(3)
        self.rng_spawn = np.random.default_rng(spawn_ss)
        self.rng_motion = np.random.default_rng(motion_ss)
        self.rng_obs = np.random.default_rng(obs_ss)

        self._known = np.zeros((self.height, self.width), dtype=np.int8)
        self._known[self._global_map == 0] = -1
        self._visits = np.zeros((self.height, self.width), dtype=np.float32)

        free_mask = self._planning_grid == 1
        free_ys, free_xs = np.where(free_mask)
        assert free_xs.size > 0, "No free space to spawn"

        def _sample_free_xy(min_clear_px: int = 0):
            if min_clear_px > 0:
                m = self._dist >= max(self._keepout, min_clear_px)
                ys, xs = np.where(m)
                if xs.size > 0:
                    i = int(self.rng_spawn.integers(0, xs.size))
                    return float(xs[i]), float(ys[i])
            i = int(self.rng_spawn.integers(0, free_xs.size))
            return float(free_xs[i]), float(free_ys[i])

        if options is None:
            options = {}

        if "robot_xy" in options or "robot_th" in options:
            rx, ry = options.get("robot_xy", (self.width / 2.0, self.height / 2.0))
            rth = options.get(
                "robot_th", float(self.rng_spawn.uniform(-np.pi, np.pi))
            )
            rx, ry, rth = float(rx), float(ry), float(rth)
        else:
            rx, ry = _sample_free_xy(min_clear_px=self._keepout + 5)
            rth = float(self.rng_spawn.uniform(-np.pi, np.pi))

        self._rbt = np.array([rx, ry, rth], dtype=float)
        self._env_step_count = 0
        self._traj = [self._rbt.copy()]

        self._tgts: List[Optional[BrownWalker]] = [None for _ in range(self.KMAX)]
        self._kf_targets: List[KalmanFilter] = []
        self._tmask[:] = 1

        cx, cy = self.width / 2.0, self.height / 2.0
        sigma_y_init = 200.0
        sigma_x_init = 1.5 * sigma_y_init
        init_P = np.diag([sigma_x_init**2, sigma_y_init**2]).astype(float)

        if isinstance(options, dict) and "targets_xy" in options:
            txys = list(options["targets_xy"])
            n_active = int(min(len(txys), self.KMAX))
            placed = [tuple(map(float, xy)) for xy in txys[:n_active]]
        else:
            n_active = int(self.rng_spawn.integers(3, 7))

            placed: List[Tuple[float, float]] = []
            min_pair = 40.0
            min_robot = 60.0
            tries = 0
            while len(placed) < n_active and tries < 5000:
                tries += 1
                tx, ty = _sample_free_xy(min_clear_px=self._keepout + 3)
                if np.hypot(tx - rx, ty - ry) < min_robot:
                    continue
                ok = True
                for px, py in placed:
                    if np.hypot(tx - px, ty - py) < min_pair:
                        ok = False
                        break
                if not ok:
                    continue
                placed.append((tx, ty))

            while len(placed) < n_active:
                tx, ty = _sample_free_xy()
                placed.append((tx, ty))

        for i in range(n_active):
            pose = np.array(
                [
                    placed[i][0],
                    placed[i][1],
                    float(self.rng_spawn.uniform(-np.pi, np.pi)),
                ],
                dtype=float,
            )
            self._tgts[i] = BrownWalker(
                pose.copy(), step_size=1.5, map_shape=(self.height, self.width)
            )

        for _ in range(self.KMAX):
            kf = KalmanFilter(init_pos=[cx, cy], init_P=init_P)
            setattr(kf, "_seen", False)
            setattr(kf, "_last_update_step", -1)
            self._kf_targets.append(kf)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: np.ndarray
             ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self._env_step_count += 1

        action = np.asarray(action, dtype=float).ravel()
        if action.size >= 2:
            v, w = float(action[0]), float(action[1])
            new_rbt = SE2_kinematics(self._rbt, np.array([v, w], dtype=float), self._tau)
            if not self.check_collision(new_rbt):
                self._rbt = new_rbt

        for i in range(self.KMAX):
            if not self._tmask[i] or self._tgts[i] is None:
                continue
            tgt = self._tgts[i]
            old_pose = tgt.get_pose().copy()
            tgt.step()
            if self.check_targets_collision(tgt.get_pose()[:2]):
                tgt.pose = old_pose

        if getattr(self, "USE_ENV_KF", True):
            visible_region = self.get_visible_region()
            for i in range(self.KMAX):
                kf = self._kf_targets[i]
                kf.predict()

                tgt = self._tgts[i]
                if tgt is None:
                    continue
                tgt_pos = tgt.get_pose()[:2]
                if polygon_SDF(visible_region, tgt_pos) < 0:
                    z = np.asarray(tgt_pos, dtype=float)
                    kf.update(z)
                    setattr(kf, "_seen", True)
                    setattr(kf, "_last_update_step", int(self._env_step_count))

        reward = 0.0
        terminated = False
        truncated = False

        if hasattr(self, "_traj"):
            self._traj.append(self._rbt.copy())

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        canvas = cv2.cvtColor(
            (self._global_map * 225).astype(np.uint8), cv2.COLOR_GRAY2BGR
        )
        poly_disp = self.get_visible_region().astype(np.int32)
        cv2.polylines(
            canvas, [poly_disp.reshape(-1, 1, 2)], True, (0, 0, 255), 2
        )

        robot_corners_disp = self.get_robot_corners()
        cv2.fillPoly(canvas, [robot_corners_disp.reshape(-1, 1, 2)], color=(0, 255, 255))
        cv2.polylines(
            canvas,
            [robot_corners_disp.reshape(-1, 1, 2)],
            isClosed=True,
            color=(0, 0, 0),
            thickness=2,
        )

        for i in range(self.KMAX):
            if self._tgts[i] is None:
                continue
            tgt_pose = self._tgts[i].get_pose()
            cv2.circle(
                canvas, (int(tgt_pose[0]), int(tgt_pose[1])), 4, (0, 0, 255), -1
            )

        for i in range(self.KMAX):
            kf = self._kf_targets[i]
            print(i, getattr(kf, "_seen", False), kf.P[0,0], kf.P[1,1])
            try:
                est = kf.x[:2] if hasattr(kf, "x") else np.array([0.0, 0.0], dtype=float)
                P_pos = kf.P[:2, :2] if hasattr(kf, "P") else np.eye(2, dtype=float)
                center, axes, angle = cov_ellipse(est, P_pos)
                vals = [
                    float(center[0]),
                    float(center[1]),
                    float(axes[0]),
                    float(axes[1]),
                    float(angle),
                ]
                if not np.all(np.isfinite(vals)):
                    continue
                axes = (max(1, int(axes[0])), max(1, int(axes[1])))
                if getattr(kf, "_seen", False):
                    cv2.ellipse(canvas, center, axes, angle, 0, 360, (255, 0, 0), 2)
                else:
                    cv2.ellipse(canvas, center, axes, angle, 0, 360, (160, 160, 160), 1)
            except Exception:
                continue

        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)
        return canvas

    def close(self):
        cv2.destroyAllWindows()

    def get_trajectory(self) -> np.ndarray:
        if not hasattr(self, "_traj"):
            return np.zeros((0, 3), dtype=float)
        return np.asarray(self._traj, dtype=float)

    def get_visible_region(self):
        return SDF_RT(self._rbt, self._fov, self._radius, 50, self._global_map)

    def is_target_visible(self, tgt: Optional[BrownWalker]) -> bool:
        if tgt is None:
            return False
        visible_region = self.get_visible_region()
        pos = tgt.get_pose()[0:2]
        return polygon_SDF(visible_region, pos) < 0

    def get_robot_corners(self) -> np.ndarray:
        x, y, theta = self._rbt
        dx = self._robot_width / 2.0
        dy = self._robot_height / 2.0
        corners_local = np.array(
            [[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=float
        )

        angle_for_visuals = -theta
        c, s = np.cos(angle_for_visuals), np.sin(angle_for_visuals)
        R = np.array([[c, -s], [s, c]])
        corners_world = (corners_local @ R) + np.array([x, y])
        return corners_world.astype(np.int32).reshape(-1, 1, 2)

    def check_collision(self, pose: np.ndarray) -> bool:
        x, y, theta = pose
        dx = self._robot_width / 2.0
        dy = self._robot_height / 2.0
        corners_local = np.array(
            [[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=float
        )
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        corners_world = (corners_local @ R) + np.array([x, y])

        H, W = self._global_map.shape
        for cx, cy in corners_world:
            ix, iy = int(round(cx)), int(round(cy))
            if 0 <= iy < H and 0 <= ix < W:
                if self._global_map[iy, ix] == 0:
                    return True
            else:
                return True
        return False

    def check_targets_collision(self, pose: np.ndarray) -> bool:
        xi, yi = int(round(pose[0])), int(round(pose[1]))
        H, W = self._global_map.shape
        if 0 <= yi < H and 0 <= xi < W:
            return self._global_map[yi, xi] == 0
        return True

    def _kf_to_state(self, kf: KalmanFilter) -> np.ndarray:
        if hasattr(kf, "get_state") and callable(kf.get_state):
            s = np.asarray(kf.get_state(), dtype=np.float64).ravel()
            if s.size >= 4:
                return s[:4].astype(np.float32)

        mean = None
        cov = None
        if hasattr(kf, "x"):
            mean = np.asarray(kf.x, dtype=np.float64).ravel()
        elif hasattr(kf, "mu"):
            mean = np.asarray(kf.mu, dtype=np.float64).ravel()

        if hasattr(kf, "P"):
            cov = np.asarray(kf.P, dtype=np.float64)
        elif hasattr(kf, "Sigma"):
            cov = np.asarray(kf.Sigma, dtype=np.float64)

        if mean is None or mean.size < 2:
            mu_x, mu_y = 0.0, 0.0
        else:
            mu_x, mu_y = float(mean[0]), float(mean[1])

        if cov is None or cov.size < 4:
            log_sx = np.inf
            log_sy = np.inf
        else:
            d = np.sqrt(np.maximum(np.diag(cov)[:2], 0.0))
            if not getattr(kf, "_seen", False):
                log_sx = np.inf
                log_sy = np.inf
            else:
                d = np.clip(d, 1e-6, None)
                log_sx = float(np.log(d[0]))
                log_sy = float(np.log(d[1]))

        return np.array([mu_x, mu_y, log_sx, log_sy], dtype=np.float32)