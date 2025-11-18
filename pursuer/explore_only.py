import numpy as np
from dataclasses import dataclass
import cv2
from collections import deque as queue

from typing import List, Tuple, Optional, Dict

TRACKED, LOST = 1, 0


@dataclass
class TrackMeta:
    status: int = LOST
    last_seen: int = -1
    misses: int = 0


from .planner.rrt_star import RRTStar, line_free, path_is_free
from .controller.simple import PurePursuitController


class GoalSelector:
    def __init__(self, grid: np.ndarray,
        plan_grid: Optional[np.ndarray] = None,
        frontier_min_area: int = 1500,
        frontier_opening: int = 6,
        frontier_excl_radius: int = 60,
        frontier_gain_weight: float = 1.0,
        frontier_dist_weight: float = 10.0,
        frontier_known_weight: float = 0.4,
        frontier_area_weight: float = 1.2,
        frontier_gain_radius: int = 200,
        use_batch_minmax: bool = True,
        softmax_temp: float = 0.4,
        sample_frontier: bool = True,
        random_seed: Optional[int] = None,
        use_simple_nearest: bool = True,
        nearest_w_dist: float = 1.0,
        nearest_w_visit: float = 0.3,
        nearest_tabu_radius: float = 70.0,
        nearest_w_tabu: float = 0.8,
    ):
        self.grid = grid
        self.plan_grid = plan_grid if plan_grid is not None else grid

        free_u8 = (self.plan_grid > 0).astype(np.uint8)
        self._edt_plan = cv2.distanceTransform(
            free_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        self._gx = cv2.Sobel(self._edt_plan, cv2.CV_32F, 1, 0, ksize=3)
        self._gy = cv2.Sobel(self._edt_plan, cv2.CV_32F, 0, 1, ksize=3)

        self.frontier_min_area = int(frontier_min_area)
        self.frontier_opening = int(frontier_opening)
        self.frontier_excl_radius = int(frontier_excl_radius)

        self.frontier_gain_weight = float(frontier_gain_weight)
        self.frontier_dist_weight = float(frontier_dist_weight)
        self.frontier_known_weight = float(frontier_known_weight)
        self.frontier_area_weight = float(frontier_area_weight)
        self.frontier_gain_radius = int(frontier_gain_radius)

        H, W = self.grid.shape
        self._map_diag = float(np.hypot(W, H))
        self._gain_area = float(np.pi * (self.frontier_gain_radius ** 2))

        self.use_batch_minmax = bool(use_batch_minmax)
        self.softmax_temp = float(softmax_temp)
        self.sample_frontier = bool(sample_frontier)
        self._rng = np.random.default_rng(random_seed)

        self.use_simple_nearest = bool(use_simple_nearest)
        self.nearest_w_dist = float(nearest_w_dist)
        self.nearest_w_visit = float(nearest_w_visit)
        self.nearest_tabu_radius = float(nearest_tabu_radius)
        self.nearest_w_tabu = float(nearest_w_tabu)
        self._recent_goals = queue(maxlen=40)

        self.last_frontier_mask = None
        self.last_frontier_best_pt = None
        self.last_num_candidates = 0
        self.last_frontier_stats = {
            "pixels": 0,
            "components": 0,
            "large": 0,
            "reachable_applied": False,
        }

    def _snap_to_clear(self, p: np.ndarray, min_clear: float = 12.0,
        rmax_factor: float = 3.0,) -> Optional[np.ndarray]:
        if p is None:
            return None
        H, W = self.plan_grid.shape
        x, y = int(round(float(p[0]))), int(round(float(p[1])))
        if not (0 <= x < W and 0 <= y < H):
            return None

        edt = self._edt_plan
        if (
            edt is not None
            and self.plan_grid[y, x] > 0
            and float(edt[y, x]) >= float(min_clear)
        ):
            return np.array([float(x), float(y)], dtype=float)

        rmax = int(max(12, rmax_factor * float(min_clear)))
        x0, x1 = max(0, x - rmax), min(W, x + rmax + 1)
        y0, y1 = max(0, y - rmax), min(H, y + rmax + 1)
        sub_grid = self.plan_grid[y0:y1, x0:x1]
        if edt is None:
            mask = sub_grid > 0
            if not mask.any():
                return None
            iy, ix = np.argwhere(mask)[0]
            x2, y2 = int(ix + x0), int(iy + y0)
            return np.array([float(x2), float(y2)], dtype=float)

        sub_edt = edt[y0:y1, x0:x1]
        mask = sub_grid > 0
        if not mask.any():
            return None
        edt_masked = np.where(mask, sub_edt, -1.0)
        iy, ix = np.unravel_index(int(np.argmax(edt_masked)), edt_masked.shape)
        x2, y2 = int(ix + x0), int(iy + y0)

        gx = self._gx
        gy = self._gy
        cx, cy = float(x2), float(y2)
        for _ in range(6):
            ix2, iy2 = int(round(cx)), int(round(cy))
            if (
                0 <= ix2 < W
                and 0 <= iy2 < H
                and self.plan_grid[iy2, ix2] > 0
                and edt[iy2, ix2] >= float(min_clear)
            ):
                break
            vx, vy = float(gx[iy2, ix2]), float(gy[iy2, ix2])
            nrm = (vx * vx + vy * vy) ** 0.5 + 1e-6
            cx += (vx / nrm) * 2.0
            cy += (vy / nrm) * 2.0
        x2, y2 = int(np.clip(round(cx), 0, W - 1)), int(
            np.clip(round(cy), 0, H - 1)
        )

        if self.plan_grid[y2, x2] == 0:
            return None
        return np.array([float(x2), float(y2)], dtype=float)

    def _sample_free_near_robot(self, robot_xy: np.ndarray, r: float = 60.0,
                                 tries: int = 256) -> np.ndarray:
        H, W = self.grid.shape
        for _ in range(tries):
            ang = np.random.uniform(-np.pi, np.pi)
            rad = np.random.uniform(0.0, r)
            px = int(round(robot_xy[0] + rad * np.cos(ang)))
            py = int(round(robot_xy[1] + rad * np.sin(ang)))
            if 0 <= px < W and 0 <= py < H and self.grid[py, px] == 1:
                return np.array([px, py], dtype=float)
        return self._sample_free_space()

    def _sample_free_space(self, max_tries=1000) -> np.ndarray:
        H, W = self.grid.shape
        for _ in range(max_tries):
            x = np.random.randint(0, W)
            y = np.random.randint(0, H)
            if self.grid[y, x] == 1:
                return np.array([x, y], dtype=float)
        return np.array(
            [np.random.uniform(0, W - 1), np.random.uniform(0, H - 1)],
            dtype=float,
        )

    def _circle_candidates(self, center, rho=15.0, n=16):
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        pts = np.stack(
            [center[0] + rho * np.cos(theta), center[1] + rho * np.sin(theta)],
            axis=1,
        )
        H, W = self.grid.shape
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        return pts

    def _ellipse_perimeter_candidates(self, mu, P, radius_scale=2.0, n=20):
        eigvals, eigvecs = np.linalg.eigh(P)
        eigvals = np.maximum(eigvals, 1e-6)
        a = radius_scale * np.sqrt(eigvals[1])
        b = radius_scale * np.sqrt(eigvals[0])
        V = eigvecs
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        circ = np.stack([a * np.cos(theta), b * np.sin(theta)], axis=0)
        pts = (V @ circ).T + mu
        H, W = self.grid.shape
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        return pts

    def _ray_farthest_free(self, origin: np.ndarray, theta: float, rmax: float,
                            step: float = 2.0) -> Optional[np.ndarray]:
        H, W = self.plan_grid.shape
        x, y = origin.astype(float)
        c, s = np.cos(theta), np.sin(theta)
        p_last = None
        for r in np.arange(0.0, rmax + step, step):
            px = int(round(x + r * c))
            py = int(round(y + r * s))
            if px < 0 or px >= W or py < 0 or py >= H:
                break
            if self.plan_grid[py, px] == 0:
                break
            p_last = np.array([px, py], dtype=float)
        return p_last

    def exploration_goal(self, robot_xy: np.ndarray) -> np.ndarray:
        H, W = self.plan_grid.shape
        Rmin = 0.2 * np.sqrt(H ** 2 + W ** 2)
        Rmax = 0.5 * np.sqrt(H ** 2 + W ** 2)
        for _ in range(96):
            theta = np.random.uniform(-np.pi, np.pi)
            rmax = np.random.uniform(Rmin, Rmax)
            p = self._ray_farthest_free(robot_xy, theta, rmax, step=3.0)
            if p is not None:
                return p
        return self._sample_free_near_robot(robot_xy)

    def tracking_goal(self, robot_xy: np.ndarray, mu: np.ndarray, P: np.ndarray) -> np.ndarray:
        eigvals, _ = np.linalg.eigh(P[:2, :2])
        rho = float(2.0 * np.sqrt(max(eigvals) + 1e-6))
        rho = float(np.clip(rho * 10.0, 8.0, 35.0))
        candidates = self._circle_candidates(mu[:2], rho=rho, n=16)
        best, best_cost = None, 1e18
        for c in candidates:
            if not line_free(self.plan_grid, tuple(robot_xy), tuple(c)):
                continue
            if not line_free(self.plan_grid, tuple(c), tuple(mu[:2])):
                continue
            dist = np.linalg.norm(c - robot_xy)
            uncert = float(np.trace(P[:2, :2]))
            cost = dist + 0.1 * uncert
            if cost < best_cost:
                best, best_cost = c, cost
        return best if best is not None else mu[:2]

    def reacquire_goal(self, robot_xy: np.ndarray, mu: np.ndarray, P: np.ndarray) -> np.ndarray:
        candidates = self._ellipse_perimeter_candidates(
            mu[:2], P[:2, :2], radius_scale=2.0, n=20
        )
        best, best_score = None, -1e18
        for c in candidates:
            if not line_free(self.plan_grid, tuple(robot_xy), tuple(c)):
                continue
            los = 1.0 if line_free(self.plan_grid, tuple(c), tuple(mu[:2])) else 0.0
            dist = np.linalg.norm(c - robot_xy)
            score = 1.0 * los - 0.02 * dist
            if score > best_score:
                best, best_score = c, score
        return best if best is not None else self.exploration_goal(robot_xy)

    def frontier_explore(self, robot_xy: np.ndarray, known: np.ndarray,
                          visit_counts: np.ndarray) -> Optional[np.ndarray]:
        H, W = self.grid.shape
        rx, ry = int(round(robot_xy[0])), int(round(robot_xy[1]))

        known_free = known == 1
        unknown = (self.grid == 1) & (known == 0)

        k4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        dilated_unknown = cv2.dilate(unknown.astype(np.uint8), k4, iterations=1)
        frontiers_raw = dilated_unknown.astype(bool) & known_free

        reachable_applied = False
        reachable_mask = np.zeros((H + 2, W + 2), np.uint8)
        if 0 <= rx < W and 0 <= ry < H and known_free[ry, rx]:
            reachable_map = known_free.astype(np.uint8) * 255
            cv2.floodFill(reachable_map, reachable_mask, (rx, ry), 128)
            reachable = reachable_map == 128
            reachable_applied = True
            frontier_mask = frontiers_raw & reachable
        else:
            frontier_mask = frontiers_raw

        ksz = max(1, self.frontier_opening)
        if ksz > 1:
            kernel = np.ones((ksz, ksz), np.uint8)
            frontier_mask = cv2.morphologyEx(
                frontier_mask.astype(np.uint8),
                cv2.MORPH_OPEN,
                kernel,
            ).astype(bool)

        self.last_frontier_mask = frontier_mask.copy()
        total_frontier_pixels = int(np.count_nonzero(frontier_mask))
        if not frontier_mask.any():
            self.last_frontier_stats = {
                "pixels": 0,
                "components": 0,
                "large": 0,
                "reachable_applied": bool(reachable_applied),
            }
            self.last_frontier_best_pt = None
            return None

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            frontier_mask.astype(np.uint8),
            connectivity=8,
        )
        n_components = max(0, num_labels - 1)
        n_large = (
            int(np.sum(stats[1:, cv2.CC_STAT_AREA] >= self.frontier_min_area))
            if num_labels > 1
            else 0
        )
        self.last_frontier_stats = {
            "pixels": total_frontier_pixels,
            "components": n_components,
            "large": n_large,
            "reachable_applied": bool(reachable_applied),
        }

        if self.use_simple_nearest:
            best_pt = None
            best_cost = 1e18
            recent = list(self._recent_goals)

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < self.frontier_min_area:
                    continue

                mask_i = labels == i
                ys_i, xs_i = np.where(mask_i)
                if ys_i.size == 0:
                    continue
                d2 = (xs_i - rx) * (xs_i - rx) + (ys_i - ry) * (ys_i - ry)
                j = int(np.argmin(d2))
                cand = np.array(
                    [float(xs_i[j]), float(ys_i[j])],
                    dtype=float,
                )
                dist = float(np.sqrt(d2[j]))
                if dist < float(self.frontier_excl_radius):
                    continue

                vdir = cand - robot_xy
                nrm = np.linalg.norm(vdir) + 1e-6
                step = min(12.0, 0.15 * self.frontier_gain_radius)
                cand = cand + (vdir / nrm) * step
                cand[0] = np.clip(cand[0], 0, W - 1)
                cand[1] = np.clip(cand[1], 0, H - 1)

                r = int(max(8, self.frontier_gain_radius // 6))
                disk = np.zeros_like(self.grid, dtype=np.uint8)
                cv2.circle(
                    disk,
                    (int(round(cand[0])), int(round(cand[1]))),
                    r,
                    1,
                    thickness=-1,
                )
                local_visit = (
                    float(visit_counts[disk == 1].mean())
                    if np.any(disk)
                    else 0.0
                )

                tabu = 0.0
                if recent:
                    dmins = [float(np.linalg.norm(cand - rp)) for rp in recent]
                    dmin = min(dmins)
                    tabu = max(
                        0.0,
                        1.0
                        - dmin / (self.nearest_tabu_radius + 1e-6),
                    )

                cost = (
                    self.nearest_w_dist * dist
                    + self.nearest_w_visit * local_visit
                    + self.nearest_w_tabu * tabu
                )
                if cost < best_cost:
                    best_cost, best_pt = cost, cand

            if best_pt is not None:
                snapped = self._snap_to_clear(
                    best_pt,
                    min_clear=max(10.0, 0.15 * self.frontier_gain_radius),
                )
                if snapped is not None:
                    best_pt = snapped
                    self.last_frontier_best_pt = best_pt.copy()
                    self._recent_goals.append(best_pt.copy())
                    self.last_num_candidates = n_large
                    return best_pt

        def _build_candidates(apply_excl: bool) -> list:
            cands = []
            if num_labels > 1:
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area < self.frontier_min_area:
                        continue
                    cx, cy = centroids[i]
                    cand = np.array([cx, cy], dtype=float)
                    dist = np.linalg.norm(cand - robot_xy)
                    if apply_excl and (dist < float(self.frontier_excl_radius)):
                        continue

                    r = int(self.frontier_gain_radius)
                    gain_mask = np.zeros_like(self.grid, dtype=np.uint8)
                    cv2.circle(
                        gain_mask,
                        (int(round(cx)), int(round(cy))),
                        r,
                        1,
                        thickness=-1,
                    )

                    unknown_mask = gain_mask == 1
                    unknown_gain = np.count_nonzero(
                        unknown_mask & unknown
                    )
                    unknown_gain = np.sqrt(max(unknown_gain, 0))
                    local_visit_mean = (
                        visit_counts[unknown_mask].mean()
                        if np.any(gain_mask)
                        else 0.0
                    )

                    dist_norm = float(dist) / (self._map_diag + 1e-6)
                    gain_norm = float(unknown_gain) / (self._gain_area + 1e-6)
                    v_globalmax = (
                        float(visit_counts.max())
                        if visit_counts.size
                        else 0.0
                    )
                    visit_norm = (
                        float(local_visit_mean / (v_globalmax + 1e-6))
                        if v_globalmax > 0.0
                        else 0.0
                    )

                    cands.append(
                        (
                            cand,
                            gain_norm,
                            dist_norm,
                            visit_norm,
                            float(area),
                        )
                    )
            return cands

        candidates = _build_candidates(apply_excl=True)
        if len(candidates) == 0:
            candidates = _build_candidates(apply_excl=False)
        self.last_num_candidates = len(candidates)

        if candidates:
            pts = np.array([c[0] for c in candidates], dtype=float)
            gains = np.array([c[1] for c in candidates], dtype=float)
            dists = np.array([c[2] for c in candidates], dtype=float)
            visits = np.array([c[3] for c in candidates], dtype=float)
            areas = np.array([c[4] for c in candidates], dtype=float)

            def _mm(v: np.ndarray) -> np.ndarray:
                vmin = float(np.min(v))
                vmax = float(np.max(v))
                if not np.isfinite(vmin) or not np.isfinite(
                    vmax
                ) or vmax <= vmin:
                    return np.zeros_like(v)
                return (v - vmin) / ((vmax - vmin) + 1e-6)

            if self.use_batch_minmax:
                g = _mm(gains)
                d = _mm(dists)
                v = _mm(visits)
                a = _mm(np.sqrt(np.maximum(areas, 1.0)))
            else:
                g, d, v = gains, dists, visits
                a = np.sqrt(np.maximum(areas, 1.0))

            scores = (
                self.frontier_gain_weight * g
                + self.frontier_area_weight * a
                - self.frontier_dist_weight * d
                - self.frontier_known_weight * v
            )

            if self.sample_frontier:
                T = max(self.softmax_temp, 1e-6)
                logits = scores / T
                logits -= np.max(logits)
                p = np.exp(logits)
                psum = float(p.sum())
                if not np.isfinite(psum) or psum <= 0:
                    idx = int(np.argmax(scores))
                else:
                    idx = int(self._rng.choice(len(pts), p=p / psum))
            else:
                idx = int(np.argmax(scores))

            cand_pt = pts[idx].copy()
            best_pt = self._snap_to_clear(
                cand_pt,
                min_clear=max(10.0, 0.15 * self.frontier_gain_radius),
            )
            if best_pt is None:
                best_pt = self._snap_to_clear(robot_xy, min_clear=8.0)
                if best_pt is None:
                    return None
            self.last_frontier_best_pt = best_pt.copy()
            self._recent_goals.append(best_pt.copy())
            return best_pt

        ys, xs = np.where(frontier_mask)
        if len(xs) == 0:
            return None
        d2 = (xs - robot_xy[0]) ** 2 + (ys - robot_xy[1]) ** 2
        best_idx = np.argmin(d2)
        best_pt = np.array([xs[best_idx], ys[best_idx]], dtype=float)
        self.last_frontier_best_pt = best_pt.copy()
        return best_pt


class Coordinator:
    def __init__(
        self,
        grid: np.ndarray,
        kf_targets: List[object],
        k_consec_miss: int = 3,
        timeout_lost: int = 60,
        goal_hold_steps: int = 30,
        plan_grid: Optional[np.ndarray] = None,
        params: Optional[dict] = None,
        goal_params: Optional[dict] = None,
    ):
        self.grid = grid

        safety_margin_px = int((params or {}).get("plan_safety_margin_px", 10))
        src = (
            plan_grid.astype(np.uint8)
            if plan_grid is not None
            else grid.astype(np.uint8)
        )

        free_u8 = (src > 0).astype(np.uint8)
        edt_src = cv2.distanceTransform(
            free_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )

        if safety_margin_px > 0:
            self.plan_grid = (edt_src >= float(safety_margin_px)).astype(
                np.uint8
            )
            self._edt_plan = edt_src.astype(np.float32)
            self.TH_SAFE_MARGIN = float(safety_margin_px)
        else:
            self.plan_grid = free_u8.astype(np.uint8)
            self._edt_plan = None
            self.TH_SAFE_MARGIN = float(max(0, safety_margin_px))

        self.kf_targets = kf_targets
        self.step_count = 0
        self.meta: List[TrackMeta] = [TrackMeta() for _ in kf_targets]
        self._prev_kf_last: Dict[int, int] = {
            i: int(getattr(k, "_last_update_step", -1))
            for i, k in enumerate(self.kf_targets)
        }
        self.K_CONSEC_MISS = k_consec_miss
        self.TIMEOUT_LOST = timeout_lost
        self.GOAL_HOLD_STEPS = goal_hold_steps
        self.last_goal: Optional[np.ndarray] = None
        self.last_goal_hold_until: int = -1

        self.goal_selector = GoalSelector(
            grid,
            plan_grid=self.plan_grid,
            frontier_min_area=1500,
            frontier_opening=0,
            frontier_gain_weight=1.0,
            frontier_dist_weight=10.0,
            frontier_known_weight=0.4,
            frontier_gain_radius=200,
            frontier_area_weight=1.5,
            frontier_excl_radius=50,
            use_batch_minmax=True,
            softmax_temp=0.4,
            sample_frontier=True,
            use_simple_nearest=True,
            nearest_w_dist=1.0,
            nearest_w_visit=0.3,
            nearest_tabu_radius=70.0,
            nearest_w_tabu=0.8,
        )
        if goal_params:
            for k, v in goal_params.items():
                if hasattr(self.goal_selector, k):
                    setattr(self.goal_selector, k, v)

        self.controller = PurePursuitController()

        self.known = np.zeros_like(self.grid, dtype=np.int8)
        self.known[self.grid == 0] = -1
        self.visit_counts = np.zeros_like(self.grid, dtype=np.float32)
        self._path: Optional[np.ndarray] = None
        self._path_goal: Optional[np.ndarray] = None

        self.traj_hist = queue(maxlen=2048)
        self.prev_xy: Optional[np.ndarray] = None
        self.stall_count: int = 0
        self.STALL_EPS: float = 1.0
        self.STALL_STEPS: int = 10
        self.RETREAT_STEPS: int = 30
        self.RETREAT_RADIUS: int = 40
        self.RETREAT_PENALTY: float = 8.0
        self.RECOVER_TIMEOUT: int = 120
        self.RECOVERING: bool = False
        self.recover_goal: Optional[np.ndarray] = None
        self.recover_deadline: int = -1

        self.WIN_N: int = 12
        self._win_v = queue(maxlen=self.WIN_N)
        self._win_abs_w = queue(maxlen=self.WIN_N)
        self._win_new = queue(maxlen=self.WIN_N)
        self._win_prev_xy: Optional[np.ndarray] = None
        self._win_prev_th: Optional[float] = None
        self._known_count: int = int(np.count_nonzero(self.known == 1))

        self.TH_NEW: int = 12
        self.TH_MEAN_W: float = 0.6
        self.TH_MEAN_V: float = 3.0
        self.SPIN_K: int = 3
        self.COOLDOWN_STEPS: int = 30

        self.spin_debounce: int = 0
        self.cooldown_until: int = -1

        self.no_path_count: int = 0
        self.NO_PATH_TRIGGER: int = 4

        if getattr(self, "_edt_plan", None) is None:
            self._edt_plan = cv2.distanceTransform(
                (self.plan_grid > 0).astype(np.uint8),
                cv2.DIST_L2,
                cv2.DIST_MASK_PRECISE,
            )
        self.controller.set_clearance_maps(self._edt_plan)

        self.TH_TRACK_SIGMA_PX: float = 10.0
        self.TH_REACQ_SIGMA_PX: float = 25.0
        self.STALE_REACQ_STEPS: int = 45

        self.mode: str = "EXPLORE"
        self.track_since: int = -1
        self.track_origin: str = ""

        self.reacq_lock_id: Optional[int] = None
        self.reacq_started_step: int = -1
        self.reacq_attempts: int = 0

        self.TH_TRACK_DUR_STEPS: int = 80
        self.TH_TRACK_DUR_AFTER_REACQ: int = 0
        self.REACQ_MAX_STEPS: int = 60

        self.ENABLE_REACQUIRE: bool = False
        self.SWITCH_ON_NEW: bool = False
        self.FORCE_EXPLORE_ONLY: bool = True

        self.manual_mode: bool = False
        self.manual_goals: list = []
        self.manual_reach_eps: float = 10.0
        self.manual_loop: bool = True
        self._manual_idx: int = 0

        self._dead_ids: set[int] = set()

        if params:
            for k, v in params.items():
                if hasattr(self, k):
                    setattr(self, k, v)
        self.ENABLE_REACQUIRE = False
        self.SWITCH_ON_NEW = False
        self.FORCE_EXPLORE_ONLY = True

    def _connected_in_band(self, start: np.ndarray, goal: np.ndarray, band: np.ndarray) -> bool:
        if start is None or goal is None or band is None:
            return False
        H, W = band.shape
        sx, sy = int(round(start[0])), int(round(start[1]))
        gx, gy = int(round(goal[0])), int(round(goal[1]))
        if not (0 <= sx < W and 0 <= sy < H and 0 <= gx < W and 0 <= gy < H):
            return False
        if band[sy, sx] == 0 or band[gy, gx] == 0:
            return False
        mask = np.zeros((H + 2, W + 2), np.uint8)
        ff = (band.astype(np.uint8) * 255).copy()
        cv2.floodFill(ff, mask, (sx, sy), 128)
        return bool(ff[gy, gx] == 128)

    def _dedup_path(self, path: np.ndarray) -> Optional[np.ndarray]:
        if path is None:
            return None
        P = np.asarray(path, float)
        if P.ndim != 2 or P.shape[0] < 2:
            return None
        keep = [0]
        for i in range(1, len(P)):
            if np.linalg.norm(P[i] - P[keep[-1]]) > 1e-6:
                keep.append(i)
        if len(keep) < 2:
            return None
        return P[keep]

    def _path_is_free(self, path: np.ndarray) -> bool:
        if path is None or len(path) < 2:
            return False
        for i in range(len(path) - 1):
            if not line_free(
                self.plan_grid,
                tuple(path[i]),
                tuple(path[i + 1]),
            ):
                return False
        return True

    def _trim_path(self, path: np.ndarray, rxy: np.ndarray, 
                   trim_dist: float = 20.0) -> np.ndarray:
        P = np.asarray(path, float)
        d = np.hypot(P[:, 0] - rxy[0], P[:, 1] - rxy[1])
        i0 = int(np.argmin(d))
        acc = 0.0
        for i in range(i0, len(P) - 1):
            seg = float(np.linalg.norm(P[i + 1] - P[i]))
            acc += seg
            if acc >= trim_dist:
                return P[i:]
        return P[i0:]

    def _need_replan(self, robot_xy: np.ndarray, goal_xy: np.ndarray) -> bool:
        if self._path is None or self._path_goal is None:
            return True
        if goal_xy is None:
            return True
        goal_shift = np.linalg.norm(goal_xy - self._path_goal) > 20.0
        path_bad = not self._path_is_free(self._path)
        near_end = np.linalg.norm(self._path[-1] - robot_xy) < 10.0
        return goal_shift or path_bad or near_end

    def _update_maps_with_fov(self, fov_poly: Optional[np.ndarray]):
        if fov_poly is None or len(fov_poly) < 3:
            return
        H, W = self.grid.shape
        poly = np.asarray(fov_poly, dtype=np.int32).reshape(-1, 1, 2)
        mask = np.zeros((H, W), dtype=np.uint8)
        poly[:, 0, 0] = np.clip(poly[:, 0, 0], 0, W - 1)
        poly[:, 0, 1] = np.clip(poly[:, 0, 1], 0, H - 1)
        cv2.fillPoly(mask, [poly], 1)
        seen_free = (mask == 1) & (self.grid == 1)
        self.known[seen_free] = 1
        self.visit_counts[mask == 1] += 1.0
        kc = int(np.count_nonzero(self.known == 1))
        inc = max(0, kc - int(self._known_count))
        self._known_count = kc
        self._win_new.append(float(inc))

    def step(self, robot_state: np.ndarray,
        detections: Optional[List[Tuple[int, Optional[np.ndarray]]]] = None,
        fov_poly: Optional[np.ndarray] = None,tmask: Optional[np.ndarray] = None,) -> Dict:
        
        _ = detections
        self.step_count += 1

        self._update_maps_with_fov(fov_poly)

        if self.step_count == 1:
            rx0, ry0 = int(round(robot_state[0])), int(
                round(robot_state[1])
            )
            H, W = self.grid.shape
            if (
                0 <= rx0 < W
                and 0 <= ry0 < H
                and self.grid[ry0, rx0] == 1
            ):
                self.known[ry0, rx0] = 1

        cur_xy = robot_state[:2].astype(float)
        cur_th = float(robot_state[2]) if robot_state.size >= 3 else 0.0
        if self._win_prev_xy is not None:
            v_step = float(np.linalg.norm(cur_xy - self._win_prev_xy))
        else:
            v_step = 0.0
        if self._win_prev_th is not None:
            dth = (cur_th - self._win_prev_th + np.pi) % (2 * np.pi) - np.pi
            w_step = abs(float(dth))
        else:
            w_step = 0.0
        self._win_prev_xy = cur_xy.copy()
        self._win_prev_th = cur_th
        self._win_v.append(v_step)
        self._win_abs_w.append(w_step)

        drop_ids: List[int] = []
        if tmask is None:
            active_idx = list(range(len(self.kf_targets)))
        else:
            tmask_arr = np.asarray(tmask).astype(bool)
            active_idx = list(np.where(tmask_arr)[0])
        active_idx = [i for i in active_idx if i not in self._dead_ids]

        newly_tracked: set[int] = set()
        seen_ids: set[int] = set()

        for tid in active_idx:
            kf = self.kf_targets[tid]
            prev_status = self.meta[tid].status
            last_update_env = int(getattr(kf, "_last_update_step", -1))
            prev_last_update_planner = int(self._prev_kf_last.get(tid, -1))

            if last_update_env > prev_last_update_planner:
                m = self.meta[tid]
                m.last_seen = self.step_count
                m.misses = 0
                m.status = TRACKED
                seen_ids.add(tid)
                if prev_status != TRACKED:
                    newly_tracked.add(tid)
            self._prev_kf_last[tid] = last_update_env

        for tid in active_idx:
            if tid not in seen_ids:
                self.meta[tid].misses += 1
                if self.meta[tid].misses >= self.K_CONSEC_MISS:
                    self.meta[tid].status = LOST

        robot_xy = robot_state[:2]
        self._update_stall(robot_xy)

        mean_w = float(np.mean(self._win_abs_w)) if len(self._win_abs_w) > 0 else 0.0
        mean_v = float(np.mean(self._win_v)) if len(self._win_v) > 0 else 0.0
        sum_new = float(np.sum(self._win_new)) if len(self._win_new) > 0 else 0.0

        spin_hit = (
            mean_w > self.TH_MEAN_W
            and mean_v < self.TH_MEAN_V
            and sum_new < self.TH_NEW
        )
        if spin_hit:
            self.spin_debounce += 1
        else:
            self.spin_debounce = 0
        spin_trig = self.spin_debounce >= self.SPIN_K
        map_stall_hit = sum_new < self.TH_NEW

        no_path_hit = False
        pending_detect = (
            spin_trig
            or map_stall_hit
            or self.stall_count >= self.STALL_STEPS
            or no_path_hit
        )

        frontier_ms = reacq_ms = track_ms = 0.0
        mode = "EXPLORE"
        goal = None

        if (
            self.manual_mode
            and isinstance(self.manual_goals, (list, tuple))
            and len(self.manual_goals) > 0
        ):
            self._manual_idx = int(
                np.clip(self._manual_idx, 0, len(self.manual_goals) - 1)
            )
            goal = np.asarray(
                self.manual_goals[self._manual_idx],
                dtype=float,
            )[:2]
            if np.linalg.norm(robot_state[:2] - goal) <= float(
                self.manual_reach_eps
            ):
                if self._manual_idx + 1 < len(self.manual_goals):
                    self._manual_idx += 1
                elif self.manual_loop:
                    self._manual_idx = 0
                goal = np.asarray(
                    self.manual_goals[self._manual_idx],
                    dtype=float,
                )[:2]

        if goal is None:
            goal = self.goal_selector.frontier_explore(
                robot_state[:2],
                self.known,
                self.visit_counts,
            )
            frontier_ms = 0.0
            if goal is None:
                goal = self.goal_selector._sample_free_near_robot(
                    robot_state[:2]
                )

        if not self.RECOVERING:
            if (
                self.last_goal is not None
                and self.step_count < self.last_goal_hold_until
                and goal is not None
            ):
                goal = self.last_goal
            else:
                self.last_goal = goal
                self.last_goal_hold_until = (
                    self.step_count + self.GOAL_HOLD_STEPS
                )

        H_pg, W_pg = self.plan_grid.shape
        rx, ry = int(round(robot_state[0])), int(round(robot_state[1]))
        in_band_start = (
            0 <= rx < W_pg
            and 0 <= ry < H_pg
            and self.plan_grid[ry, rx] > 0
        )
        edt_here = (
            float(self._edt_plan[ry, rx])
            if self._edt_plan is not None
            and 0 <= rx < W_pg
            and 0 <= ry < H_pg
            else float("nan")
        )

        entry_mode = False
        if not in_band_start:
            snap_here = self.goal_selector._snap_to_clear(
                np.asarray(robot_state[:2], float),
                min_clear=float(self.TH_SAFE_MARGIN),
            )
            if snap_here is not None:
                entry_path = np.vstack(
                    [np.asarray(robot_state[:2], float), snap_here.astype(float)]
                )
                entry_path = self._dedup_path(entry_path)
                if entry_path is not None:
                    self._path = entry_path
                    self._path_goal = snap_here.copy()
                    entry_mode = True
                    self.no_path_count = 0

        if not entry_mode:
            plan_start = robot_xy
            plan_goal = goal

            snapped_start = self.goal_selector._snap_to_clear(
                np.asarray(robot_state[:2], dtype=float),
                min_clear=float(self.TH_SAFE_MARGIN),
            )
            if snapped_start is not None:
                plan_start = snapped_start
            if plan_goal is not None:
                snapped_goal = self.goal_selector._snap_to_clear(
                    np.asarray(plan_goal, dtype=float),
                    min_clear=float(self.TH_SAFE_MARGIN),
                )
                if snapped_goal is not None:
                    plan_goal = snapped_goal

            if self._need_replan(robot_xy, goal):
                band = self.plan_grid
                relaxed_used = False
                relax_margin = max(0.0, float(self.TH_SAFE_MARGIN) - 2.0)

                if (
                    plan_goal is not None
                    and plan_start is not None
                    and self._edt_plan is not None
                ):
                    conn_strict = self._connected_in_band(
                        plan_start,
                        plan_goal,
                        band,
                    )
                    if not conn_strict:
                        band_relaxed = (
                            self._edt_plan >= float(relax_margin)
                        ).astype(np.uint8)
                        conn_relaxed = self._connected_in_band(
                            plan_start,
                            plan_goal,
                            band_relaxed,
                        )
                        if conn_relaxed:
                            band = band_relaxed
                            relaxed_used = True

                if plan_goal is not None:
                    burst = self.no_path_count >= self.NO_PATH_TRIGGER
                    max_iter = 1400 if burst else 700
                    goal_rad = 35.0 if burst else 25.0
                    planner = RRTStar(
                        start=plan_start,
                        goal=plan_goal,
                        grid=band,
                        max_iter=max_iter,
                        step_size=20.0,
                        goal_sample_rate=0.25,
                        search_radius=115.0,
                        goal_radius=goal_rad,
                    )
                    plan = planner.plan()
                    if plan is not None and path_is_free(
                        self.plan_grid, plan
                    ):
                        plan2 = self._dedup_path(plan)
                        if plan2 is not None:
                            self._path = plan2
                            self._path_goal = (
                                plan_goal.copy()
                                if plan_goal is not None
                                else None
                            )
                        else:
                            self._path, self._path_goal = None, None
                    else:
                        if plan_goal is not None:
                            gx, gy = int(round(plan_goal[0])), int(
                                round(plan_goal[1])
                            )
                            H, W = self.grid.shape
                            if 0 <= gx < W and 0 <= gy < H:
                                cv2.circle(
                                    self.visit_counts,
                                    (gx, gy),
                                    int(
                                        max(
                                            20,
                                            0.2
                                            * self.goal_selector.frontier_gain_radius,
                                        )
                                    ),
                                    float(5.0),
                                    thickness=-1,
                                )
                        self._path, self._path_goal = None, None
                else:
                    self._path, self._path_goal = None, None

            elif self._path is not None:
                self._path = self._trim_path(
                    self._path,
                    robot_xy,
                    trim_dist=self.controller.lookahead * 0.7,
                )

        can_trigger = (
            self.step_count >= self.cooldown_until
            and not self.RECOVERING
            and not entry_mode
        )
        if pending_detect and can_trigger:
            self._start_recovery(robot_xy)

        self._check_recovery_done(robot_xy)

        if self._path is None:
            self.no_path_count += 1
        else:
            self.no_path_count = 0
        no_path_hit = self.no_path_count >= self.NO_PATH_TRIGGER
        if not self.RECOVERING and not entry_mode and no_path_hit:
            self._start_recovery(robot_xy)

        action = (
            self.controller.compute(robot_state, self._path)
            if self._path is not None
            else None
        )

        return {
            "mode": "RECOVER" if self.RECOVERING else mode,
            "goal": goal,
            "path": self._path,
            "action": action,
            "drop_ids": drop_ids,
        }

    def _update_stall(self, robot_xy: np.ndarray):
        if self.prev_xy is None:
            self.prev_xy = robot_xy.copy()
        moved = float(np.linalg.norm(robot_xy - self.prev_xy))
        self.prev_xy = robot_xy.copy()
        if moved < self.STALL_EPS:
            self.stall_count += 1
        else:
            self.stall_count = 0
        self.traj_hist.append(robot_xy.copy())

    def _retreat_goal_from_history(self) -> Optional[np.ndarray]:
        if len(self.traj_hist) < 2:
            return None
        idx = max(len(self.traj_hist) - 1 - self.RETREAT_STEPS, 0)
        return np.asarray(list(self.traj_hist)[idx], dtype=float).copy()

    def _penalize_recent_area(self):
        H, W = self.grid.shape
        mask = np.zeros((H, W), dtype=np.uint8)
        pts = np.array(
            list(self.traj_hist)[-self.RETREAT_STEPS:],
            dtype=int,
        )
        for p in pts:
            x, y = int(p[0]), int(p[1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(
                    mask,
                    (x, y),
                    self.RETREAT_RADIUS,
                    1,
                    thickness=-1,
                )
        self.visit_counts[mask == 1] += float(self.RETREAT_PENALTY)

    def _start_recovery(self, robot_xy: np.ndarray):
        self.spin_debounce = 0
        H, W = self.grid.shape
        rng = np.random.default_rng(int(self.step_count) + 1337)
        K = 60
        r_min, r_max = 200.0, 300.0

        recent_mask = np.zeros((H, W), dtype=np.uint8)
        if len(self.traj_hist) > 1:
            pts = np.array(
                list(self.traj_hist)[-self.RETREAT_STEPS:],
                dtype=int,
            )
            for p in pts:
                x, y = int(p[0]), int(p[1])
                if 0 <= x < W and 0 <= y < H:
                    cv2.circle(
                        recent_mask,
                        (x, y),
                        self.RETREAT_RADIUS,
                        1,
                        thickness=-1,
                    )

        cands = []
        for _ in range(K):
            ang = float(rng.uniform(-np.pi, np.pi))
            rad = float(rng.uniform(r_min, r_max))
            cx = int(round(robot_xy[0] + rad * np.cos(ang)))
            cy = int(round(robot_xy[1] + rad * np.sin(ang)))
            if cx < 0 or cx >= W or cy < 0 or cy >= H:
                continue
            if self.plan_grid[cy, cx] == 0:
                continue
            cands.append((cx, cy))
        if not cands:
            goal = self._retreat_goal_from_history()
            if goal is None:
                return
            self._penalize_recent_area()
            self.recover_goal = goal
            self.recover_deadline = self.step_count + self.RECOVER_TIMEOUT
            self.RECOVERING = True
            planner = RRTStar(
                start=robot_xy,
                goal=goal,
                grid=self.plan_grid,
                max_iter=300,
                step_size=40.0,
            )
            plan = planner.plan()
            if plan is not None:
                self._path, self._path_goal = plan, goal.copy()
            else:
                self.RECOVERING = False
                self.recover_goal = None
            return

        edt = self._edt_plan
        best, best_score = None, -1e18
        for (cx, cy) in cands:
            clr = float(edt[cy, cx]) if edt is not None else 0.0
            vloc = float(self.visit_counts[cy, cx])
            away = 1.0 if recent_mask[cy, cx] == 0 else 0.0
            score = 2.0 * clr + 1.5 * away - 0.5 * vloc
            if score > best_score:
                best_score, best = score, (float(cx), float(cy))
        if best is None:
            return
        goal = np.array(best, dtype=float)
        self._penalize_recent_area()
        self.recover_goal = goal
        self.recover_deadline = self.step_count + self.RECOVER_TIMEOUT
        self.RECOVERING = True
        planner = RRTStar(
            start=robot_xy,
            goal=goal,
            grid=self.plan_grid,
            max_iter=300,
            step_size=40.0,
        )
        plan = planner.plan()
        if plan is not None:
            self._path, self._path_goal = plan, goal.copy()
        else:
            self.RECOVERING = False
            self.recover_goal = None

    def _check_recovery_done(self, robot_xy: np.ndarray):
        if not self.RECOVERING:
            return
        close = self.recover_goal is not None and np.linalg.norm(
            robot_xy - self.recover_goal
        ) < 8.0
        timeout = self.step_count >= self.recover_deadline
        path_finished = (
            self._path is not None
            and np.linalg.norm(self._path[-1] - robot_xy) < 8.0
        )
        if close or timeout or path_finished:
            self.RECOVERING = False
            self.recover_goal = None
            self.last_goal = None
            self._path = None
            self.cooldown_until = self.step_count + self.COOLDOWN_STEPS
            self.last_goal_hold_until = max(
                self.last_goal_hold_until,
                self.step_count + max(0, self.GOAL_HOLD_STEPS // 2),
            )