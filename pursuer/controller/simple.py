import numpy as np
import cv2
from typing import Optional, Tuple


class PurePursuitController:
    def __init__(self, v_max: float = 13.0, w_max: float = 1.2, v_min: float = 5.8,
        theta_gain: float = 0.9, lookahead: float = 22.0, kappa_v_gain: float = 60.0,
        clear_slow_px: float = 8.0, clear_stop_px: float = 5.0, clear_push_gain: float = 22.0,):
        self.v_max = v_max
        self.w_max = w_max
        self.v_min = v_min
        self.theta_gain = theta_gain
        self.lookahead = lookahead
        self.kappa_v_gain = kappa_v_gain

        self.spin_ang_thresh: float = 0.9
        self.spin_min_steps: int = 18
        self.spin_w_max: float = 0.45
        self.spin_lookahead: float = 35.0
        self.spin_cooldown_steps: int = 35

        self._prev_ang: float = 0.0
        self._spin_streak: int = 0
        self._spin_cooldown: int = 0

        self._cursor_idx: int = 0
        self._last_path_sig: Optional[tuple] = None

        self._edt: Optional[np.ndarray] = None
        self._gx: Optional[np.ndarray] = None
        self._gy: Optional[np.ndarray] = None
        self.clear_slow_px = float(clear_slow_px)
        self.clear_stop_px = float(clear_stop_px)
        self.clear_push_gain = float(clear_push_gain)

    def set_clearance_maps(self, edt: np.ndarray):
        if edt is None:
            self._edt = None
            self._gx = None
            self._gy = None
            return
        self._edt = edt.astype(np.float32)
        self._gx = cv2.Sobel(self._edt, cv2.CV_32F, 1, 0, ksize=3)
        self._gy = cv2.Sobel(self._edt, cv2.CV_32F, 0, 1, ksize=3)

    def compute(self, robot_pose: np.ndarray, path: np.ndarray) -> Optional[Tuple[float, float]]:
        if path is None:
            return None
        if len(path) < 2:
            return None

        P = np.asarray(path, dtype=float)
        if P.ndim != 2 or P.shape[1] != 2:
            return None

        la_nominal = float(self.lookahead)
        la_eff = (
            max(la_nominal, self.spin_lookahead)
            if (self._spin_cooldown > 0)
            else la_nominal
        )
        la_eff = max(6.0, la_eff)

        sig = (
            len(P),
            int(P[0, 0]),
            int(P[0, 1]),
            int(P[-1, 0]),
            int(P[-1, 1]),
            int(P.sum()),
        )
        x, y, th = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
        rxy = np.array([x, y], dtype=float)
        if self._last_path_sig != sig:
            d0 = np.hypot(P[:, 0] - rxy[0], P[:, 1] - rxy[1])
            self._cursor_idx = int(np.argmin(d0))
            self._last_path_sig = sig

        d_all = np.hypot(P[:, 0] - rxy[0], P[:, 1] - rxy[1])
        i0 = int(np.argmin(d_all))
        self._cursor_idx = max(self._cursor_idx, i0)
        self._cursor_idx = min(self._cursor_idx, len(P) - 2)

        hx, hy = float(np.cos(th)), float(np.sin(th))

        best_idx = self._cursor_idx + 1
        acc = 0.0
        for i in range(self._cursor_idx, len(P) - 1):
            seg = float(np.linalg.norm(P[i + 1] - P[i]))
            acc += seg
            vx = float(P[i + 1, 0] - rxy[0])
            vy = float(P[i + 1, 1] - rxy[1])
            ahead = (vx * hx + vy * hy) >= 0.0
            if ahead:
                best_idx = i + 1
                if acc >= la_eff:
                    break
        target_point = P[best_idx]

        clr_for_speed = np.nan
        if self._edt is not None:
            row = int(np.clip(round(y), 0, self._edt.shape[0] - 1))
            col = int(np.clip(round(x), 0, self._edt.shape[1] - 1))
            clr_for_speed = float(self._edt[row, col])

        dx = float(target_point[0] - x)
        dy = float(target_point[1] - y)
        world_angle_to_target = np.arctan2(dy, dx)
        ang = world_angle_to_target - th
        ang = (ang + np.pi) % (2 * np.pi) - np.pi

        same_sign = (ang * self._prev_ang) > 0.0
        if abs(ang) > self.spin_ang_thresh and same_sign:
            self._spin_streak += 1
        else:
            self._spin_streak = 0
        self._prev_ang = ang

        breaking_spin = self._spin_streak >= self.spin_min_steps
        if breaking_spin:
            self._spin_cooldown = self.spin_cooldown_steps
        if self._spin_cooldown > 0:
            self._spin_cooldown -= 1

        w_cap = self.spin_w_max if self._spin_cooldown > 0 else self.w_max

        abs_ang = abs(ang)
        kappa = 2.0 * np.sin(abs_ang) / la_eff
        v_cap_kappa = self.kappa_v_gain / (abs(kappa) + 1e-6)

        v_err_scale = max(0.20, 1.0 - 1.5 * abs_ang / np.pi)
        v_pref = max(self.v_min, self.v_max * v_err_scale)

        if self._edt is not None and np.isfinite(clr_for_speed):
            if clr_for_speed <= self.clear_stop_px:
                if self._gx is not None and self._gy is not None:
                    gx = float(self._gx[row, col])
                    gy = float(self._gy[row, col])
                    nrm = (gx * gx + gy * gy) ** 0.5
                    if nrm > 1e-6:
                        push = (np.array([gx, gy]) / nrm) * self.clear_push_gain
                        push_xy = np.array([push[1], push[0]], dtype=float)
                        target_point = rxy + push_xy
                v_pref = max(0.6 * self.v_min, 0.0)
            else:
                clr_scale = max(
                    0.2, clr_for_speed / max(self.clear_slow_px, 1e-6)
                )
                v_pref *= clr_scale

        if abs_ang < 1e-3:
            v_wcap = self.v_max
        else:
            v_wcap = w_cap * la_eff / (2.0 * max(1e-6, np.sin(abs_ang)))

        v_cmd = min(v_pref, v_wcap, v_cap_kappa, self.v_max)
        v_cmd = max(self.v_min, v_cmd)

        w = np.clip(2.0 * v_cmd * np.sin(ang) / la_eff, -w_cap, w_cap)

        return float(v_cmd), float(w)