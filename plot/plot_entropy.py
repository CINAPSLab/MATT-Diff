from __future__ import annotations

import os
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt



NPZ_SPECS: List[Tuple[str, str]] = [
    ("results/dp/dp_ep004.npz", "MATT-diff"),
    ("results/explore/frontier_ep004.npz", "Frontier-based"),
    ("results/track/track_ep004.npz", "Time-based"),
]

OUT_PATH = "results/plots/entropy.png"

SIGMA_CAP = 250.0
SMOOTH_WINDOW = 1
TMAX: int | None = None
X_PAD = 10
MIN_COUNT = 1

PLANNER_COLORS: Dict[str, str] = {
    "Frontier-based": "#1f77b4",
    "MATT-diff":      "#2ca02c",
    "Time-based":     "#9467bd",
}
DEFAULT_COLOR = "#7f7f0f"
DEFAULT_LINEWIDTH = 2.0
DEFAULT_ALPHA = 0.95

EPS = 1e-9



def clip_cov2d(S: np.ndarray,
               sigma_min: float = 1e-3,
               sigma_cap: float = SIGMA_CAP) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    if S.shape != (2, 2):
        S = np.array(S, dtype=float).reshape(2, 2)
    vals, vecs = np.linalg.eigh(S)
    vals = np.clip(vals, sigma_min ** 2, sigma_cap ** 2)
    return (vecs @ np.diag(vals) @ vecs.T).astype(float)


def entropy_from_cov2d(S: np.ndarray, sigma_cap: float = SIGMA_CAP) -> float:
    S_clip = clip_cov2d(S, sigma_cap=sigma_cap) + EPS * np.eye(2)
    sign, logdet = np.linalg.slogdet(S_clip)
    if sign <= 0:
        vals = np.linalg.eigvalsh(S_clip)
        logdet = float(np.sum(np.log(vals + EPS)))
    return float(np.log(2.0 * np.pi * np.e) + 0.5 * logdet)


def moving_average_nan_safe(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    x = np.asarray(x, dtype=float)
    mask = ~np.isnan(x)
    num = np.convolve(np.where(mask, x, 0.0), np.ones(k), mode="same")
    den = np.convolve(mask.astype(float), np.ones(k), mode="same")
    y = np.full_like(x, np.nan)
    np.divide(num, den, out=y, where=den > 0)
    return y


def per_step_entropy(mu: np.ndarray,
                     Sigma: np.ndarray,
                     exist_mask: np.ndarray,
                     sigma_cap: float = SIGMA_CAP,
                     min_count: int = MIN_COUNT) -> np.ndarray:
    mu = np.asarray(mu)
    Sigma = np.asarray(Sigma)
    exist = np.asarray(exist_mask).astype(bool)

    T, K, _ = mu.shape
    H_t = np.full(T, np.nan, dtype=float)

    for t in range(T):
        idx = exist[t]
        n = int(np.sum(idx))
        if n < int(min_count):
            continue
        S_t = Sigma[t, idx]
        ent_vals = [
            entropy_from_cov2d(S_t[i], sigma_cap=sigma_cap)
            for i in range(n)
        ]
        if ent_vals:
            H_t[t] = float(np.mean(ent_vals))

    return H_t


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as d:
        return d["mu"], d["Sigma"], d["exist_mask"]



def main() -> None:
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)

    plt.figure(figsize=(12, 4.0))
    any_series = False

    for path, label in NPZ_SPECS:
        if not os.path.exists(path):
            print(f"[WARN] missing: {path}")
            continue

        mu, S, em = load_npz(path)
        H = per_step_entropy(mu, S, em, sigma_cap=SIGMA_CAP, min_count=MIN_COUNT)
        t = np.arange(len(H))

        color = PLANNER_COLORS.get(label, DEFAULT_COLOR)
        y = moving_average_nan_safe(H, SMOOTH_WINDOW)

        plt.plot(
            t,
            y,
            label=label,
            color=color,
            lw=DEFAULT_LINEWIDTH,
            alpha=DEFAULT_ALPHA,
        )
        print(f"[{label}] mean Entropy={np.nanmean(H):.4f}")
        any_series = True

    if not any_series:
        print("[ERROR] no valid inputs")
        return

    if TMAX is not None:
        left = -int(max(0, X_PAD))
        plt.xlim(left, int(TMAX))

    plt.xlabel("time step")
    plt.ylabel("Entropy")
    plt.grid(alpha=0.35, linestyle="--")
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=240)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()