from __future__ import annotations

import numpy as np

from typing import Any, Dict, List, Optional, Sequence, Tuple

EPS = 1e-9


def clip_cov2d(S: np.ndarray, sigma_min: float = 1e-3, sigma_cap: float = 250.0) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    if S.shape != (2, 2):
        S = np.array(S, dtype=float).reshape(2, 2)
    vals, vecs = np.linalg.eigh(S)
    vals = np.clip(vals, sigma_min ** 2, sigma_cap ** 2)
    return (vecs @ np.diag(vals) @ vecs.T).astype(float)


def nll_gauss2d(x: np.ndarray, mu: np.ndarray, S: np.ndarray, sigma_cap: float = 250.0) -> float:
    x = np.asarray(x, dtype=float).reshape(2)
    mu = np.asarray(mu, dtype=float).reshape(2)
    S = clip_cov2d(S, sigma_cap=sigma_cap) + EPS * np.eye(2)
    d = x - mu
    try:
        invS = np.linalg.inv(S)
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            vals = np.linalg.eigvalsh(S)
            logdet = float(np.sum(np.log(vals + EPS)))
    except np.linalg.LinAlgError:
        vals = np.linalg.eigvalsh(S)
        invS = np.linalg.pinv(S)
        logdet = float(np.sum(np.log(vals + EPS)))
    quad = float(d.T @ invS @ d)
    return 0.5 * (quad + logdet + 2.0 * np.log(2.0 * np.pi))


def entropy2d(S: np.ndarray, sigma_cap: float = 250.0) -> float:
    S = clip_cov2d(S, sigma_cap=sigma_cap) + EPS * np.eye(2)
    sign, logdet = np.linalg.slogdet(2.0 * np.pi * np.e * S)
    if sign <= 0:
        vals = np.linalg.eigvalsh(2.0 * np.pi * np.e * S)
        logdet = float(np.sum(np.log(vals + EPS)))
    return 0.5 * logdet


def compute_did_update(last_updates: np.ndarray) -> np.ndarray:
    L = np.asarray(last_updates)
    assert L.ndim == 2, f"last_updates must be (T,K), got {L.shape}"
    T, K = L.shape
    out = np.zeros((T, K), dtype=np.float32)
    for k in range(K):
        prev = -1
        for t in range(T):
            cur = int(L[t, k]) if np.isfinite(L[t, k]) else -1
            if cur >= 0 and cur != prev:
                out[t, k] = 1.0
            prev = cur
    return out


def evaluate_episode(
    mu: np.ndarray,
    Sigma: np.ndarray,
    x_true: np.ndarray,
    did_update: np.ndarray,
    exist_mask: Optional[np.ndarray] = None,
    *,
    sigma_cap: float = 250.0,
    rmse_all_censored: bool = False,
) -> Dict[str, float]:
    mu = np.asarray(mu, dtype=float)
    S  = np.asarray(Sigma, dtype=float)
    x  = np.asarray(x_true, dtype=float)
    U  = (np.asarray(did_update) > 0).astype(np.float32)

    assert mu.shape[:2] == x.shape[:2] == U.shape[:2]
    assert mu.shape[-1] == 2 and x.shape[-1] == 2
    assert S.shape[:2] == mu.shape[:2] and S.shape[-2:] == (2, 2)

    T, K, _ = mu.shape
    E = np.ones((T, K), dtype=np.float32) if (exist_mask is None) else ((np.asarray(exist_mask) > 0).astype(np.float32))

    valid_pair = (np.isfinite(x).all(axis=-1) & np.isfinite(mu).all(axis=-1)).astype(np.float32)

    detected_time_steps = int((U * E).sum())
    exist_time_steps    = int(E.sum())
    det_frac = float(detected_time_steps / float(max(exist_time_steps, 1)))

    se = ((x - mu) ** 2).sum(axis=-1)
    se = np.where(valid_pair > 0, se, 0.0)

    vis = (U * E) * valid_pair
    vis_count = int(vis.sum())
    if vis_count > 0:
        rmse_visible = float(np.sqrt((se * vis).sum() / float(vis_count)))
    else:
        exist_valid  = E * valid_pair
        rmse_visible = float(np.sqrt((se * exist_valid).sum() / float(max(int(exist_valid.sum()), 1))))

    exist_valid = E * valid_pair
    rmse_exist  = float(np.sqrt((se * exist_valid).sum() / float(max(int(exist_valid.sum()), 1))))

    out: Dict[str, float] = dict(
        detected_time_fraction=det_frac,
        detected_time_steps=detected_time_steps,
        exist_time_steps=exist_time_steps,
        vis_count=vis_count,
        rmse_visible=rmse_visible,
        rmse_exist=rmse_exist,
    )

    if rmse_all_censored:
        trS = np.zeros((T, K), dtype=float)
        for t in range(T):
            for i in range(K):
                trS[t, i] = float(np.trace(clip_cov2d(S[t, i], sigma_cap=sigma_cap)))
        se_all = ((x - mu) ** 2).sum(axis=-1) + trS
        out["rmse_all_censored"] = float(np.sqrt((se_all * E).sum() / float(max(E.sum(), 1))))

    nll_acc = 0.0
    H_acc   = 0.0
    valid_cnt = 0
    for t in range(T):
        for i in range(K):
            if E[t, i] <= 0:
                continue
            if not np.all(np.isfinite(x[t, i])):
                continue
            nll_acc += nll_gauss2d(x[t, i], mu[t, i], S[t, i], sigma_cap=sigma_cap)
            H_acc   += entropy2d(S[t, i], sigma_cap=sigma_cap)
            valid_cnt += 1

    denom = float(max(valid_cnt, 1))
    out["nll"] = float(nll_acc / denom)
    out["entropy"] = float(H_acc / denom)

    def _rmse_post_update(W: int) -> float:
        acc = 0.0; cnt = 0
        for t in range(T):
            for i in range(K):
                if U[t, i] > 0 and E[t, i] > 0:
                    t2 = min(T, t + W)
                    mask = valid_pair[t:t2, i].astype(bool)
                    if mask.any():
                        acc += float(se[t:t2, i][mask].sum())
                        cnt += int(mask.sum())
        return float(np.sqrt(acc / float(cnt))) if cnt > 0 else float('nan')

    out["rmse_post_update_W1"]  = _rmse_post_update(1)
    out["rmse_post_update_W5"]  = _rmse_post_update(5)
    out["rmse_post_update_W10"] = _rmse_post_update(10)

    return out

def pack_from_eval_states(
    eval_states_seq: Sequence[Sequence[Dict[str, Any]]],
    *,
    sigma_cap: float = 250.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    T = len(eval_states_seq)
    assert T > 0, "eval_states_seq is empty"
    K = len(eval_states_seq[0])
    mu = np.zeros((T, K, 2), dtype=float)
    S = np.zeros((T, K, 2, 2), dtype=float)
    x_true = np.zeros((T, K, 2), dtype=float)
    last = np.full((T, K), -1, dtype=int)
    exists = np.zeros((T, K), dtype=np.float32)
    last_mu = np.full((K, 2), np.nan, dtype=float)

    for t, states in enumerate(eval_states_seq):
        assert len(states) == K, f"step {t}: expected K={K} states, got {len(states)}"
        for i, s in enumerate(states):
            ex = bool(s.get("exists", True))
            exists[t, i] = 1.0 if ex else 0.0

            xt = np.asarray(s.get("x_true", [np.nan, np.nan]), dtype=float).reshape(2)
            mu_i = np.asarray(s.get("mu", [np.nan, np.nan]), dtype=float).reshape(2)
            if not np.all(np.isfinite(mu_i)):
                if np.all(np.isfinite(last_mu[i])):
                    mu_i = last_mu[i].copy()
                else:
                    mu_i = np.zeros(2, dtype=float)
            else:
                last_mu[i] = mu_i.copy()
            S_i = np.asarray(s.get("Sigma", np.full((2, 2), np.nan)), dtype=float).reshape(2, 2)
            lu = int(s.get("last_update_step", -1))

            if not np.all(np.isfinite(S_i)):
                S_i = np.diag([sigma_cap ** 2, sigma_cap ** 2])

            x_true[t, i] = xt
            mu[t, i] = mu_i
            S[t, i] = S_i
            last[t, i] = lu

    did = compute_did_update(last)
    return mu, S, x_true, did, exists


__all__ = [
    "clip_cov2d",
    "nll_gauss2d",
    "entropy2d",
    "compute_did_update",
    "evaluate_episode",
    "pack_from_eval_states",
]