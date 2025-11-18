from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

RESULT_JSONS: Dict[str, str] = {
    "BC":             "results/bc/bc.json",
    "Time-based":     "results/dp/dp.json",
    "Track":          "results/track/track_results.json",
    "Reacq":          "results/reacq/reacq_results.json",
    "Frontier-based": "results/explore/frontier_results.json",
}


def load_metrics(json_path: str) -> List[dict]:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["metrics"]


def compute_means(metrics: List[dict]) -> Dict[str, float]:
    rmse_vals = [m["rmse_exist"] for m in metrics]
    nll_vals = [m["nll"] for m in metrics]
    H_vals = [m["entropy"] for m in metrics]

    return {
        "rmse_exist": float(np.nanmean(rmse_vals)),
        "nll": float(np.nanmean(nll_vals)),
        "entropy": float(np.nanmean(H_vals)),
        "N": len(metrics),
    }


def main() -> None:
    print("=== Per-planner averaged metrics ===")
    for label, path in RESULT_JSONS.items():
        metrics = load_metrics(path)
        avg = compute_means(metrics)

        print(f"\n[{label}] from {os.path.basename(path)}  (N={avg['N']})")
        print("  mean rmse_exist : {:.3f}".format(avg["rmse_exist"]))
        print("  mean nll        : {:.3f}".format(avg["nll"]))
        print("  mean entropy    : {:.3f}".format(avg["entropy"]))

    print("\nDone.")


if __name__ == "__main__":
    main()