"""
Run VAR baselines across multiple targets and horizons, producing a
summary table that can be compared against Time-LLM results.

Usage
-----
    python scripts/run_all_baselines.py
"""

from __future__ import annotations

import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.fred_md import TARGET_VARIABLES, load_fred_md
from src.baselines.var_model import rolling_var_forecast
from src.evaluation.metrics import compute_all

HORIZONS = [1, 3, 6, 12]
DATA_PATH = "data/raw/fred_md.csv"
MAX_LAGS = 13
N_VAR_VARS = 8


def main():
    print("Loading FRED-MD ...")
    data, _ = load_fred_md(path=DATA_PATH, transform=True)
    available = [v for v in TARGET_VARIABLES if v in data.columns]
    print(f"  Available targets: {available}")

    var_cols = available[:N_VAR_VARS]
    subset = data[var_cols].dropna()
    n = len(subset)
    train_end = int(n * 0.7)

    all_results = []

    for target in available:
        for h in HORIZONS:
            n_test = int(n * 0.2)
            n_windows = n_test - h + 1
            if n_windows <= 0:
                continue

            print(f"  VAR | {target:>12s} | h={h:>2d} ... ", end="", flush=True)
            try:
                actuals, preds = rolling_var_forecast(
                    subset, train_end, h, n_windows,
                    target_cols=[target], max_lags=MAX_LAGS,
                )
                train_vals = subset[target].iloc[:train_end].values
                metrics = compute_all(
                    actuals.reshape(-1), preds.reshape(-1),
                    y_train=train_vals, seasonality=12,
                )
                metrics["target"] = target
                metrics["horizon"] = h
                metrics["model"] = "VAR"
                all_results.append(metrics)
                print(f"MSE={metrics['MSE']:.6f}  MAE={metrics['MAE']:.6f}")
            except Exception as e:
                print(f"FAILED: {e}")

    df = pd.DataFrame(all_results)
    cols = ["model", "target", "horizon", "MSE", "RMSE", "MAE", "MAPE", "sMAPE", "MASE"]
    df = df[[c for c in cols if c in df.columns]]

    os.makedirs("results", exist_ok=True)
    out_path = "results/var_baseline_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\n=== Summary saved to {out_path} ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
