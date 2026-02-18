"""
Run VAR baseline on FRED-MD and print evaluation metrics.

Usage
-----
    python scripts/run_var_baseline.py
    python scripts/run_var_baseline.py --pred_len 6 --target CPIAUCSL
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.fred_md import TARGET_VARIABLES, load_fred_md
from src.baselines.var_model import VARForecaster, rolling_var_forecast
from src.evaluation.metrics import compute_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/raw/fred_md.csv")
    parser.add_argument("--target", type=str, default="INDPRO")
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--max_lags", type=int, default=13)
    parser.add_argument("--n_vars", type=int, default=8,
                        help="Number of variables to include in VAR (top target vars)")
    args = parser.parse_args()

    print("Loading FRED-MD ...")
    data, tcodes = load_fred_md(path=args.data_path, transform=True)
    print(f"  Shape: {data.shape} | Period: {data.index[0]} – {data.index[-1]}")

    available_targets = [v for v in TARGET_VARIABLES if v in data.columns]
    if args.target in data.columns:
        target = args.target
    else:
        target = available_targets[0]
        print(f"  Warning: {args.target} not found. Using {target}.")

    var_cols = [c for c in available_targets if c in data.columns][: args.n_vars]
    if target not in var_cols:
        var_cols[-1] = target
    subset = data[var_cols].dropna()
    print(f"  VAR variables: {var_cols}")
    print(f"  Usable rows: {len(subset)}")

    n = len(subset)
    train_end = int(n * 0.7)
    n_test = int(n * 0.2)
    n_windows = n_test - args.pred_len + 1

    print(f"\nRunning expanding-window VAR (lag selection by AIC, max={args.max_lags}) ...")
    print(f"  Train end index: {train_end} | Test windows: {n_windows} | Horizon: {args.pred_len}")

    actuals, preds = rolling_var_forecast(
        subset, train_end, args.pred_len, n_windows,
        target_cols=[target], max_lags=args.max_lags,
    )

    actuals_flat = actuals.reshape(-1)
    preds_flat = preds.reshape(-1)
    train_vals = subset[target].iloc[:train_end].values

    metrics = compute_all(actuals_flat, preds_flat, y_train=train_vals, seasonality=12)
    print(f"\n=== VAR Baseline | Target: {target} | h={args.pred_len} ===")
    for k, v in metrics.items():
        print(f"  {k:>6s}: {v:.6f}")

    os.makedirs("results", exist_ok=True)
    out_path = f"results/var_baseline_{target}_h{args.pred_len}.csv"
    pd.DataFrame([metrics]).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
