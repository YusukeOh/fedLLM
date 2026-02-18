"""
Verify FRED-MD data availability for the 29 core variables (DEC-003).

Checks: column presence, missing-value fraction, time range, and fallback
substitution.  Run this after downloading a new FRED-MD vintage.

Usage:
    python scripts/check_data_availability.py
    python scripts/check_data_availability.py --csv data/raw/fred_md.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.fred_md import (
    CORE_VARIABLES,
    FALLBACK_VARIABLES,
    load_fred_md,
    validate_data_availability,
)


def main():
    parser = argparse.ArgumentParser(description="Check FRED-MD variable availability")
    parser.add_argument("--csv", default="./data/raw/fred_md.csv", help="Path to FRED-MD CSV")
    parser.add_argument("--no-transform", action="store_true", help="Check raw (untransformed) data")
    args = parser.parse_args()

    print("=" * 70)
    print("FRED-MD Data Availability Report — 29 Core Variables")
    print("=" * 70)

    core_ids = list(CORE_VARIABLES.keys())
    data, tcodes = load_fred_md(
        path=args.csv, transform=not args.no_transform, keep_columns=core_ids,
    )
    print(f"\nDataset shape: {data.shape[0]} months × {data.shape[1]} variables")
    print(f"Date range: {data.index.min().strftime('%Y-%m')} to {data.index.max().strftime('%Y-%m')}")
    print()

    available, report = validate_data_availability(
        data, verbose=False, fallbacks=FALLBACK_VARIABLES,
    )

    print(f"{'Variable':<22} {'Category':<15} {'Status':<30} {'First Obs':<12} {'Last Obs':<12} {'NA%':<8}")
    print("-" * 100)

    ok_count = fb_count = miss_count = 0

    for var_id in CORE_VARIABLES:
        info = CORE_VARIABLES[var_id]
        status = report[var_id]

        if status.startswith("ok"):
            col = var_id
            ok_count += 1
            status_str = "OK"
        elif status.startswith("fallback"):
            col = FALLBACK_VARIABLES.get(var_id)
            fb_count += 1
            status_str = f"FALLBACK → {col}"
        else:
            col = None
            miss_count += 1
            status_str = "MISSING"

        if col and col in data.columns:
            series = data[col].dropna()
            first_obs = series.index.min().strftime("%Y-%m") if len(series) > 0 else "N/A"
            last_obs = series.index.max().strftime("%Y-%m") if len(series) > 0 else "N/A"
            na_pct = f"{data[col].isna().mean():.1%}"
        else:
            first_obs = last_obs = na_pct = "N/A"

        print(f"{var_id:<22} {info['category']:<15} {status_str:<30} {first_obs:<12} {last_obs:<12} {na_pct:<8}")

    print("-" * 100)
    print(f"\nSummary: {ok_count} OK, {fb_count} fallback, {miss_count} missing / {len(CORE_VARIABLES)} total")

    # Category coverage
    print("\nCategory coverage:")
    categories = {}
    for var_id, info in CORE_VARIABLES.items():
        cat = info["category"]
        status = report[var_id]
        categories.setdefault(cat, {"total": 0, "available": 0})
        categories[cat]["total"] += 1
        if not status == "missing":
            categories[cat]["available"] += 1

    for cat, counts in categories.items():
        bar = "█" * counts["available"] + "░" * (counts["total"] - counts["available"])
        print(f"  {cat:<15} {bar} {counts['available']}/{counts['total']}")

    if miss_count > 0:
        print(f"\n⚠  {miss_count} variable(s) unavailable. Check FRED-MD vintage or add fallbacks.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
