"""
Temporal alignment of FOMC documents with FRED-MD forecast vintages.

Phase 1 task 1-5: For any forecast origin date t, determine which FOMC
documents were *publicly available* at time t.  This is critical for
avoiding look-ahead bias.

Key rules
---------
- **Statements** are released on the meeting day → available from t = meeting_date.
- **Minutes** are released ~3 weeks after the meeting → available from t = release_date.
- SEP (Summary of Economic Projections) follows the same rule as minutes
  (released with a lag).

Usage pattern (Phase 2)
-----------------------
For each FRED-MD monthly observation at date t, the Prompt-as-Prefix module
retrieves the most recent FOMC statement that satisfies release_date <= t.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

MINUTES_RELEASE_LAG_DAYS = 21  # ~3 weeks typical delay


@dataclass
class AlignedDocument:
    """An FOMC document aligned to a forecast date."""

    forecast_date: date
    doc_type: str
    meeting_date: date
    release_date: date
    file_path: str
    staleness_days: int   # days between release and forecast date


def build_fomc_calendar(
    fomc_dir: str | Path,
    minutes_lag_days: int = MINUTES_RELEASE_LAG_DAYS,
) -> pd.DataFrame:
    """Build a calendar of FOMC documents with release dates.

    Reads metadata JSON files from the FOMC directory and constructs
    a DataFrame with columns: doc_type, meeting_date, release_date, file_path.
    """
    fomc_dir = Path(fomc_dir)
    records = []

    for sub in ("statements", "minutes"):
        sub_dir = fomc_dir / sub
        if not sub_dir.exists():
            continue

        for meta_file in sorted(sub_dir.glob("*.json")):
            meta = json.loads(meta_file.read_text())
            meeting_date = date.fromisoformat(meta["meeting_date"])

            if sub == "statements":
                release_date = meeting_date
            else:
                # Minutes release date: meeting + lag.  If actual release
                # date is in metadata, prefer it.
                release_str = meta.get("release_date")
                if release_str and release_str != meta["meeting_date"]:
                    release_date = date.fromisoformat(release_str)
                else:
                    release_date = meeting_date + timedelta(days=minutes_lag_days)

            text_file = meta_file.with_suffix(".txt")
            records.append({
                "doc_type": sub.rstrip("s"),
                "meeting_date": meeting_date,
                "release_date": release_date,
                "file_path": str(text_file),
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("release_date").reset_index(drop=True)
    logger.info("Built FOMC calendar: %d documents", len(df))
    return df


def align_to_forecast_dates(
    forecast_dates: list[date] | pd.DatetimeIndex,
    fomc_calendar: pd.DataFrame,
    doc_type: str = "statement",
    max_staleness_days: Optional[int] = None,
) -> list[Optional[AlignedDocument]]:
    """For each forecast date, find the most recent available FOMC document.

    Parameters
    ----------
    forecast_dates : sequence of dates
        Monthly forecast origin dates (e.g. the last day of each FRED-MD month).
    fomc_calendar : pd.DataFrame
        Output of `build_fomc_calendar`.
    doc_type : str
        "statement" or "minute".
    max_staleness_days : int, optional
        If the most recent document is older than this, return None instead.

    Returns
    -------
    list of AlignedDocument or None for each forecast date.
    """
    sub = fomc_calendar[fomc_calendar["doc_type"] == doc_type].copy()
    if sub.empty:
        return [None] * len(forecast_dates)

    sub = sub.sort_values("release_date")
    release_dates = sub["release_date"].tolist()

    aligned: list[Optional[AlignedDocument]] = []

    for fd in forecast_dates:
        if isinstance(fd, pd.Timestamp):
            fd = fd.date()

        # Binary search for the most recent release_date <= fd
        best_idx = None
        for i, rd in enumerate(release_dates):
            if rd <= fd:
                best_idx = i
            else:
                break

        if best_idx is None:
            aligned.append(None)
            continue

        row = sub.iloc[best_idx]
        staleness = (fd - row["release_date"]).days

        if max_staleness_days is not None and staleness > max_staleness_days:
            aligned.append(None)
            continue

        aligned.append(AlignedDocument(
            forecast_date=fd,
            doc_type=doc_type,
            meeting_date=row["meeting_date"],
            release_date=row["release_date"],
            file_path=row["file_path"],
            staleness_days=staleness,
        ))

    n_matched = sum(1 for a in aligned if a is not None)
    logger.info(
        "Aligned %d/%d forecast dates to %s documents (max staleness: %s days)",
        n_matched, len(forecast_dates), doc_type,
        max_staleness_days or "unlimited",
    )
    return aligned


def load_aligned_texts(
    aligned_docs: list[Optional[AlignedDocument]],
) -> list[Optional[str]]:
    """Load the text content for a list of aligned documents."""
    texts = []
    for doc in aligned_docs:
        if doc is None:
            texts.append(None)
        else:
            p = Path(doc.file_path)
            if p.exists():
                texts.append(p.read_text().strip())
            else:
                logger.warning("Missing file: %s", doc.file_path)
                texts.append(None)
    return texts
