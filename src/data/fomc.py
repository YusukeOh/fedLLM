"""
FOMC document collector — statements and minutes.

Scrapes FOMC statements (1994–present) and minutes (1993–present) from the
Federal Reserve Board website.  Each document is stored as a plain-text file
tagged with the meeting date, enabling temporal alignment with FRED-MD data.

Implementation plan (Phase 1, tasks 1-2 & 1-3):
  https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

Look-ahead bias protocol
------------------------
Only the *release date* (not the meeting date) determines eligibility for a
given forecast vintage.  Statements are released on the meeting day; minutes
are released ~3 weeks after the meeting.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────

FRB_BASE = "https://www.federalreserve.gov"
CALENDAR_URL = f"{FRB_BASE}/monetarypolicy/fomccalendars.htm"

HEADERS = {
    "User-Agent": (
        "fedLLM-research/0.1 (academic research; "
        "https://github.com/fedLLM; contact: research@example.com)"
    ),
}

REQUEST_DELAY = 1.0  # polite delay between requests (seconds)


# ── Data structures ────────────────────────────────────────────


@dataclass
class FOMCDocument:
    """A single FOMC statement or minutes document."""

    doc_type: str                # "statement" or "minutes"
    meeting_date: date           # date of the FOMC meeting
    release_date: date           # date the document was publicly released
    url: str                     # source URL
    text: str = ""               # raw extracted text
    metadata: dict = field(default_factory=dict)

    @property
    def filename(self) -> str:
        return f"{self.doc_type}_{self.meeting_date.isoformat()}.txt"

    @property
    def meta_filename(self) -> str:
        return f"{self.doc_type}_{self.meeting_date.isoformat()}.json"


# ── Meeting index scraper ──────────────────────────────────────


def fetch_meeting_urls(
    start_year: int = 1994,
    end_year: Optional[int] = None,
    session: Optional[requests.Session] = None,
) -> list[dict]:
    """Scrape the FOMC calendar pages to build a list of meeting entries.

    Returns a list of dicts with keys:
        meeting_date, statement_url, minutes_url
    """
    if end_year is None:
        end_year = datetime.now().year

    sess = session or requests.Session()
    sess.headers.update(HEADERS)

    entries: list[dict] = []

    # Historical pages exist per-year up to 2020; 2021+ is on the shared calendar page
    seen_urls: set[str] = set()
    for year in range(start_year, end_year + 1):
        if year >= 2021:
            url = f"{FRB_BASE}/monetarypolicy/fomccalendars.htm"
        else:
            url = f"{FRB_BASE}/monetarypolicy/fomchistorical{year}.htm"

        if url in seen_urls:
            continue
        seen_urls.add(url)

        logger.info("Fetching FOMC calendar: %s", url)
        try:
            resp = sess.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("Failed to fetch %s: %s", url, e)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        page_entries = _parse_calendar_page(soup, year)
        # For the shared calendar page, filter to the requested year range
        if year >= 2021:
            page_entries = [
                e for e in page_entries
                if start_year <= e["meeting_date"].year <= end_year
            ]
        entries.extend(page_entries)
        time.sleep(REQUEST_DELAY)

    logger.info("Found %d FOMC meeting entries (%d–%d)", len(entries), start_year, end_year)
    return entries


def _parse_calendar_page(soup: BeautifulSoup, year: int) -> list[dict]:
    """Extract meeting dates and document URLs from a calendar page.

    Handles three eras of Fed website URL patterns:
      - Pre-2005:  /fomc/YYYYMMDDdefault.htm  &  /fomc/MINUTES/YYYY/YYYYMMDDmin.htm
      - 2005-2014: /newsevents/press/monetary/YYYYMMDDa.htm  &  /monetarypolicy/fomcminutes*.htm
      - 2015+:     /newsevents/pressreleases/monetary*  &  /monetarypolicy/fomcminutes*.htm
    """
    entries = []

    # --- Statement links ---
    stmt_patterns = [
        re.compile(r"monetary\d{8}a"),          # 2005+ press releases
        re.compile(r"/fomc/\d{8}default"),       # pre-2005
    ]
    stmt_map: dict[str, str] = {}
    for pattern in stmt_patterns:
        for link in soup.find_all("a", href=pattern):
            href = link.get("href", "")
            if href.endswith(".pdf"):
                continue
            m = re.search(r"(\d{8})", href)
            if m:
                full_url = href if href.startswith("http") else FRB_BASE + href
                stmt_map.setdefault(m.group(1), full_url)

    # Also pick up anchor-text-based detection (link text == "Statement")
    for link in soup.find_all("a", string=re.compile(r"^Statement$", re.I)):
        href = link.get("href", "")
        if not href or href.endswith(".pdf"):
            continue
        m = re.search(r"(\d{8})", href)
        if m:
            full_url = href if href.startswith("http") else FRB_BASE + href
            stmt_map.setdefault(m.group(1), full_url)

    # --- Minutes links ---
    mins_patterns = [
        re.compile(r"fomcminutes\d{8}"),         # 2005+ format
        re.compile(r"/fomc/MINUTES/\d{4}/\d{8}"), # pre-2005 format
        re.compile(r"minutes/\d{8}"),             # alternate
    ]
    mins_map: dict[str, str] = {}
    for pattern in mins_patterns:
        for link in soup.find_all("a", href=pattern):
            href = link.get("href", "")
            if href.endswith(".pdf"):
                continue
            m = re.search(r"(\d{8})", href)
            if m:
                full_url = href if href.startswith("http") else FRB_BASE + href
                mins_map.setdefault(m.group(1), full_url)

    # Also pick up anchor-text-based detection (link text == "Minutes")
    for link in soup.find_all("a", string=re.compile(r"^Minutes$", re.I)):
        href = link.get("href", "")
        if not href or href.endswith(".pdf"):
            continue
        m = re.search(r"(\d{8})", href)
        if m:
            full_url = href if href.startswith("http") else FRB_BASE + href
            mins_map.setdefault(m.group(1), full_url)

    all_dates = sorted(set(stmt_map.keys()) | set(mins_map.keys()))
    for date_str in all_dates:
        try:
            meeting_dt = datetime.strptime(date_str, "%Y%m%d").date()
        except ValueError:
            continue
        entries.append({
            "meeting_date": meeting_dt,
            "statement_url": stmt_map.get(date_str),
            "minutes_url": mins_map.get(date_str),
        })

    return entries


# ── Document text extraction ───────────────────────────────────


def fetch_document_text(url: str, session: Optional[requests.Session] = None) -> str:
    """Download an FOMC document page and extract the main text body."""
    sess = session or requests.Session()
    sess.headers.update(HEADERS)

    resp = sess.get(url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Try common content containers on the Fed site
    content = (
        soup.find("div", {"id": "article"})
        or soup.find("div", class_="col-xs-12 col-sm-8 col-md-8")
        or soup.find("div", id="content")
        or soup.find("article")
    )

    if content is None:
        content = soup.find("body")

    text = content.get_text(separator="\n", strip=True) if content else ""
    return _clean_text(text)


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove boilerplate from Fed website text."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = text.split("\n")
    # Strip common footer/header patterns
    cleaned = []
    for line in lines:
        if re.match(r"^(Last Update|Accessibility|Board of Governors.*Footer)", line, re.I):
            break
        cleaned.append(line.strip())
    return "\n".join(cleaned).strip()


# ── Batch collection pipeline ─────────────────────────────────


def collect_fomc_documents(
    output_dir: str | Path,
    doc_types: tuple[str, ...] = ("statement", "minutes"),
    start_year: int = 1994,
    end_year: Optional[int] = None,
    overwrite: bool = False,
) -> list[FOMCDocument]:
    """Main entry point: scrape and save FOMC documents to disk.

    Directory layout:
        output_dir/
            statements/
                statement_1994-02-04.txt
                statement_1994-02-04.json   (metadata)
            minutes/
                minutes_1994-02-04.txt
                minutes_1994-02-04.json
    """
    output_dir = Path(output_dir)
    for sub in ("statements", "minutes"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    sess = requests.Session()
    sess.headers.update(HEADERS)

    meetings = fetch_meeting_urls(start_year, end_year, session=sess)
    documents: list[FOMCDocument] = []

    for meeting in meetings:
        for doc_type in doc_types:
            # Normalize: accept both "statement"/"statements" and "minute"/"minutes"
            is_statement = doc_type.rstrip("s") == "statement"
            if is_statement:
                url = meeting.get("statement_url")
                sub_dir = "statements"
            else:
                url = meeting.get("minutes_url")
                sub_dir = "minutes"

            if url is None:
                continue

            doc = FOMCDocument(
                doc_type="statement" if is_statement else "minute",
                meeting_date=meeting["meeting_date"],
                release_date=meeting["meeting_date"],
                url=url,
            )

            text_path = output_dir / sub_dir / doc.filename
            meta_path = output_dir / sub_dir / doc.meta_filename

            if text_path.exists() and not overwrite:
                logger.debug("Skipping existing %s", text_path)
                doc.text = text_path.read_text()
                documents.append(doc)
                continue

            logger.info("Fetching %s for %s", doc_type, meeting["meeting_date"])
            try:
                doc.text = fetch_document_text(url, session=sess)
                text_path.write_text(doc.text)
                meta_path.write_text(json.dumps(asdict(doc, dict_factory=_json_factory), indent=2))
                documents.append(doc)
            except requests.RequestException as e:
                logger.warning("Failed to fetch %s: %s", url, e)

            time.sleep(REQUEST_DELAY)

    logger.info("Collected %d FOMC documents", len(documents))
    return documents


def _json_factory(pairs):
    """Custom dict factory for dataclasses.asdict — handles date serialization."""
    d = {}
    for k, v in pairs:
        if isinstance(v, date):
            d[k] = v.isoformat()
        elif k == "text":
            d[k] = f"[{len(v)} chars]"  # don't dump full text into metadata
        else:
            d[k] = v
    return d


# ── CLI entry point ────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Collect FOMC statements and minutes")
    parser.add_argument("--output-dir", default="./data/raw/fomc", help="Output directory")
    parser.add_argument("--start-year", type=int, default=1994)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--doc-types", nargs="+", default=["statements", "minutes"],
        choices=["statements", "minutes"],
    )
    args = parser.parse_args()

    docs = collect_fomc_documents(
        output_dir=args.output_dir,
        doc_types=tuple(args.doc_types),
        start_year=args.start_year,
        end_year=args.end_year,
        overwrite=args.overwrite,
    )
    print(f"\nCollected {len(docs)} documents → {args.output_dir}")
