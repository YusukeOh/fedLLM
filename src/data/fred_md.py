"""
FRED-MD dataset loader for Time-LLM.

FRED-MD (McCracken & Ng, 2016) is a monthly panel of ~130 US macroeconomic
variables published by the Federal Reserve Bank of St. Louis.  Each series
carries a "transformation code" (tcode) that maps the raw level to a
stationary representation.

Reference
---------
McCracken, M. W. & Ng, S. (2016). FRED-MD: A Monthly Database for
Macroeconomic Research.  *Journal of Business & Economic Statistics*, 34(4).
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

FRED_MD_URL = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed"
    "/research/fred-md/monthly/2026-01-md.csv"
)

# tcode → transformation function
# 1: level, 2: Δx, 3: Δ²x, 4: log(x), 5: Δlog(x), 6: Δ²log(x), 7: Δ(x/x_{-1} - 1)
TCODE_MAP = {
    1: lambda x: x,
    2: lambda x: x.diff(),
    3: lambda x: x.diff().diff(),
    4: lambda x: np.log(x),
    5: lambda x: np.log(x).diff(),
    6: lambda x: np.log(x).diff().diff(),
    7: lambda x: x.pct_change(),
}


def apply_transformations(df: pd.DataFrame, tcodes: pd.Series) -> pd.DataFrame:
    """Apply FRED-MD transformation codes column-wise and drop initial NaN rows."""
    cols = {}
    for col in df.columns:
        tc = int(tcodes[col])
        fn = TCODE_MAP.get(tc, TCODE_MAP[1])
        cols[col] = fn(df[col])
    return pd.DataFrame(cols, index=df.index)


def interpolate_internal_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate *internal* NaN gaps in raw data (pre-transform).

    DEC-005: Prevents NaN propagation during tcode transformation (e.g. a
    single NaN in tcode=6 would expand to 3 NaN rows).  Trailing NaNs
    (jagged edges from publication lags) are preserved intentionally for
    the [UNPUB] token mechanism (DEC-006).
    """
    result = df.copy()
    for col in result.columns:
        s = result[col]
        first_valid = s.first_valid_index()
        last_valid = s.last_valid_index()
        if first_valid is None or last_valid is None:
            continue
        mask = (s.index >= first_valid) & (s.index <= last_valid)
        result.loc[mask, col] = s.loc[mask].interpolate(method="linear")
    return result


def winsorize(df: pd.DataFrame, k: float = 10.0) -> pd.DataFrame:
    """Winsorize each column at median ± k × IQR.

    DEC-005: Preserves direction and timing of extreme events (e.g.
    Volcker-era FEDFUNDS) while preventing StandardScaler distortion.
    """
    result = df.copy()
    for col in result.columns:
        s = result[col].dropna()
        if len(s) == 0:
            continue
        q25 = s.quantile(0.25)
        q75 = s.quantile(0.75)
        iqr = q75 - q25
        median = s.median()
        lower = median - k * iqr
        upper = median + k * iqr
        result[col] = result[col].clip(lower=lower, upper=upper)
    return result


def generate_publication_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a boolean mask: True where data is published, False where NaN.

    DEC-006: The [UNPUB] token mechanism in PatchEmbedding uses this mask
    to replace unpublished patches with a learnable embedding.
    """
    return ~df.isna()


def load_fred_md(
    path: str | Path | None = None,
    url: str = FRED_MD_URL,
    transform: bool = True,
    drop_na_threshold: float = 0.1,
    keep_columns: list[str] | None = None,
    start_date: str | None = None,
    pre_transform_interpolate: bool = True,
    winsorize_k: float | None = 10.0,
    preserve_trailing_nans: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess FRED-MD CSV following the DEC-005 pipeline.

    Pipeline order:
      1a. Pre-transform linear interpolation (internal NaN only)
      1b. tcode transformation (McCracken & Ng stationarity codes)
      2.  Start-date filter (default: 1978-02, all 29 core vars available)
      3.  Winsorization (median ± k×IQR)
      4.  Column filtering (drop_na_threshold + keep_columns)
      5.  Remaining gap fill (ffill/bfill) — trailing NaNs optionally preserved

    Parameters
    ----------
    path, url : CSV source.
    transform : Apply tcode transformations.
    drop_na_threshold : Drop columns exceeding this NA fraction.
    keep_columns : Columns to keep regardless of NA threshold.
    start_date : Trim data to start from this date (e.g. "1978-02").
    pre_transform_interpolate : Interpolate internal NaN gaps before tcode.
    winsorize_k : Winsorization threshold in IQR multiples (None to skip).
    preserve_trailing_nans : If True, keep trailing NaN for [UNPUB] mask.

    Returns
    -------
    data : pd.DataFrame
    tcodes : pd.Series
    """
    if path is not None and os.path.exists(path):
        raw = pd.read_csv(path)
    else:
        import subprocess

        if path is None:
            path = "/tmp/fred_md_current.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        subprocess.run(
            ["curl", "--http1.1", "-L", "--max-time", "60", "-o", path, url],
            check=True,
            capture_output=True,
        )
        raw = pd.read_csv(path)

    tcode_row = raw.iloc[0, 1:]
    tcodes = tcode_row.astype(float).astype(int)

    data = raw.iloc[1:].copy()
    data.rename(columns={data.columns[0]: "date"}, inplace=True)
    data["date"] = pd.to_datetime(data["date"], format="mixed")
    data = data.set_index("date")
    data = data.apply(pd.to_numeric, errors="coerce")

    # Step 1a: pre-transform interpolation (internal NaN only)
    if pre_transform_interpolate and transform:
        data = interpolate_internal_nans(data)

    # Step 1b: tcode transformation
    if transform:
        data = apply_transformations(data, tcodes)
        data = data.iloc[2:]  # drop rows lost to differencing

    # Step 2: start-date filter
    if start_date is not None:
        data = data.loc[start_date:]

    # Step 3: Winsorization
    if winsorize_k is not None:
        data = winsorize(data, k=winsorize_k)

    # Step 4: column filtering
    missing_frac = data.isna().mean()
    keep = missing_frac[missing_frac <= drop_na_threshold].index.tolist()
    if keep_columns:
        for col in keep_columns:
            if col in data.columns and col not in keep:
                keep.append(col)
    data = data[keep]

    # Step 5: fill remaining gaps (preserve trailing NaN if requested)
    if preserve_trailing_nans:
        for col in data.columns:
            s = data[col]
            last_valid = s.last_valid_index()
            if last_valid is not None:
                internal = s.loc[:last_valid]
                data.loc[:last_valid, col] = internal.ffill().bfill()
    else:
        data = data.ffill().bfill()

    return data, tcodes


# ── DEC-003: 29 core variables (discussions/002_fred_md_variable_selection.md) ──

CORE_VARIABLES: dict[str, dict] = {
    # Real economy (5)
    "INDPRO":            {"name": "Industrial Production Index",              "category": "real_economy", "tcode": 5},
    "DPCERA3M086SBEA":   {"name": "Real Personal Consumption Expenditures",  "category": "real_economy", "tcode": 5},
    "CUMFNS":            {"name": "Capacity Utilization: Manufacturing",      "category": "real_economy", "tcode": 2},
    "W875RX1":           {"name": "Real Personal Income ex Transfers",       "category": "real_economy", "tcode": 5},
    "CMRMTSPLx":         {"name": "Real Mfg & Trade Industries Sales",       "category": "real_economy", "tcode": 5},
    # Labor market (6)
    "PAYEMS":            {"name": "Total Nonfarm Payrolls",                  "category": "labor",        "tcode": 5},
    "UNRATE":            {"name": "Unemployment Rate",                       "category": "labor",        "tcode": 2},
    "CES0600000008":     {"name": "Avg Hourly Earnings: Goods-Producing",    "category": "labor",        "tcode": 6},
    "CLAIMSx":           {"name": "Initial Unemployment Claims",             "category": "labor",        "tcode": 5},
    "CLF16OV":           {"name": "Civilian Labor Force",                    "category": "labor",        "tcode": 5},
    "AWHMAN":            {"name": "Avg Weekly Hours: Manufacturing",         "category": "labor",        "tcode": 1},
    # Prices (6)
    "CPIAUCSL":          {"name": "CPI All Items",                           "category": "prices",       "tcode": 6},
    "CPIULFSL":          {"name": "CPI Less Food & Energy",                  "category": "prices",       "tcode": 6},
    "PCEPI":             {"name": "PCE Price Index",                         "category": "prices",       "tcode": 6},
    "OILPRICEx":         {"name": "Crude Oil Prices (WTI)",                  "category": "prices",       "tcode": 6},
    "DSERRG3M086SBEA":   {"name": "PCE Services Price Index",               "category": "prices",       "tcode": 6},
    "WPSFD49207":        {"name": "PPI: Finished Goods",                     "category": "prices",       "tcode": 6},
    # Housing (2)
    "HOUST":             {"name": "Housing Starts",                          "category": "housing",      "tcode": 4},
    "PERMIT":            {"name": "Building Permits",                        "category": "housing",      "tcode": 4},
    # Investment & Inventory (2)
    "ANDENOx":           {"name": "New Orders: Nondefense Cap Goods ex Aircraft", "category": "investment", "tcode": 5},
    "ISRATIOx":          {"name": "Total Business: Inventories/Sales Ratio", "category": "investment",   "tcode": 2},
    # Financial conditions (4)
    "FEDFUNDS":          {"name": "Federal Funds Rate",                      "category": "financial",    "tcode": 2},
    "GS10":              {"name": "10-Year Treasury Rate",                   "category": "financial",    "tcode": 2},
    "BAA":               {"name": "Moody's Baa Corporate Bond Yield",       "category": "financial",    "tcode": 2},
    "S&P 500":           {"name": "S&P 500 Index",                          "category": "financial",    "tcode": 5},
    # Exchange rate (1)
    "TWEXAFEGSMTHx":     {"name": "Trade-Weighted USD Index",               "category": "exchange",     "tcode": 5},
    # Money & Credit (2)
    "M2SL":              {"name": "M2 Money Stock",                          "category": "money_credit", "tcode": 6},
    "BUSLOANS":          {"name": "Commercial & Industrial Loans",           "category": "money_credit", "tcode": 6},
    # Sentiment (1)
    "UMCSENTx":          {"name": "U. Michigan Consumer Sentiment",          "category": "sentiment",    "tcode": 2},
}

FALLBACK_VARIABLES: dict[str, str] = {
    "ANDENOx": "AMDMNOx",
}

# Backward-compat alias
TARGET_VARIABLES = {k: v["name"] for k, v in CORE_VARIABLES.items()}


def get_core_variable_ids() -> list[str]:
    """Return the ordered list of 29 core FRED-MD variable IDs."""
    return list(CORE_VARIABLES.keys())


def validate_data_availability(
    data: pd.DataFrame,
    variables: list[str] | None = None,
    fallbacks: dict[str, str] | None = None,
    verbose: bool = True,
) -> tuple[list[str], dict[str, str]]:
    """Check which core variables exist in *data* and apply fallbacks.

    Returns
    -------
    available : list[str]
        Variable IDs present in *data* (possibly via fallback substitution).
    report : dict[str, str]
        Per-variable status: "ok", "fallback:<id>", or "missing".
    """
    if variables is None:
        variables = get_core_variable_ids()
    if fallbacks is None:
        fallbacks = FALLBACK_VARIABLES

    available: list[str] = []
    report: dict[str, str] = {}

    for var in variables:
        if var in data.columns:
            na_frac = data[var].isna().mean()
            available.append(var)
            report[var] = f"ok (na={na_frac:.1%})"
        elif var in fallbacks and fallbacks[var] in data.columns:
            fb = fallbacks[var]
            na_frac = data[fb].isna().mean()
            available.append(fb)
            report[var] = f"fallback:{fb} (na={na_frac:.1%})"
        else:
            report[var] = "missing"

    if verbose:
        ok = sum(1 for v in report.values() if v.startswith("ok"))
        fb = sum(1 for v in report.values() if v.startswith("fallback"))
        ms = sum(1 for v in report.values() if v == "missing")
        print(f"[FRED-MD] Variable availability: {ok} ok, {fb} fallback, {ms} missing / {len(variables)} total")
        for var, status in report.items():
            if not status.startswith("ok"):
                print(f"  {var}: {status}")

    return available, report


class FREDMDDataset(Dataset):
    """PyTorch Dataset wrapping FRED-MD for Time-LLM style forecasting.

    The interface mirrors Time-LLM's ``Dataset_Custom`` so it can be used as a
    drop-in replacement in the training loop.

    Each sample returns:
        seq_x   : (seq_len, 1)       input window for one variable
        seq_y   : (label_len + pred_len, 1)  target window
        seq_x_mark : (seq_len, 1)    month index  [0..11]
        seq_y_mark : (label_len + pred_len, 1)
    """

    def __init__(
        self,
        root_path: str = "./data/raw",
        flag: str = "train",
        size: Optional[list[int]] = None,
        features: str = "M",
        target: str = "INDPRO",
        scale: bool = True,
        freq: str = "m",
        percent: int = 100,
        use_core_variables: bool = True,
        start_date: str = "1978-02",
        winsorize_k: float = 10.0,
        **kwargs,
    ):
        if size is None:
            self.seq_len = 36
            self.label_len = 12
            self.pred_len = 12
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ("train", "val", "test")
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.freq = freq
        self.percent = percent
        self.root_path = root_path
        self.use_core_variables = use_core_variables
        self.start_date = start_date
        self.winsorize_k = winsorize_k

        self._read_data()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    # ── data loading ────────────────────────────────────────────

    def _read_data(self):
        csv_path = os.path.join(self.root_path, "fred_md.csv")
        core_ids = get_core_variable_ids() if self.use_core_variables else None
        data, _ = load_fred_md(
            path=csv_path,
            transform=True,
            keep_columns=core_ids,
            start_date=self.start_date,
            winsorize_k=self.winsorize_k,
            preserve_trailing_nans=False,
        )

        if self.use_core_variables:
            available_ids, _ = validate_data_availability(
                data, verbose=True, fallbacks=FALLBACK_VARIABLES,
            )
            data = data[[c for c in available_ids if c in data.columns]]

        # Generate and store publication mask before filling NaN
        self.pub_mask = generate_publication_mask(data)

        if self.target not in data.columns:
            candidate_ids = get_core_variable_ids() if self.use_core_variables else list(TARGET_VARIABLES.keys())
            available = [c for c in candidate_ids if c in data.columns]
            if available:
                self.target = available[0]

        cols = [c for c in data.columns if c != self.target] + [self.target]
        data = data[cols]

        # 70 / 10 / 20 split
        n = len(data)
        num_train = int(n * 0.7)
        num_test = int(n * 0.2)
        num_val = n - num_train - num_test

        border1s = [0, num_train - self.seq_len, n - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, n]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (
                (border2 - self.seq_len) * self.percent // 100 + self.seq_len
            )

        if self.features in ("M", "MS"):
            df_data = data
        else:  # "S"
            df_data = data[[self.target]]

        self.scaler = StandardScaler()
        if self.scale:
            train_data = df_data.iloc[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            values = self.scaler.transform(df_data.values)
        else:
            values = df_data.values

        # Month-of-year as the only time feature for monthly data
        dates = data.index[border1:border2]
        self.data_stamp = np.array([[d.month - 1] for d in dates], dtype=np.float32)

        self.data_x = values[border1:border2]
        self.data_y = values[border1:border2]

        self.column_names = list(df_data.columns)

    # ── Dataset interface ───────────────────────────────────────

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end, feat_id : feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id : feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return (
            seq_x.astype(np.float32),
            seq_y.astype(np.float32),
            seq_x_mark,
            seq_y_mark,
        )

    def __len__(self):
        return self.tot_len * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
