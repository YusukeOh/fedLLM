"""
VAR baseline for macroeconomic forecasting on FRED-MD.

Provides a simple wrapper around statsmodels VAR for consistent
evaluation against Time-LLM variants.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


class VARForecaster:
    """Fit a VAR model and produce multi-step forecasts.

    Parameters
    ----------
    max_lags : int or None
        Maximum lag order to consider.  If None, selected by AIC.
    ic : str
        Information criterion for lag selection (``"aic"``, ``"bic"``, ``"hqic"``).
    """

    def __init__(self, max_lags: int | None = 13, ic: str = "aic"):
        self.max_lags = max_lags
        self.ic = ic
        self._model = None
        self._result = None

    def fit(self, data: pd.DataFrame | np.ndarray) -> "VARForecaster":
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        model = VAR(data)
        if self.max_lags is not None:
            self._result = model.fit(maxlags=self.max_lags, ic=self.ic)
        else:
            self._result = model.fit(ic=self.ic)
        self._model = model
        return self

    def forecast(self, steps: int, last_obs: np.ndarray | None = None) -> np.ndarray:
        """Return (steps, n_vars) array of forecasts."""
        if self._result is None:
            raise RuntimeError("Call fit() first")
        if last_obs is None:
            last_obs = self._result.endog[-self._result.k_ar :]
        return self._result.forecast(last_obs, steps=steps)

    @property
    def selected_lag(self) -> int:
        if self._result is None:
            raise RuntimeError("Call fit() first")
        return self._result.k_ar


def rolling_var_forecast(
    data: pd.DataFrame,
    train_end: int,
    pred_len: int,
    n_windows: int,
    target_cols: Optional[list[str]] = None,
    max_lags: int = 13,
) -> tuple[np.ndarray, np.ndarray]:
    """Expanding-window VAR forecast for benchmarking.

    Returns
    -------
    actuals : (n_windows, pred_len, n_targets)
    preds   : (n_windows, pred_len, n_targets)
    """
    if target_cols is None:
        target_cols = list(data.columns)

    all_cols = list(data.columns)
    target_idx = [all_cols.index(c) for c in target_cols]

    actuals, preds = [], []
    for w in range(n_windows):
        t = train_end + w
        train_df = data.iloc[:t]
        var = VARForecaster(max_lags=max_lags)
        var.fit(train_df)
        fc = var.forecast(steps=pred_len)

        actual = data.iloc[t : t + pred_len].values
        if len(actual) < pred_len:
            break

        actuals.append(actual[:, target_idx])
        preds.append(fc[:, target_idx])

    return np.array(actuals), np.array(preds)
