"""Tests for validate_prices using synthetic panels (no network needed)."""

import numpy as np
import pandas as pd
import pytest

from backtest.data import validate_prices, _max_run_length


def make_panel(n_days=260, tickers=("AAA", "BBB"), seed=0):
    """Clean synthetic (Open, Close) panel with random-walk prices."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    cols = pd.MultiIndex.from_product([["Open", "Close"], tickers])
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, len(tickers))), axis=0))
    open_ = close * (1 + rng.normal(0, 0.002, close.shape))
    data = np.concatenate([open_, close], axis=1)
    return pd.DataFrame(data, index=dates, columns=cols)


def test_clean_panel_passes():
    report = validate_prices(make_panel())
    assert report.ok
    assert report.nan_counts == {}
    assert report.nonpositive_prices == {}
    assert report.extreme_returns.empty
    # bdate_range index has no gaps vs itself
    assert report.missing_business_days == []


def test_detects_negative_price():
    px = make_panel()
    px.iloc[10, px.columns.get_loc(("Close", "AAA"))] = -5.0
    report = validate_prices(px)
    assert not report.ok
    assert report.nonpositive_prices.get("Close:AAA") == 1


def test_detects_extreme_return():
    px = make_panel()
    # A 100% jump on one day
    col = px.columns.get_loc(("Close", "BBB"))
    px.iloc[50, col] = px.iloc[49, col] * 2.0
    report = validate_prices(px)
    hits = report.extreme_returns
    assert len(hits) >= 1
    assert "BBB" in set(hits["ticker"])


def test_detects_nan_runs():
    px = make_panel()
    idx = px.columns.get_loc(("Close", "AAA"))
    px.iloc[20:25, idx] = np.nan  # run of 5
    px.iloc[40, idx] = np.nan     # isolated
    report = validate_prices(px)
    assert report.nan_counts["AAA"] == 6
    assert report.max_nan_run["AAA"] == 5


def test_detects_missing_business_day_and_duplicates():
    px = make_panel()
    px = px.drop(px.index[7])                       # remove one business day
    px = pd.concat([px, px.iloc[[3]]]).sort_index()  # duplicate one date
    report = validate_prices(px)
    assert len(report.missing_business_days) == 1
    assert report.duplicated_dates == 1
    assert not report.ok


def test_max_run_length():
    s = pd.Series([False, True, True, False, True, True, True, False])
    assert _max_run_length(s) == 3
    assert _max_run_length(pd.Series([False, False])) == 0