"""Data layer: universe definition, price download with caching, and validation.

Universe choice and survivorship bias
-------------------------------------
Backtesting today's mega-caps (AAPL, NVDA, ...) over 2005-today is survivorship
bias: we would be selecting assets *because* we already know they survived and
grew, which inflates any backtest built on them.

Mitigation used here: SPDR sector ETFs as the cross-section. ETFs rebalance
their own constituents and do not "die" the way single stocks do (the failing
companies drop out of the index inside the ETF), so the survivorship problem
mostly vanishes at the asset level. Nine sectors is a small but sufficient
cross-section for a long-only cross-sectional momentum study.

A stricter alternative (not implemented): trade point-in-time S&P 500
constituents using the index's historical change log, so that on each date the
strategy only sees names that were actually in the index at that time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# SPDR Select Sector ETFs. All nine trade continuously from Dec 1998, so a
# 2005 start date gives every asset a full history (no partial-history assets
# to special-case in the engine).
UNIVERSE: list[str] = [
    "XLK",  # Technology
    "XLF",  # Financials
    "XLE",  # Energy
    "XLV",  # Health Care
    "XLI",  # Industrials
    "XLP",  # Consumer Staples
    "XLY",  # Consumer Discretionary
    "XLU",  # Utilities
    "XLB",  # Materials
]

DEFAULT_START = "2005-01-03"
CACHE_DIR = Path("data")


def get_prices(
    tickers: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str | None = None,
    cache_dir: Path = CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return daily Open and Close prices for the universe.

    Output shape: DataFrame with a 2-level column MultiIndex
    (field in {"Open", "Close"}, ticker), indexed by trading date.

    Prices are auto-adjusted by yfinance (splits and dividends folded in), so
    Close is total-return-consistent and Open is adjusted by the same factor,
    which keeps next-open execution consistent with close-based signals.

    Downloads once and caches to parquet; subsequent calls read the cache.
    """
    tickers = tickers if tickers is not None else UNIVERSE
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    cache_file = cache_dir / f"prices_{'-'.join(sorted(tickers))}_{start}_{end}.parquet"

    if cache_file.exists() and not force_refresh:
        logger.info("Loading cached prices from %s", cache_file)
        return pd.read_parquet(cache_file)

    import yfinance as yf  # local import: everything else works offline

    logger.info("Downloading %d tickers from %s to %s", len(tickers), start, end)
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no data; check tickers/network.")

    px = raw[["Open", "Close"]].copy()
    # Normalise column order: (field, ticker), tickers alphabetical.
    px = px.reindex(columns=sorted(tickers), level=1)
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)

    cache_dir.mkdir(parents=True, exist_ok=True)
    px.to_parquet(cache_file)
    logger.info("Cached %d rows to %s", len(px), cache_file)
    return px


@dataclass
class ValidationReport:
    """Everything validate_prices found, machine-readable for tests/logging."""

    n_rows: int = 0
    date_range: tuple[str, str] = ("", "")
    missing_business_days: list[pd.Timestamp] = field(default_factory=list)
    nan_counts: dict[str, int] = field(default_factory=dict)
    max_nan_run: dict[str, int] = field(default_factory=dict)
    nonpositive_prices: dict[str, int] = field(default_factory=dict)
    extreme_returns: pd.DataFrame | None = None  # (date, ticker, return)
    duplicated_dates: int = 0
    non_monotonic_index: bool = False

    @property
    def ok(self) -> bool:
        return (
            not self.nonpositive_prices
            and (self.extreme_returns is None or self.extreme_returns.empty)
            and self.duplicated_dates == 0
            and not self.non_monotonic_index
        )


def _max_run_length(mask: pd.Series) -> int:
    """Length of the longest consecutive run of True in a boolean Series."""
    if not mask.any():
        return 0
    groups = (~mask).cumsum()[mask]
    return int(groups.value_counts().max())


def validate_prices(
    px: pd.DataFrame,
    extreme_return_threshold: float = 0.50,
) -> ValidationReport:
    """Sanity-check a (field, ticker) price panel and log findings.

    Checks:
      1. index is unique, sorted, datetime
      2. missing days vs the NYSE-ish business-day calendar (Mon-Fri; reported
         gaps include US holidays, so a steady ~9-10/year is expected noise)
      3. NaN counts and the longest consecutive NaN run per ticker
      4. zero or negative prices (always an error)
      5. absurd single-day close-to-close returns (|r| > threshold), which on
         adjusted data usually mean an unadjusted split or a bad print

    Returns a ValidationReport; raises nothing. Caller decides what is fatal.
    """
    report = ValidationReport()
    close = px["Close"]

    # 1. Index integrity -----------------------------------------------------
    report.n_rows = len(px)
    report.duplicated_dates = int(px.index.duplicated().sum())
    report.non_monotonic_index = not px.index.is_monotonic_increasing
    if report.n_rows:
        report.date_range = (str(px.index[0].date()), str(px.index[-1].date()))

    # 2. Missing business days ----------------------------------------------
    expected = pd.bdate_range(px.index.min(), px.index.max())
    missing = expected.difference(px.index)
    report.missing_business_days = list(missing)

    # 3. NaNs ---------------------------------------------------------------
    for ticker in close.columns:
        nan_mask = close[ticker].isna()
        n = int(nan_mask.sum())
        if n:
            report.nan_counts[ticker] = n
            report.max_nan_run[ticker] = _max_run_length(nan_mask)

    # 4. Non-positive prices ------------------------------------------------
    for field_name in ("Open", "Close"):
        bad = (px[field_name] <= 0).sum()
        for ticker, count in bad[bad > 0].items():
            key = f"{field_name}:{ticker}"
            report.nonpositive_prices[key] = int(count)

    # 5. Extreme single-day returns -----------------------------------------
    rets = close.pct_change()
    hits = rets.abs() > extreme_return_threshold
    if hits.any().any():
        rows = []
        for ticker in rets.columns:
            for date in rets.index[hits[ticker]]:
                rows.append(
                    {"date": date, "ticker": ticker, "return": rets.at[date, ticker]}
                )
        report.extreme_returns = pd.DataFrame(rows)
    else:
        report.extreme_returns = pd.DataFrame(columns=["date", "ticker", "return"])

    _log_report(report)
    return report


def _log_report(r: ValidationReport) -> None:
    logger.info("Validated %d rows, %s to %s", r.n_rows, *r.date_range)
    if r.duplicated_dates:
        logger.error("%d duplicated dates in index", r.duplicated_dates)
    if r.non_monotonic_index:
        logger.error("Index is not sorted ascending")
    logger.info(
        "%d business days absent from index (US holidays account for ~9-10/yr)",
        len(r.missing_business_days),
    )
    if r.nan_counts:
        for ticker, n in r.nan_counts.items():
            logger.warning(
                "%s: %d NaN closes (longest run %d)",
                ticker, n, r.max_nan_run[ticker],
            )
    else:
        logger.info("No NaNs")
    if r.nonpositive_prices:
        logger.error("Non-positive prices: %s", r.nonpositive_prices)
    n_extreme = 0 if r.extreme_returns is None else len(r.extreme_returns)
    if n_extreme:
        logger.warning("%d extreme single-day returns:\n%s", n_extreme, r.extreme_returns)
    else:
        logger.info("No single-day returns beyond threshold")
    logger.info("Validation %s", "PASSED" if r.ok else "FAILED")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prices = get_prices()
    validate_prices(prices)