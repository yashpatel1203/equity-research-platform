"""
Module 1d — Feature engineering.
Reads raw prices from stock_prices, computes all financial features,
and writes results to the stock_features table.

Features computed:
  Returns      : daily, log, 5-day, 21-day
  Volatility   : 21-day and 63-day rolling std
  Trend        : SMA-20, SMA-50, EMA-12, EMA-26, MACD, signal line
  Momentum     : RSI-14, Rate of Change 10
  Volume       : 20-day avg volume, volume ratio
  Risk         : rolling 63-day beta vs S&P 500
  Target       : next-day return direction (1=up, 0=down) — used in ML module

Usage:
    python 04_engineer_features.py
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")

from config import ALL_TICKERS
from schema_models import StockFeatures, create_database

load_dotenv()
DB_PATH = os.getenv("DB_PATH", "equity_data.db")


# ── Helper: technical indicators ─────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series):
    ema12   = series.ewm(span=12, adjust=False).mean()
    ema26   = series.ewm(span=26, adjust=False).mean()
    macd    = ema12 - ema26
    signal  = macd.ewm(span=9, adjust=False).mean()
    hist    = macd - signal
    return ema12, ema26, macd, signal, hist


def compute_beta(stock_returns: pd.Series, bench_returns: pd.Series, window: int = 63) -> pd.Series:
    """Rolling beta using covariance / variance method."""
    cov = stock_returns.rolling(window).cov(bench_returns)
    var = bench_returns.rolling(window).var()
    return cov / var.replace(0, np.nan)


# ── Main feature engineering ──────────────────────────────────────────────────

def engineer_features(ticker: str, engine) -> pd.DataFrame:
    """Load price data for one ticker and return a features DataFrame."""

    # Load stock prices
    with engine.connect() as conn:
        prices = pd.read_sql(
            text(f"SELECT * FROM stock_prices WHERE ticker = :t ORDER BY date"),
            conn, params={"t": ticker}, parse_dates=["date"]
        )
    if prices.empty or len(prices) < 70:
        return pd.DataFrame()

    # Load benchmark returns (for beta)
    with engine.connect() as conn:
        bench = pd.read_sql(
            text("SELECT date, daily_return FROM benchmark_prices ORDER BY date"),
            conn, parse_dates=["date"]
        )

    prices = prices.sort_values("date").set_index("date")
    bench  = bench.sort_values("date").set_index("date")

    close  = prices["adj_close"]
    volume = prices["volume"]

    feat = pd.DataFrame(index=prices.index)
    feat["ticker"] = ticker

    # ── Returns ──────────────────────────────────────────
    feat["daily_return"] = close.pct_change()
    feat["log_return"]   = np.log(close / close.shift(1))
    feat["return_5d"]    = close.pct_change(5)
    feat["return_21d"]   = close.pct_change(21)

    # ── Volatility ───────────────────────────────────────
    feat["volatility_21d"] = feat["daily_return"].rolling(21).std() * np.sqrt(252)
    feat["volatility_63d"] = feat["daily_return"].rolling(63).std() * np.sqrt(252)

    # ── Trend indicators ─────────────────────────────────
    feat["sma_20"]  = close.rolling(20).mean()
    feat["sma_50"]  = close.rolling(50).mean()

    ema12, ema26, macd, signal, hist = compute_macd(close)
    feat["ema_12"]      = ema12
    feat["ema_26"]      = ema26
    feat["macd"]        = macd
    feat["macd_signal"] = signal
    feat["macd_hist"]   = hist

    # ── Momentum ─────────────────────────────────────────
    feat["rsi_14"] = compute_rsi(close, 14)
    feat["roc_10"] = close.pct_change(10) * 100

    # ── Volume ───────────────────────────────────────────
    feat["volume_sma_20"] = volume.rolling(20).mean()
    feat["volume_ratio"]  = volume / feat["volume_sma_20"].replace(0, np.nan)

    # ── Beta ─────────────────────────────────────────────
    aligned = feat[["daily_return"]].join(bench[["daily_return"]], rsuffix="_bench")
    feat["beta_63d"] = compute_beta(
        aligned["daily_return"], aligned["daily_return_bench"], window=63
    )

    # ── Target variable ──────────────────────────────────
    next_day_return = close.pct_change().shift(-1)
    feat["target_direction"] = (next_day_return > 0).astype(int)

    feat = feat.reset_index()
    feat = feat.dropna(subset=["daily_return"])   # drop the very first row

    return feat


def save_features(df: pd.DataFrame, engine):
    if df.empty:
        return 0

    feature_cols = [
        "ticker", "date", "daily_return", "log_return", "return_5d", "return_21d",
        "volatility_21d", "volatility_63d", "sma_20", "sma_50", "ema_12", "ema_26",
        "macd", "macd_signal", "macd_hist", "rsi_14", "roc_10",
        "volume_sma_20", "volume_ratio", "beta_63d", "target_direction"
    ]

    df = df[feature_cols].copy()
    # Round floats to 6dp to keep DB lean
    float_cols = df.select_dtypes(include=[float]).columns
    df[float_cols] = df[float_cols].round(6)

    with engine.connect() as conn:
        rows = df.to_dict(orient="records")
        table = StockFeatures.__table__
        stmt = table.insert().prefix_with("OR IGNORE")
        conn.execute(stmt, rows)
        conn.commit()
    return len(rows)


def run_all(engine):
    print("\n--- Engineering features for all tickers ---")
    total = 0
    for i, ticker in enumerate(ALL_TICKERS, 1):
        print(f"  [{i}/{len(ALL_TICKERS)}] {ticker}...", end=" ")
        df = engineer_features(ticker, engine)
        if not df.empty:
            saved = save_features(df, engine)
            print(f"{saved} rows")
            total += saved
        else:
            print("skipped (insufficient data)")
    print(f"\nTotal feature rows saved: {total}")


def verify_features(engine):
    print("\n--- Feature verification (sample stats) ---")
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT ticker,
                   COUNT(*) as rows,
                   ROUND(AVG(daily_return)*100, 4) as avg_ret_pct,
                   ROUND(AVG(volatility_21d)*100, 2) as avg_vol_pct,
                   ROUND(AVG(rsi_14), 1) as avg_rsi,
                   ROUND(AVG(beta_63d), 3) as avg_beta,
                   ROUND(AVG(CAST(target_direction AS FLOAT))*100, 1) as pct_up_days
            FROM stock_features
            GROUP BY ticker
            ORDER BY ticker
        """))
        rows = result.fetchall()
        header = f"  {'Ticker':<8} {'Rows':>6}  {'Ret%':>8} {'Vol%':>8} {'RSI':>6} {'Beta':>7} {'%Up':>6}"
        print(header)
        print("  " + "-" * 55)
        for r in rows:
            print(f"  {r[0]:<8} {r[1]:>6}  {r[2]:>8} {r[3]:>8} {r[4]:>6} {r[5]:>7} {r[6]:>6}%")


if __name__ == "__main__":
    engine = create_database()
    run_all(engine)
    verify_features(engine)
    print("\nModule 1 complete! All data is in equity_data.db")
    print("Next step: run Module 2 — Risk Analytics (05_risk_analytics.py)")
