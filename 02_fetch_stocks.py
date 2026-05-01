"""
Module 1b — Fetch stock & benchmark prices from Yahoo Finance.
Saves raw OHLCV data to the stock_prices and benchmark_prices tables.

Usage:
    python 02_fetch_stocks.py
"""

import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from dotenv import load_dotenv
import os
import time

from config import ALL_TICKERS, BENCHMARK, START_DATE, END_DATE
from schema_models import StockPrice, BenchmarkPrice, create_database

load_dotenv()
DB_PATH = os.getenv("DB_PATH", "equity_data.db")


def fetch_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data for a single ticker. Compatible with yfinance 0.2.40+"""
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end, auto_adjust=False)

        if df is None or df.empty:
            print(f"  WARNING: No data returned for {ticker}")
            return pd.DataFrame()

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Normalize column names to lowercase
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # adj_close may be missing in some versions
        if "adj_close" not in df.columns and "close" in df.columns:
            df["adj_close"] = df["close"]

        df["ticker"] = ticker
        df.index.name = "date"
        df = df.reset_index()

        # Strip timezone info from date
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.date

        cols = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
        existing = [c for c in cols if c in df.columns]
        return df[existing]

    except Exception as e:
        print(f"  ERROR fetching {ticker}: {e}")
        return pd.DataFrame()


def save_prices(df: pd.DataFrame, engine, table_model, extra_cols=None):
    """Upsert rows — skip duplicates on (ticker, date) constraint."""
    if df.empty:
        return 0

    with engine.connect() as conn:
        # Use INSERT OR IGNORE for SQLite upsert
        cols = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
        if extra_cols:
            cols = extra_cols

        rows = df[cols].to_dict(orient="records")
        table = table_model.__table__
        stmt = table.insert().prefix_with("OR IGNORE")
        conn.execute(stmt, rows)
        conn.commit()
    return len(rows)


def fetch_all_stocks(engine):
    print("\n--- Fetching stock prices ---")
    total = 0
    for i, ticker in enumerate(ALL_TICKERS, 1):
        print(f"  [{i}/{len(ALL_TICKERS)}] {ticker}...", end=" ")
        df = fetch_ticker(ticker, START_DATE, END_DATE)
        if not df.empty:
            saved = save_prices(df, engine, StockPrice)
            print(f"{saved} rows saved")
            total += saved
        else:
            print("skipped")
        time.sleep(0.3)   # be polite to the API

    print(f"\nTotal stock rows saved: {total}")


def fetch_benchmark(engine):
    print("\n--- Fetching S&P 500 benchmark ---")
    df = fetch_ticker(BENCHMARK, START_DATE, END_DATE)
    if df.empty:
        print("ERROR: Could not fetch benchmark data")
        return

    # Calculate daily return for beta calculations later
    df = df.sort_values("date")
    df["daily_return"] = df["adj_close"].pct_change()

    with engine.connect() as conn:
        rows = df[["date", "close", "adj_close", "daily_return"]].to_dict(orient="records")
        table = BenchmarkPrice.__table__
        stmt = table.insert().prefix_with("OR IGNORE")
        conn.execute(stmt, rows)
        conn.commit()

    print(f"  Benchmark rows saved: {len(df)}")


def verify_data(engine):
    print("\n--- Data verification ---")
    with engine.connect() as conn:
        # Row count per ticker
        result = conn.execute(text("""
            SELECT ticker, COUNT(*) as rows,
                   MIN(date) as from_date,
                   MAX(date) as to_date
            FROM stock_prices
            GROUP BY ticker
            ORDER BY ticker
        """))
        rows = result.fetchall()
        print(f"  {'Ticker':<8} {'Rows':>6}  {'From':<12} {'To':<12}")
        print("  " + "-" * 42)
        for r in rows:
            print(f"  {r[0]:<8} {r[1]:>6}  {str(r[2]):<12} {str(r[3]):<12}")

        # Benchmark
        result = conn.execute(text("SELECT COUNT(*) FROM benchmark_prices"))
        print(f"\n  Benchmark rows: {result.scalar()}")


if __name__ == "__main__":
    engine = create_database()
    fetch_all_stocks(engine)
    fetch_benchmark(engine)
    verify_data(engine)
    print("\nModule 1b complete. Run 03_fetch_macro.py next.")