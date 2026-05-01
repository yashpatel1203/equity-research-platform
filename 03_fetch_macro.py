"""
Module 1c — Fetch macro indicators from FRED API.
Saves to the macro_data table.

You need a free FRED API key:
  1. Go to https://fred.stlouisfed.org/docs/api/api_key.html
  2. Create a free account and request a key
  3. Add it to your .env file: FRED_API_KEY=your_key_here

Usage:
    python 03_fetch_macro.py
"""

import pandas as pd
from fredapi import Fred
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

from config import MACRO_SERIES, START_DATE, END_DATE
from schema_models import MacroData, create_database

load_dotenv()
DB_PATH    = os.getenv("DB_PATH", "equity_data.db")
FRED_KEY   = os.getenv("FRED_API_KEY", "")


def fetch_fred_series(fred: Fred, series_name: str, series_id: str) -> pd.DataFrame:
    """Fetch a single FRED series and return as a tidy DataFrame."""
    try:
        series = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
        df = series.reset_index()
        df.columns = ["date", "value"]
        df["series_name"] = series_name
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.dropna(subset=["value"])
        return df[["series_name", "date", "value"]]
    except Exception as e:
        print(f"  ERROR fetching {series_name} ({series_id}): {e}")
        return pd.DataFrame()


def forward_fill_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    FRED series like CPI and unemployment are monthly.
    Forward-fill them to daily frequency so they can join with stock data.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Create a full daily date range
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    df = df.reindex(date_range).ffill()
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = df["date"].dt.date
    return df


def save_macro(df: pd.DataFrame, engine):
    if df.empty:
        return 0
    with engine.connect() as conn:
        rows = df.to_dict(orient="records")
        table = MacroData.__table__
        stmt = table.insert().prefix_with("OR IGNORE")
        conn.execute(stmt, rows)
        conn.commit()
    return len(rows)


def fetch_all_macro(engine):
    if not FRED_KEY or FRED_KEY == "your_fred_api_key_here":
        print("\nWARNING: No FRED API key found in .env file.")
        print("Generating synthetic macro data for development...\n")
        _generate_synthetic_macro(engine)
        return

    fred = Fred(api_key=FRED_KEY)
    print("\n--- Fetching macro data from FRED ---")
    total = 0

    for series_name, series_id in MACRO_SERIES.items():
        print(f"  Fetching {series_name} ({series_id})...", end=" ")
        df = fetch_fred_series(fred, series_name, series_id)
        if not df.empty:
            # Forward-fill monthly series to daily
            df = forward_fill_to_daily(df)
            saved = save_macro(df, engine)
            print(f"{saved} rows")
            total += saved
        else:
            print("failed")

    print(f"\nTotal macro rows saved: {total}")


def _generate_synthetic_macro(engine):
    """
    Fallback: generate realistic synthetic macro data when no FRED key is available.
    Replace with real data once you have your API key.
    """
    import numpy as np
    np.random.seed(42)

    dates = pd.date_range(START_DATE, END_DATE, freq="B")  # business days
    n = len(dates)

    series_data = []
    for series_name, (mean, std, drift) in {
        "vix":            (18, 5, 0),
        "fed_funds_rate": (2.0, 0.1, 0.001),
        "treasury_10y":   (2.5, 0.15, 0.0005),
        "treasury_2y":    (2.2, 0.15, 0.0005),
        "cpi":            (260, 2, 0.05),
        "unemployment":   (4.5, 0.3, 0),
        "gdp_growth":     (2.5, 1.5, 0),
    }.items():
        values = mean + std * np.random.randn(n) + drift * np.arange(n)
        for date, val in zip(dates, values):
            series_data.append({
                "series_name": series_name,
                "date": date.date(),
                "value": round(float(val), 4)
            })

    df = pd.DataFrame(series_data)
    saved = save_macro(df, engine)
    print(f"  Synthetic macro data generated: {saved} rows")
    print("  NOTE: Replace with real FRED data for your final project.\n")


def verify_macro(engine):
    print("\n--- Macro data verification ---")
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT series_name, COUNT(*) as rows,
                   MIN(date) as from_date, MAX(date) as to_date,
                   ROUND(AVG(value), 4) as avg_value
            FROM macro_data
            GROUP BY series_name
            ORDER BY series_name
        """))
        rows = result.fetchall()
        print(f"  {'Series':<20} {'Rows':>6}  {'From':<12} {'To':<12} {'Avg':>10}")
        print("  " + "-" * 65)
        for r in rows:
            print(f"  {r[0]:<20} {r[1]:>6}  {str(r[2]):<12} {str(r[3]):<12} {r[4]:>10}")


if __name__ == "__main__":
    engine = create_database()
    fetch_all_macro(engine)
    verify_macro(engine)
    print("\nModule 1c complete. Run 04_engineer_features.py next.")
