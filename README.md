# Equity Research & Portfolio Risk Analytics Platform

A full data analysis project for investment banking / capital markets roles.
Demonstrates: Python, Pandas, SQL, scikit-learn, and Power BI/Tableau.

---

## Project Structure

```
equity_project/
│
├── config.py                 # Stock universe, date range, macro series
├── schema_models.py          # SQLAlchemy table definitions
├── requirements.txt          # Python dependencies
├── .env.example              # Copy to .env and add your FRED API key
│
├── 01_schema.py              # (auto-called) Creates the SQLite database
├── 02_fetch_stocks.py        # Fetch OHLCV data from Yahoo Finance
├── 03_fetch_macro.py         # Fetch macro indicators from FRED
├── 04_engineer_features.py   # Compute all financial features
│
├── 05_risk_analytics.py      # [Module 2] VaR, Sharpe, drawdown, correlation
├── 06_ml_model.py            # [Module 3] ML signal model
└── equity_data.db            # SQLite database (created at runtime)
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your FRED API key
# Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html
```

### 3. Run the pipeline in order
```bash
python 02_fetch_stocks.py       # ~5 minutes
python 03_fetch_macro.py        # ~1 minute (or uses synthetic data without FRED key)
python 04_engineer_features.py  # ~2 minutes
```

---

## Database Schema

| Table              | Description                                |
|--------------------|--------------------------------------------|
| `stock_prices`     | Raw OHLCV prices for 20 S&P 500 stocks     |
| `benchmark_prices` | S&P 500 index prices                       |
| `macro_data`       | FRED macro indicators (VIX, rates, CPI...) |
| `stock_features`   | All engineered features + ML target        |

---

## Module Roadmap

| # | Module                    | Status      |
|---|---------------------------|-------------|
| 1 | Data Engineering + SQL    | ✅ Complete  |
| 2 | Risk Analytics            | 🔜 Next      |
| 3 | ML Signal Model           | 🔜 Upcoming  |
| 4 | Dashboard + Memo          | 🔜 Upcoming  |
