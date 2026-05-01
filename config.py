"""
Stock universe — 20 large-cap S&P 500 stocks across 5 sectors.
Edit this file to change which stocks the pipeline fetches.
"""

STOCKS = {
    "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL"],
    "Financials":  ["JPM", "BAC", "GS", "MS"],
    "Healthcare":  ["JNJ", "UNH", "PFE", "ABBV"],
    "Energy":      ["XOM", "CVX", "COP", "SLB"],
    "Consumer":    ["AMZN", "TSLA", "HD", "MCD"],
}

# Flatten to a simple list for use in other modules
ALL_TICKERS = [t for tickers in STOCKS.values() for t in tickers]

# Benchmark
BENCHMARK = "^GSPC"   # S&P 500

# Date range
START_DATE = "2019-01-01"
END_DATE   = "2024-12-31"

# FRED macro series to fetch
MACRO_SERIES = {
    "vix":            "VIXCLS",       # CBOE Volatility Index
    "fed_funds_rate": "FEDFUNDS",     # Federal Funds Rate
    "treasury_10y":   "DGS10",        # 10-Year Treasury Yield
    "treasury_2y":    "DGS2",         # 2-Year Treasury Yield
    "cpi":            "CPIAUCSL",     # Consumer Price Index
    "unemployment":   "UNRATE",       # Unemployment Rate
    "gdp_growth":     "A191RL1Q225SBEA", # Real GDP Growth Rate
}
