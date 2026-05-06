
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")

load_dotenv()
DB_PATH        = os.getenv("DB_PATH", "equity_data.db")
OUTPUT_DIR     = "risk_output"
TRADING_DAYS   = 252
RISK_FREE_RATE = 0.05
CONFIDENCE     = 0.95
PORTFOLIO_VALUE = 1_000_000

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_returns(engine) -> pd.DataFrame:
    with engine.connect() as conn:
        prices = pd.read_sql(
            text("SELECT ticker, date, adj_close FROM stock_prices ORDER BY date"),
            conn, parse_dates=["date"]
        )

    if prices.empty:
        raise ValueError("No data found. Run 02_fetch_stocks.py first.")

    pivot   = prices.pivot(index="date", columns="ticker", values="adj_close")
    returns = pivot.pct_change().dropna()
    print(f"  Loaded {len(returns)} trading days for {len(returns.columns)} tickers")
    print(f"  Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    return returns


def load_benchmark(engine) -> pd.Series:
    with engine.connect() as conn:
        bench = pd.read_sql(
            text("SELECT date, daily_return FROM benchmark_prices ORDER BY date"),
            conn, parse_dates=["date"]
        )
    return bench.set_index("date")["daily_return"].dropna()


# ── 2. Value at Risk (3 methods) ──────────────────────────────────────────────

def var_historical(returns: pd.Series, confidence: float = CONFIDENCE):
    """
    Historical Simulation VaR
    Sort actual past returns and read off the loss at the tail percentile.
    No assumptions about return distribution — uses real observed data.
    """
    r      = returns.dropna().sort_values()
    var    = -r.quantile(1 - confidence)
    tail   = r[r <= -var]
    cvar   = -tail.mean() if len(tail) > 0 else var
    return var, cvar


def var_parametric(returns: pd.Series, confidence: float = CONFIDENCE):
    mu, sig = returns.mean(), returns.std()
    z       = stats.norm.ppf(1 - confidence)
    var     = -(mu + z * sig)
    phi     = stats.norm.pdf(stats.norm.ppf(1 - confidence))
    cvar    = -(mu - sig * phi / (1 - confidence))
    return var, cvar


def var_monte_carlo(returns: pd.Series, confidence: float = CONFIDENCE,
                    n_sims: int = 10_000):

    mu, sig   = returns.mean(), returns.std()
    np.random.seed(42)
    simulated = np.random.normal(mu, sig, n_sims)
    var       = -np.percentile(simulated, (1 - confidence) * 100)
    cvar      = -simulated[simulated <= -var].mean()
    return var, cvar


def compute_all_var(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in returns.columns:
        r               = returns[ticker].dropna()
        h_var,  h_cvar  = var_historical(r)
        p_var,  p_cvar  = var_parametric(r)
        mc_var, mc_cvar = var_monte_carlo(r)
        rows.append({
            "ticker":          ticker,
            "hist_var_pct":    round(h_var   * 100, 3),
            "hist_cvar_pct":   round(h_cvar  * 100, 3),
            "param_var_pct":   round(p_var   * 100, 3),
            "param_cvar_pct":  round(p_cvar  * 100, 3),
            "mc_var_pct":      round(mc_var  * 100, 3),
            "mc_cvar_pct":     round(mc_cvar * 100, 3),
            "dollar_var_hist": int(h_var  * PORTFOLIO_VALUE / len(returns.columns)),
            "dollar_var_mc":   int(mc_var * PORTFOLIO_VALUE / len(returns.columns)),
        })
    return pd.DataFrame(rows).set_index("ticker")


# ── 3. Sharpe, Sortino, Calmar ────────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:

    excess = returns.mean() * TRADING_DAYS - rf
    vol    = returns.std()  * np.sqrt(TRADING_DAYS)
    return round(excess / vol, 4) if vol > 0 else np.nan


def sortino_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:

    excess       = returns.mean() * TRADING_DAYS - rf
    downside     = returns[returns < 0]
    downside_dev = downside.std() * np.sqrt(TRADING_DAYS)
    return round(excess / downside_dev, 4) if downside_dev > 0 else np.nan


def calmar_ratio(returns: pd.Series) -> float:
    ann_ret = returns.mean() * TRADING_DAYS
    mdd     = max_drawdown(returns)
    return round(ann_ret / abs(mdd), 4) if mdd != 0 else np.nan


# ── 4. Drawdown ───────────────────────────────────────────────────────────────

def max_drawdown(returns: pd.Series) -> float:

    cum         = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown    = (cum - rolling_max) / rolling_max
    return round(drawdown.min(), 4)


# ── 5. Beta & Alpha ───────────────────────────────────────────────────────────

def compute_beta(stock: pd.Series, bench: pd.Series) -> float:
    aligned = pd.concat([stock, bench], axis=1).dropna()
    if len(aligned) < 30:
        return np.nan
    cov = aligned.cov().iloc[0, 1]
    var = aligned.iloc[:, 1].var()
    return round(cov / var, 4) if var > 0 else np.nan


def compute_alpha(stock: pd.Series, bench: pd.Series,
                  beta: float, rf: float = RISK_FREE_RATE) -> float:
    ann_stock = stock.mean() * TRADING_DAYS
    ann_bench = bench.mean() * TRADING_DAYS
    return round(ann_stock - (rf + beta * (ann_bench - rf)), 4)


# ── 6. Build full summary ─────────────────────────────────────────────────────

def build_summary(returns: pd.DataFrame, bench: pd.Series) -> pd.DataFrame:
    rows = []
    for ticker in returns.columns:
        r    = returns[ticker].dropna()
        beta = compute_beta(r, bench)
        rows.append({
            "ticker":           ticker,
            "ann_return_pct":   round(r.mean() * TRADING_DAYS * 100, 2),
            "ann_vol_pct":      round(r.std()  * np.sqrt(TRADING_DAYS) * 100, 2),
            "sharpe":           sharpe_ratio(r),
            "sortino":          sortino_ratio(r),
            "calmar":           calmar_ratio(r),
            "max_drawdown_pct": round(max_drawdown(r) * 100, 2),
            "beta":             beta,
            "alpha_pct":        round(compute_alpha(r, bench, beta) * 100, 2) if not np.isnan(beta) else np.nan,
            "skewness":         round(r.skew(), 4),
            "kurtosis":         round(r.kurtosis(), 4),
        })
    return pd.DataFrame(rows).set_index("ticker")


def portfolio_metrics(returns: pd.DataFrame, bench: pd.Series) -> dict:
    port    = returns.mean(axis=1)
    h_var, h_cvar = var_historical(port)
    return {
        "ann_return_pct":      round(port.mean() * TRADING_DAYS * 100, 2),
        "ann_vol_pct":         round(port.std()  * np.sqrt(TRADING_DAYS) * 100, 2),
        "sharpe":              sharpe_ratio(port),
        "sortino":             sortino_ratio(port),
        "max_drawdown_pct":    round(max_drawdown(port) * 100, 2),
        "hist_var_95_pct":     round(h_var  * 100, 3),
        "hist_cvar_95_pct":    round(h_cvar * 100, 3),
        "dollar_var_1M":       f"${int(h_var * PORTFOLIO_VALUE):,}",
        "beta_vs_sp500":       compute_beta(port, bench),
    }


# ── 7. Charts ─────────────────────────────────────────────────────────────────

def plot_all(returns, bench, var_df, corr, summary):

    # Chart 1: Cumulative returns
    fig, ax = plt.subplots(figsize=(14, 6))
    cum = (1 + returns).cumprod()
    for col in cum.columns:
        ax.plot(cum.index, cum[col], linewidth=1, alpha=0.6, label=col)
    sp_cum = (1 + bench.reindex(returns.index).fillna(0)).cumprod()
    ax.plot(sp_cum.index, sp_cum, color="black", linewidth=2.5,
            linestyle="--", label="S&P 500", zorder=10)
    ax.set_title("Cumulative Returns vs S&P 500 (2019–2024)", fontsize=13)
    ax.set_ylabel("Growth of $1")
    ax.legend(ncol=5, fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/01_cumulative_returns.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")

    # Chart 2: VaR comparison bar chart
    fig, ax = plt.subplots(figsize=(14, 5))
    x, w = np.arange(len(var_df)), 0.25
    ax.bar(x - w, var_df["hist_var_pct"],  w, label="Historical",   color="#1976D2", alpha=0.85)
    ax.bar(x,     var_df["param_var_pct"], w, label="Parametric",   color="#F57C00", alpha=0.85)
    ax.bar(x + w, var_df["mc_var_pct"],    w, label="Monte Carlo",  color="#388E3C", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(var_df.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("1-Day VaR (%)")
    ax.set_title(f"Value at Risk by Method ({int(CONFIDENCE*100)}% Confidence Level)", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/02_var_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")

    # Chart 3: Correlation heatmap
    fig, ax = plt.subplots(figsize=(13, 10))
    tickers = corr.columns.tolist()
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tickers, fontsize=9)
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            val   = corr.values[i, j]
            color = "white" if abs(val) > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)
    plt.colorbar(im, ax=ax, shrink=0.75, label="Pearson Correlation")
    ax.set_title("Return Correlation Matrix", fontsize=13)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/03_correlation_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")

    # Chart 4: Risk-Return scatter
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, len(summary)))
    for i, (ticker, row) in enumerate(summary.iterrows()):
        ax.scatter(row["ann_vol_pct"], row["ann_return_pct"],
                   s=130, color=colors[i], zorder=5, edgecolors="white", linewidth=0.5)
        ax.annotate(ticker, (row["ann_vol_pct"], row["ann_return_pct"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Annualised Volatility (%)")
    ax.set_ylabel("Annualised Return (%)")
    ax.set_title("Risk-Return Scatter", fontsize=13)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/04_risk_return_scatter.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "="*50)
    print("  MODULE 2: PORTFOLIO RISK ANALYTICS")
    print("="*50)

    engine  = create_engine(f"sqlite:///{DB_PATH}", echo=False)

    print("\nLoading data from database...")
    returns = load_returns(engine)
    bench   = load_benchmark(engine)
    bench   = bench.reindex(returns.index).ffill().dropna()

    print("\nComputing Value at Risk (3 methods)...")
    var_df  = compute_all_var(returns)
    print("\n  VaR Results (% of position, 95% confidence):")
    print(var_df[["hist_var_pct","hist_cvar_pct","param_var_pct","mc_var_pct"]].to_string())

    print("\nComputing per-stock risk metrics...")
    summary = build_summary(returns, bench)
    print("\n  Per-Stock Summary:")
    display_cols = ["ann_return_pct","ann_vol_pct","sharpe","sortino","max_drawdown_pct","beta","alpha_pct"]
    print(summary[display_cols].to_string())

    print("\nComputing correlation matrix...")
    corr = returns.corr().round(3)
    print("\n  Correlation Matrix:")
    print(corr.to_string())

    print("\nPortfolio-level metrics (equal weight):")
    port = portfolio_metrics(returns, bench)
    print(f"  {'─'*40}")
    for k, v in port.items():
        print(f"  {k.replace('_',' ').title():<30} {v}")

    print("\nGenerating charts...")
    plot_all(returns, bench, var_df, corr, summary)

    print("\nSaving CSV outputs...")
    summary.to_csv(f"{OUTPUT_DIR}/risk_metrics.csv")
    corr.to_csv(f"{OUTPUT_DIR}/correlation_matrix.csv")
    var_df.to_csv(f"{OUTPUT_DIR}/var_results.csv")
    pd.DataFrame([port]).to_csv(f"{OUTPUT_DIR}/portfolio_summary.csv", index=False)
    print(f"  All CSVs saved to {OUTPUT_DIR}/")

    print("\n" + "="*50)
    print("  MODULE 2 COMPLETE")
    print(f"  Outputs in: {OUTPUT_DIR}/")
    print("  Next: python 06_ml_model.py")
    print("="*50 + "\n")


if __name__ == "__main__":
    run()
