
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os
import warnings
warnings.filterwarnings("ignore")

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
DB_PATH      = os.getenv("DB_PATH", "equity_data.db")
RISK_DIR     = "risk_output"
ML_DIR       = "ml_output"
DASH_DIR     = "dashboard_output"
os.makedirs(DASH_DIR, exist_ok=True)

# ── Brand colours ─────────────────────────────────────────────────────────────
C_BLUE   = "#1565C0"
C_GREEN  = "#2E7D32"
C_RED    = "#C62828"
C_AMBER  = "#E65100"
C_GRAY   = "#455A64"
C_LIGHT  = "#F5F7FA"
C_WHITE  = "#FFFFFF"

SECTOR_COLORS = {
    "Technology": "#1565C0",
    "Financials":  "#2E7D32",
    "Healthcare":  "#6A1B9A",
    "Energy":      "#E65100",
    "Consumer":    "#00838F",
}

SECTOR_MAP = {
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology","GOOGL":"Technology",
    "JPM":"Financials", "BAC":"Financials", "GS":"Financials",  "MS":"Financials",
    "JNJ":"Healthcare", "UNH":"Healthcare", "PFE":"Healthcare", "ABBV":"Healthcare",
    "XOM":"Energy",     "CVX":"Energy",     "COP":"Energy",     "SLB":"Energy",
    "AMZN":"Consumer",  "TSLA":"Consumer",  "HD":"Consumer",    "MCD":"Consumer",
}


def set_style():
    plt.rcParams.update({
        "figure.facecolor":   C_WHITE,
        "axes.facecolor":     C_LIGHT,
        "axes.edgecolor":     "#B0BEC5",
        "axes.labelcolor":    C_GRAY,
        "axes.titleweight":   "bold",
        "axes.titlesize":     11,
        "axes.labelsize":     9,
        "xtick.color":        C_GRAY,
        "ytick.color":        C_GRAY,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "grid.color":         "#CFD8DC",
        "grid.linewidth":     0.5,
        "grid.alpha":         0.7,
        "legend.fontsize":    8,
        "legend.framealpha":  0.9,
        "font.family":        "sans-serif",
    })


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_prices(engine) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT ticker, date, adj_close FROM stock_prices ORDER BY date"),
            conn, parse_dates=["date"]
        )
    return df.pivot(index="date", columns="ticker", values="adj_close")


def load_benchmark(engine) -> pd.Series:
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT date, adj_close FROM benchmark_prices ORDER BY date"),
            conn, parse_dates=["date"]
        )
    return df.set_index("date")["adj_close"]


def load_risk_metrics() -> pd.DataFrame:
    path = f"{RISK_DIR}/risk_metrics.csv"
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    print(f"  WARNING: {path} not found. Run 05_risk_analytics.py first.")
    return pd.DataFrame()


def load_var_results() -> pd.DataFrame:
    path = f"{RISK_DIR}/var_results.csv"
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return pd.DataFrame()


def load_correlation() -> pd.DataFrame:
    path = f"{RISK_DIR}/correlation_matrix.csv"
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return pd.DataFrame()


def load_portfolio_summary() -> dict:
    path = f"{RISK_DIR}/portfolio_summary.csv"
    if os.path.exists(path):
        return pd.read_csv(path).iloc[0].to_dict()
    return {}


def load_ml_eval() -> pd.DataFrame:
    path = f"{ML_DIR}/model_evaluation.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def load_feature_importance() -> pd.DataFrame:
    path = f"{ML_DIR}/feature_importance.csv"
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return pd.DataFrame()


def load_backtest() -> pd.DataFrame:
    path = f"{ML_DIR}/backtest_signals.csv"
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    return pd.DataFrame()


# ── Page 1: Portfolio Performance ────────────────────────────────────────────

def page_portfolio_performance(prices: pd.DataFrame, bench: pd.Series,
                                risk: pd.DataFrame):
    fig = plt.figure(figsize=(16, 10), facecolor=C_WHITE)
    fig.suptitle("EQUITY RESEARCH PLATFORM  |  Portfolio Performance",
                 fontsize=14, fontweight="bold", color=C_BLUE, y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.92, bottom=0.08, left=0.06, right=0.97)

    returns  = prices.pct_change().dropna()
    port_ret = returns.mean(axis=1)
    port_cum = (1 + port_ret).cumprod()
    bench_ret = bench.pct_change().dropna().reindex(port_ret.index).ffill()
    bench_cum = (1 + bench_ret.fillna(0)).cumprod()

    # ── Panel 1: Cumulative returns (wide) ──
    ax1 = fig.add_subplot(gs[0, :2])
    colors20 = plt.cm.tab20(np.linspace(0, 1, len(prices.columns)))
    for i, col in enumerate(prices.columns):
        cum = (1 + returns[col]).cumprod()
        ax1.plot(cum.index, cum, linewidth=0.8, alpha=0.5,
                 color=colors20[i], label=col)
    ax1.plot(port_cum.index, port_cum, color=C_BLUE,
             linewidth=2.5, label="Portfolio", zorder=10)
    ax1.plot(bench_cum.index, bench_cum, color=C_RED,
             linewidth=2, linestyle="--", label="S&P 500", zorder=9)
    ax1.set_title("Cumulative Returns")
    ax1.set_ylabel("Growth of $1")
    ax1.legend(ncol=6, fontsize=6.5, loc="upper left")
    ax1.grid(True)

    # ── Panel 2: Sector donut ──
    ax2 = fig.add_subplot(gs[0, 2])
    if not risk.empty and "ann_return_pct" in risk.columns:
        sectors = {}
        for ticker in risk.index:
            s = SECTOR_MAP.get(ticker, "Other")
            sectors[s] = sectors.get(s, 0) + 1
        wedge_colors = [SECTOR_COLORS.get(s, C_GRAY) for s in sectors.keys()]
        wedges, texts, autotexts = ax2.pie(
            sectors.values(), labels=sectors.keys(),
            colors=wedge_colors, autopct="%1.0f%%",
            startangle=90, wedgeprops={"width": 0.55, "edgecolor": "white"}
        )
        for t in autotexts:
            t.set_fontsize(8)
        ax2.set_title("Sector Exposure")
    else:
        ax2.text(0.5, 0.5, "Run Module 2\nfor risk data",
                 ha="center", va="center", transform=ax2.transAxes, color=C_GRAY)
        ax2.set_title("Sector Exposure")

    # ── Panel 3: Rolling drawdown ──
    ax3 = fig.add_subplot(gs[1, :2])
    roll_max  = port_cum.cummax()
    drawdown  = (port_cum - roll_max) / roll_max * 100
    ax3.fill_between(drawdown.index, drawdown, 0, alpha=0.4, color=C_RED)
    ax3.plot(drawdown.index, drawdown, color=C_RED, linewidth=1)
    ax3.set_title("Portfolio Drawdown (%)")
    ax3.set_ylabel("Drawdown (%)")
    ax3.grid(True)

    # ── Panel 4: Top / Bottom performers ──
    ax4 = fig.add_subplot(gs[1, 2])
    if not risk.empty and "ann_return_pct" in risk.columns:
        sorted_r = risk["ann_return_pct"].sort_values()
        colors_bar = [C_RED if v < 0 else C_GREEN for v in sorted_r.values]
        bars = ax4.barh(sorted_r.index, sorted_r.values,
                        color=colors_bar, alpha=0.85, height=0.6)
        ax4.axvline(0, color=C_GRAY, linewidth=0.8)
        ax4.set_title("Annualised Return by Stock (%)")
        ax4.set_xlabel("Ann. Return (%)")
        ax4.grid(axis="x", alpha=0.5)
    else:
        ax4.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=ax4.transAxes, color=C_GRAY)

    plt.savefig(f"{DASH_DIR}/page1_portfolio_performance.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {DASH_DIR}/page1_portfolio_performance.png")


# ── Page 2: Risk Heatmap ──────────────────────────────────────────────────────

def page_risk_heatmap(risk: pd.DataFrame, var_df: pd.DataFrame, corr: pd.DataFrame):
    fig = plt.figure(figsize=(16, 10), facecolor=C_WHITE)
    fig.suptitle("EQUITY RESEARCH PLATFORM  |  Risk Analytics",
                 fontsize=14, fontweight="bold", color=C_BLUE, y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.92, bottom=0.08, left=0.06, right=0.97)

    # ── Panel 1: Correlation heatmap ──
    ax1 = fig.add_subplot(gs[:, :2])
    if not corr.empty:
        cmap = LinearSegmentedColormap.from_list(
            "rg", ["#C62828", "#FFFFFF", "#1B5E20"], N=200
        )
        im = ax1.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        tickers = corr.columns.tolist()
        ax1.set_xticks(range(len(tickers)))
        ax1.set_yticks(range(len(tickers)))
        ax1.set_xticklabels(tickers, rotation=45, ha="right", fontsize=8)
        ax1.set_yticklabels(tickers, fontsize=8)
        for i in range(len(tickers)):
            for j in range(len(tickers)):
                val   = corr.values[i, j]
                color = "white" if abs(val) > 0.65 else "black"
                ax1.text(j, i, f"{val:.2f}", ha="center", va="center",
                         fontsize=6.5, color=color)
        plt.colorbar(im, ax=ax1, shrink=0.6, label="Pearson Correlation")
        ax1.set_title("Return Correlation Matrix", pad=10)
    else:
        ax1.text(0.5, 0.5, "Run Module 2 for correlation data",
                 ha="center", va="center", transform=ax1.transAxes)

    # ── Panel 2: VaR bar chart ──
    ax2 = fig.add_subplot(gs[0, 2])
    if not var_df.empty and "hist_var_pct" in var_df.columns:
        top_var = var_df["hist_var_pct"].sort_values(ascending=False).head(10)
        bar_colors = [SECTOR_COLORS.get(SECTOR_MAP.get(t, ""), C_BLUE)
                      for t in top_var.index]
        ax2.barh(top_var.index[::-1], top_var.values[::-1],
                 color=bar_colors[::-1], alpha=0.85, height=0.6)
        ax2.set_title("1-Day Historical VaR (95%)", pad=8)
        ax2.set_xlabel("VaR (%)")
        ax2.grid(axis="x", alpha=0.5)
    else:
        ax2.text(0.5, 0.5, "No VaR data", ha="center", va="center",
                 transform=ax2.transAxes)

    # ── Panel 3: Risk-Return scatter ──
    ax3 = fig.add_subplot(gs[1, 2])
    if not risk.empty and "ann_vol_pct" in risk.columns:
        for ticker, row in risk.iterrows():
            sector = SECTOR_MAP.get(ticker, "Other")
            color  = SECTOR_COLORS.get(sector, C_GRAY)
            ax3.scatter(row["ann_vol_pct"], row["ann_return_pct"],
                        s=80, color=color, alpha=0.85, zorder=5)
            ax3.annotate(ticker, (row["ann_vol_pct"], row["ann_return_pct"]),
                         textcoords="offset points", xytext=(5, 3), fontsize=6.5)
        handles = [mpatches.Patch(color=v, label=k)
                   for k, v in SECTOR_COLORS.items()]
        ax3.legend(handles=handles, fontsize=6.5, loc="lower right")
        ax3.axhline(0, color=C_GRAY, linewidth=0.6, linestyle="--")
        ax3.set_xlabel("Annualised Volatility (%)")
        ax3.set_ylabel("Annualised Return (%)")
        ax3.set_title("Risk-Return by Sector", pad=8)
        ax3.grid(True)
    else:
        ax3.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=ax3.transAxes)

    plt.savefig(f"{DASH_DIR}/page2_risk_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {DASH_DIR}/page2_risk_heatmap.png")


# ── Page 3: ML Signal Tracker ────────────────────────────────────────────────

def page_ml_signals(ml_eval: pd.DataFrame, feat_imp: pd.DataFrame,
                    backtest: pd.DataFrame):
    fig = plt.figure(figsize=(16, 10), facecolor=C_WHITE)
    fig.suptitle("EQUITY RESEARCH PLATFORM  |  ML Signal Model",
                 fontsize=14, fontweight="bold", color=C_BLUE, y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.92, bottom=0.08, left=0.06, right=0.97)

    # ── Panel 1: Model metrics table ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    if not ml_eval.empty:
        display_cols = ["model", "roc_auc", "accuracy", "precision_up", "recall_up", "f1_up"]
        display_cols = [c for c in display_cols if c in ml_eval.columns]
        table_data   = ml_eval[display_cols].values.tolist()
        col_labels   = [c.replace("_", "\n") for c in display_cols]
        tbl = ax1.table(
            cellText=table_data, colLabels=col_labels,
            loc="center", cellLoc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.6)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor(C_BLUE)
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#E3F2FD")
    ax1.set_title("Model Performance Metrics", pad=10)

    # ── Panel 2: Feature importance ──
    ax2 = fig.add_subplot(gs[0, 1])
    if not feat_imp.empty and "permutation_importance" in feat_imp.columns:
        top = feat_imp.head(12)
        bar_colors = [C_BLUE if v >= 0 else C_RED
                      for v in top["permutation_importance"]]
        ax2.barh(top.index[::-1], top["permutation_importance"][::-1],
                 color=bar_colors[::-1], alpha=0.85, height=0.65)
        ax2.axvline(0, color=C_GRAY, linewidth=0.8)
        ax2.set_title("Top Feature Importances", pad=8)
        ax2.set_xlabel("Permutation Importance")
        ax2.grid(axis="x", alpha=0.5)
    else:
        ax2.text(0.5, 0.5, "Run Module 3 for ML data",
                 ha="center", va="center", transform=ax2.transAxes)

    # ── Panel 3: Signal confidence distribution ──
    ax3 = fig.add_subplot(gs[0, 2])
    if not backtest.empty and "signal_prob" in backtest.columns:
        ax3.hist(backtest["signal_prob"], bins=40,
                 color=C_BLUE, alpha=0.8, edgecolor="white")
        ax3.axvline(0.5, color=C_RED, linewidth=1.5,
                    linestyle="--", label="Decision boundary")
        ax3.set_xlabel("Predicted Prob (Up)")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Signal Confidence Distribution", pad=8)
        ax3.legend()
        ax3.grid(True)
    else:
        ax3.text(0.5, 0.5, "No backtest data", ha="center", va="center",
                 transform=ax3.transAxes)

    # ── Panel 4: Backtest cumulative return (wide) ──
    ax4 = fig.add_subplot(gs[1, :])
    if not backtest.empty and "strategy_ret" in backtest.columns:
        # bah_ret may not be in the saved CSV — reconstruct from actual_ret
        if "bah_ret" not in backtest.columns and "actual_ret" in backtest.columns:
            backtest = backtest.copy()
            backtest["bah_ret"] = backtest["actual_ret"]
        elif "bah_ret" not in backtest.columns:
            backtest = backtest.copy()
            backtest["bah_ret"] = 0.0

        daily = backtest.groupby("date")[["strategy_ret", "bah_ret"]].mean()
        strat_cum = (1 + daily["strategy_ret"].fillna(0)).cumprod()
        bah_cum   = (1 + daily["bah_ret"].fillna(0)).cumprod()

        ax4.plot(strat_cum.index, strat_cum, color=C_BLUE,
                 linewidth=2, label="ML Signal Strategy")
        ax4.plot(bah_cum.index, bah_cum, color=C_RED,
                 linewidth=2, linestyle="--", label="Buy & Hold")
        ax4.fill_between(strat_cum.index, strat_cum, bah_cum,
                         where=strat_cum >= bah_cum,
                         alpha=0.12, color=C_BLUE, label="Outperformance")
        ax4.fill_between(strat_cum.index, strat_cum, bah_cum,
                         where=strat_cum < bah_cum,
                         alpha=0.08, color=C_RED)
        ax4.set_title("Signal Strategy vs Buy & Hold (Test Period)", pad=8)
        ax4.set_ylabel("Cumulative Return")
        ax4.legend(loc="upper left")
        ax4.grid(True)
    else:
        ax4.text(0.5, 0.5, "Run Module 3 for backtest data",
                 ha="center", va="center", transform=ax4.transAxes)

    plt.savefig(f"{DASH_DIR}/page3_ml_signals.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {DASH_DIR}/page3_ml_signals.png")


# ── Page 4: Executive Summary ────────────────────────────────────────────────

def page_executive_summary(risk: pd.DataFrame, port_summary: dict,
                            ml_eval: pd.DataFrame):
    fig = plt.figure(figsize=(16, 10), facecolor=C_WHITE)
    fig.suptitle("EQUITY RESEARCH PLATFORM  |  Executive Summary",
                 fontsize=14, fontweight="bold", color=C_BLUE, y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.92, bottom=0.08, left=0.06, right=0.97)

    # ── Panel 1: Portfolio KPI cards ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    if port_summary:
        kpi_items = [
            ("Ann. Return",    f"{port_summary.get('ann_return_pct', 'N/A')}%"),
            ("Ann. Volatility", f"{port_summary.get('ann_vol_pct', 'N/A')}%"),
            ("Sharpe Ratio",    str(port_summary.get("sharpe", "N/A"))),
            ("Sortino Ratio",   str(port_summary.get("sortino", "N/A"))),
            ("Max Drawdown",    f"{port_summary.get('max_drawdown_pct', 'N/A')}%"),
            ("VaR (95%, 1D)",   f"{port_summary.get('hist_var_95_pct', 'N/A')}%"),
            ("Dollar VaR ($1M)",str(port_summary.get("dollar_var_1M", "N/A"))),
            ("Beta vs S&P 500", str(port_summary.get("beta_vs_sp500", "N/A"))),
        ]
        y = 0.95
        ax1.text(0.5, 1.0, "Portfolio KPIs", ha="center", va="top",
                 transform=ax1.transAxes, fontsize=10, fontweight="bold",
                 color=C_BLUE)
        for label, value in kpi_items:
            ax1.text(0.05, y, label, transform=ax1.transAxes,
                     fontsize=9, color=C_GRAY)
            ax1.text(0.95, y, value, transform=ax1.transAxes,
                     fontsize=9, fontweight="bold", color=C_BLUE, ha="right")
            y -= 0.10
            ax1.plot([0.02, 0.98], [y + 0.02, y + 0.02],
                     color="#CFD8DC", linewidth=0.5,
                     transform=ax1.transAxes)
    else:
        ax1.text(0.5, 0.5, "Run Module 2\nfor portfolio metrics",
                 ha="center", va="center", transform=ax1.transAxes)

    # ── Panel 2: Top 5 by Sharpe ──
    ax2 = fig.add_subplot(gs[0, 1])
    if not risk.empty and "sharpe" in risk.columns:
        top5 = risk["sharpe"].dropna().sort_values(ascending=False).head(5)
        colors_bar = [SECTOR_COLORS.get(SECTOR_MAP.get(t, ""), C_BLUE)
                      for t in top5.index]
        ax2.barh(top5.index[::-1], top5.values[::-1],
                 color=colors_bar[::-1], alpha=0.85, height=0.5)
        ax2.set_title("Top 5 — Sharpe Ratio", pad=8)
        ax2.set_xlabel("Sharpe Ratio")
        ax2.axvline(1, color=C_GRAY, linewidth=0.8, linestyle="--",
                    label="Threshold = 1.0")
        ax2.legend(fontsize=7)
        ax2.grid(axis="x", alpha=0.5)
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=ax2.transAxes)

    # ── Panel 3: Top 5 by Sortino ──
    ax3 = fig.add_subplot(gs[0, 2])
    if not risk.empty and "sortino" in risk.columns:
        top5s = risk["sortino"].dropna().sort_values(ascending=False).head(5)
        colors_bar = [SECTOR_COLORS.get(SECTOR_MAP.get(t, ""), C_GREEN)
                      for t in top5s.index]
        ax3.barh(top5s.index[::-1], top5s.values[::-1],
                 color=colors_bar[::-1], alpha=0.85, height=0.5)
        ax3.set_title("Top 5 — Sortino Ratio", pad=8)
        ax3.set_xlabel("Sortino Ratio")
        ax3.grid(axis="x", alpha=0.5)
    else:
        ax3.text(0.5, 0.5, "No data", ha="center", va="center",
                 transform=ax3.transAxes)

    # ── Panel 4: Full metrics table (wide, bottom) ──
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis("off")
    if not risk.empty:
        display = ["ann_return_pct","ann_vol_pct","sharpe","sortino",
                   "max_drawdown_pct","beta","alpha_pct"]
        display = [c for c in display if c in risk.columns]
        col_labels = ["Ann Ret %","Ann Vol %","Sharpe","Sortino",
                      "Max DD %","Beta","Alpha %"][:len(display)]
        table_data = risk[display].round(3).values.tolist()
        row_labels  = risk.index.tolist()

        tbl = ax4.table(
            cellText=table_data,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc="center", cellLoc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1, 1.35)

        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor(C_BLUE)
                cell.set_text_props(color="white", fontweight="bold")
            elif c == -1:
                cell.set_facecolor("#E3F2FD")
                cell.set_text_props(fontweight="bold", color=C_BLUE)
            elif r % 2 == 0:
                cell.set_facecolor("#F5F7FA")
        ax4.set_title("Full Risk Metrics by Stock", pad=12, fontsize=10,
                      fontweight="bold")
    else:
        ax4.text(0.5, 0.5, "Run Module 2 for risk metrics",
                 ha="center", va="center", transform=ax4.transAxes)

    plt.savefig(f"{DASH_DIR}/page4_executive_summary.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {DASH_DIR}/page4_executive_summary.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "="*55)
    print("  MODULE 4a: DASHBOARD GENERATION")
    print("="*55)

    set_style()
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)

    print("\nLoading data...")
    try:
        prices  = load_prices(engine)
        bench   = load_benchmark(engine)
    except Exception as e:
        print(f"  ERROR loading DB: {e}")
        print("  Make sure equity_data.db exists and Module 1 completed.")
        return

    risk         = load_risk_metrics()
    var_df       = load_var_results()
    corr         = load_correlation()
    port_summary = load_portfolio_summary()
    ml_eval      = load_ml_eval()
    feat_imp     = load_feature_importance()
    backtest     = load_backtest()

    print("\nGenerating dashboard pages...")

    print("  Page 1: Portfolio Performance...")
    page_portfolio_performance(prices, bench, risk)

    print("  Page 2: Risk Heatmap...")
    page_risk_heatmap(risk, var_df, corr)

    print("  Page 3: ML Signal Tracker...")
    page_ml_signals(ml_eval, feat_imp, backtest)

    print("  Page 4: Executive Summary...")
    page_executive_summary(risk, port_summary, ml_eval)

    print("\n" + "="*55)
    print("  DASHBOARD COMPLETE")
    print(f"  4 pages saved to: {DASH_DIR}/")
    print("="*55)
    print("\n  Next: python 08_investment_memo.py")


if __name__ == "__main__":
    run()
