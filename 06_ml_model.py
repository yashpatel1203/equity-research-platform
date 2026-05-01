"""
Module 3 — ML Signal Model
===========================
Predicts next-day return direction (Up/Down) using engineered features.
Framed as a TRADING SIGNAL GENERATOR — not a magic predictor.

Pipeline:
  1. Load features from database
  2. Feature selection & preprocessing
  3. Walk-forward train/test split (no data leakage)
  4. Train Random Forest + XGBoost classifiers
  5. Cross-validate with TimeSeriesSplit
  6. Evaluate: ROC-AUC, Precision, Recall, F1, Confusion Matrix
  7. Feature importance (permutation-based)
  8. Signal backtest — does following the signal make money?
  9. Save all results + charts to ml_output/

Usage:
    pip install xgboost shap
    python 06_ml_model.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics         import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    average_precision_score
)
from sklearn.inspection      import permutation_importance

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("WARNING: xgboost not installed. Run: pip install xgboost")

load_dotenv()
DB_PATH      = os.getenv("DB_PATH", "equity_data.db")
OUTPUT_DIR   = "ml_output"
RANDOM_STATE = 42
TEST_SIZE    = 0.2
TRADING_DAYS = 252

os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "return_5d", "return_21d",
    "volatility_21d", "volatility_63d",
    "macd", "macd_signal", "macd_hist",
    "rsi_14", "roc_10",
    "volume_ratio",
    "beta_63d",
    "sma_20", "sma_50",
]
TARGET_COL = "target_direction"


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_features(engine) -> pd.DataFrame:
    """Load engineered features — uses connection object for pandas 3.x compatibility."""
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT f.*, p.adj_close
                FROM stock_features f
                JOIN stock_prices p ON f.ticker = p.ticker AND f.date = p.date
                ORDER BY f.ticker, f.date
            """),
            conn,
            parse_dates=["date"]
        )

    if df.empty:
        raise ValueError("No features found. Run 04_engineer_features.py first.")

    print(f"  Loaded {len(df):,} rows across {df['ticker'].nunique()} tickers")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price relative to moving averages — strong momentum signals."""
    df = df.copy()
    df["price_to_sma20"] = df["adj_close"] / df["sma_20"]  - 1
    df["price_to_sma50"] = df["adj_close"] / df["sma_50"]  - 1
    df["sma20_to_sma50"] = df["sma_20"]    / df["sma_50"]  - 1
    return df


def add_macro_features(df: pd.DataFrame, engine) -> pd.DataFrame:
    """Join VIX and rate data — improves model during market stress periods."""
    try:
        with engine.connect() as conn:
            macro = pd.read_sql(
                text("""
                    SELECT date, series_name, value
                    FROM macro_data
                    WHERE series_name IN ('vix', 'treasury_10y', 'fed_funds_rate')
                """),
                conn,
                parse_dates=["date"]
            )

        if macro.empty:
            print("  WARNING: No macro data — skipping macro features")
            return df

        macro_wide = macro.pivot(index="date", columns="series_name", values="value")
        macro_wide.columns = [f"macro_{c}" for c in macro_wide.columns]
        macro_wide = macro_wide.reset_index()
        df = df.merge(macro_wide, on="date", how="left")
        print(f"  Added {len(macro_wide.columns)-1} macro features")
    except Exception as e:
        print(f"  WARNING: Could not add macro features: {e}")

    return df


def prepare_dataset(df: pd.DataFrame):
    """Select features, drop NaN rows, return clean df + feature list."""
    extra_feat = ["price_to_sma20", "price_to_sma50", "sma20_to_sma50"]
    macro_feat = [c for c in df.columns if c.startswith("macro_")]

    all_features = FEATURE_COLS + extra_feat + macro_feat
    all_features = [f for f in all_features if f in df.columns]

    keep = ["ticker", "date", TARGET_COL, "adj_close"] + all_features
    df   = df[[c for c in keep if c in df.columns]].copy()

    before = len(df)
    df     = df.dropna(subset=all_features + [TARGET_COL])
    print(f"  Dropped {before - len(df):,} NaN rows → {len(df):,} clean rows remaining")
    print(f"  Features ({len(all_features)}): {', '.join(all_features)}")

    return df, all_features


# ── 2. Walk-forward split ─────────────────────────────────────────────────────

def walk_forward_split(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """
    CRITICAL: always split on TIME, never randomly.
    Random splits leak future data into training → fake inflated accuracy.
    """
    df     = df.sort_values("date").reset_index(drop=True)
    cutoff = int(len(df) * (1 - test_size))
    train  = df.iloc[:cutoff].copy()
    test   = df.iloc[cutoff:].copy()

    print(f"  Train: {train['date'].min().date()} → {train['date'].max().date()} ({len(train):,} rows)")
    print(f"  Test:  {test['date'].min().date()}  → {test['date'].max().date()}  ({len(test):,} rows)")
    return train, test


# ── 3. Models ─────────────────────────────────────────────────────────────────

def build_models() -> dict:
    models = {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=20,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                verbosity=0,
            ))
        ])

    return models


# ── 4. Cross-validation ───────────────────────────────────────────────────────

def cross_validate_models(X_train, y_train, models: dict) -> pd.DataFrame:
    """TimeSeriesSplit CV — never uses future data to validate."""
    tscv    = TimeSeriesSplit(n_splits=5)
    results = []

    for name, model in models.items():
        scores = cross_val_score(
            model, X_train, y_train,
            cv=tscv, scoring="roc_auc", n_jobs=-1
        )
        results.append({
            "model":        name,
            "cv_auc_mean":  round(scores.mean(), 4),
            "cv_auc_std":   round(scores.std(),  4),
        })
        print(f"  {name:<15} CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    return pd.DataFrame(results)


# ── 5. Evaluate ───────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    auc    = roc_auc_score(y_test, y_prob)
    ap     = average_precision_score(y_test, y_prob)

    print(f"\n  ── {name} Test Results ──")
    print(f"  ROC-AUC:           {auc:.4f}  (>0.53 is meaningful for daily prediction)")
    print(f"  Avg Precision:     {ap:.4f}")
    print(f"  Accuracy:          {report['accuracy']:.4f}")
    print(f"  Precision (Up):    {report['1']['precision']:.4f}")
    print(f"  Recall (Up):       {report['1']['recall']:.4f}")
    print(f"  F1-score (Up):     {report['1']['f1-score']:.4f}")
    print(f"  Precision (Down):  {report['0']['precision']:.4f}")
    print(f"  Recall (Down):     {report['0']['recall']:.4f}")

    return {
        "model":          name,
        "roc_auc":        round(auc, 4),
        "avg_precision":  round(ap, 4),
        "accuracy":       round(report["accuracy"], 4),
        "precision_up":   round(report["1"]["precision"], 4),
        "recall_up":      round(report["1"]["recall"], 4),
        "f1_up":          round(report["1"]["f1-score"], 4),
        "precision_down": round(report["0"]["precision"], 4),
        "recall_down":    round(report["0"]["recall"], 4),
    }


# ── 6. Feature importance ─────────────────────────────────────────────────────

def get_feature_importance(model, feature_names: list, X_test, y_test) -> pd.DataFrame:
    clf = model.named_steps["clf"]

    # Built-in importance
    if hasattr(clf, "feature_importances_"):
        builtin = pd.Series(clf.feature_importances_, index=feature_names)
    else:
        builtin = pd.Series(np.zeros(len(feature_names)), index=feature_names)

    # Permutation importance (more reliable)
    try:
        X_scaled = model.named_steps["scaler"].transform(X_test)
        perm     = permutation_importance(
            clf, X_scaled, y_test,
            n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
        )
        perm_imp = pd.Series(perm.importances_mean, index=feature_names)
    except Exception:
        perm_imp = builtin.copy()

    return pd.DataFrame({
        "builtin_importance":     builtin,
        "permutation_importance": perm_imp,
    }).sort_values("permutation_importance", ascending=False)


# ── 7. Signal backtest ────────────────────────────────────────────────────────

def backtest_signal(model, X_test, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Long-only signal strategy:
      signal = 1 → go long (buy)
      signal = 0 → stay flat (cash)
    Compare vs buy-and-hold benchmark.
    """
    results = test_df[["date", "ticker", "adj_close"]].copy()
    results["signal"]       = model.predict(X_test)
    results["signal_prob"]  = model.predict_proba(X_test)[:, 1]
    results["actual_ret"]   = test_df["adj_close"].pct_change()
    results["strategy_ret"] = results["signal"].shift(1) * results["actual_ret"]
    results["bah_ret"]      = results["actual_ret"]
    results["strategy_cum"] = (1 + results["strategy_ret"].fillna(0)).cumprod()
    results["bah_cum"]      = (1 + results["bah_ret"].fillna(0)).cumprod()
    return results


# ── 8. Charts ─────────────────────────────────────────────────────────────────

def plot_roc_curves(fitted_models: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ["#1976D2", "#388E3C", "#F57C00"]

    for i, (name, model) in enumerate(fitted_models.items()):
        y_prob       = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _  = roc_curve(y_test, y_prob)
        auc          = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=colors[i % len(colors)],
                linewidth=2, label=f"{name} (AUC = {auc:.3f})")

    ax.plot([0,1],[0,1], "k--", linewidth=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Signal Model", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/01_roc_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_feature_importance(imp_df: pd.DataFrame, model_name: str):
    top    = imp_df.head(15)
    colors = ["#1976D2" if v >= 0 else "#E53935" for v in top["permutation_importance"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index[::-1], top["permutation_importance"][::-1],
            color=colors[::-1], alpha=0.85)
    ax.set_xlabel("Permutation Importance (mean AUC decrease)")
    ax.set_title(f"Top 15 Feature Importances — {model_name}", fontsize=13)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/02_feature_importance_{model_name.lower()}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_confusion_matrix(model, X_test, y_test, model_name: str):
    cm  = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(6, 5))
    im  = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Down", "Pred Up"])
    ax.set_yticklabels(["Actual Down", "Actual Up"])
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    fontsize=14, color=color)
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/03_confusion_{model_name.lower()}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_backtest(bt: pd.DataFrame, model_name: str):
    daily = bt.groupby("date")[["strategy_ret", "bah_ret"]].mean()
    daily["strategy_cum"] = (1 + daily["strategy_ret"].fillna(0)).cumprod()
    daily["bah_cum"]      = (1 + daily["bah_ret"].fillna(0)).cumprod()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                              gridspec_kw={"height_ratios": [3, 1]})

    axes[0].plot(daily.index, daily["strategy_cum"], color="#1976D2",
                 linewidth=2, label=f"{model_name} Signal Strategy")
    axes[0].plot(daily.index, daily["bah_cum"],  color="#E53935",
                 linewidth=2, linestyle="--", label="Buy & Hold")
    axes[0].fill_between(
        daily.index, daily["strategy_cum"], daily["bah_cum"],
        where=daily["strategy_cum"] >= daily["bah_cum"],
        alpha=0.1, color="#1976D2"
    )
    axes[0].set_title(f"Signal Strategy vs Buy & Hold — {model_name}", fontsize=13)
    axes[0].set_ylabel("Cumulative Return")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    bar_colors = ["#1976D2" if r >= 0 else "#E53935"
                  for r in daily["strategy_ret"].fillna(0)]
    axes[1].bar(daily.index, daily["strategy_ret"].fillna(0),
                color=bar_colors, alpha=0.6, width=1)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Daily Return")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/04_backtest_{model_name.lower()}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_signal_distribution(bt: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Signal probability histogram
    axes[0].hist(bt["signal_prob"], bins=50, color="#1976D2", alpha=0.8, edgecolor="white")
    axes[0].axvline(0.5, color="red", linewidth=1.5, linestyle="--", label="Decision threshold")
    axes[0].set_xlabel("Predicted Probability (Up)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Signal Probability Distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Average return by confidence bucket
    bt["prob_bucket"] = pd.cut(
        bt["signal_prob"],
        bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0],
        labels=["0–30%", "30–40%", "40–50%", "50–60%", "60–70%", "70–100%"]
    )
    bucket_ret  = bt.groupby("prob_bucket", observed=True)["actual_ret"].mean() * 100
    bar_colors  = ["#E53935","#EF9A9A","#FFCDD2","#C8E6C9","#81C784","#388E3C"]
    axes[1].bar(bucket_ret.index, bucket_ret.values, color=bar_colors, alpha=0.85)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xlabel("Signal Confidence Bucket")
    axes[1].set_ylabel("Avg Next-Day Return (%)")
    axes[1].set_title("Average Return by Signal Confidence")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/05_signal_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "="*55)
    print("  MODULE 3: ML SIGNAL MODEL")
    print("="*55)

    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)

    # ── Load ─────────────────────────────────────────────
    print("\n[1/7] Loading features from database...")
    raw_df = load_features(engine)
    raw_df = add_price_features(raw_df)
    raw_df = add_macro_features(raw_df, engine)
    df, feature_cols = prepare_dataset(raw_df)

    # ── Split ────────────────────────────────────────────
    print("\n[2/7] Walk-forward train/test split...")
    train_df, test_df = walk_forward_split(df)

    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df[TARGET_COL].values

    print(f"  Class balance — Up: {y_train.mean():.1%}  Down: {1-y_train.mean():.1%}")

    # ── Cross-validate ────────────────────────────────────
    print("\n[3/7] Cross-validating with TimeSeriesSplit (5 folds)...")
    models     = build_models()
    cv_results = cross_validate_models(X_train, y_train, models)

    # ── Train ─────────────────────────────────────────────
    print("\n[4/7] Training final models on full training set...")
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        print(f"  {name} — trained.")

    # ── Evaluate ──────────────────────────────────────────
    print("\n[5/7] Evaluating on held-out test set...")
    eval_results = [evaluate_model(m, X_test, y_test, n) for n, m in fitted.items()]
    eval_df      = pd.DataFrame(eval_results)

    # ── Feature importance ────────────────────────────────
    print("\n[6/7] Computing feature importance...")
    best_name  = eval_df.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    best_model = fitted[best_name]
    imp_df     = get_feature_importance(best_model, feature_cols, X_test, y_test)

    print(f"\n  Top 10 features ({best_name}):")
    print(imp_df.head(10)[["permutation_importance"]].to_string())

    # ── Charts ────────────────────────────────────────────
    print("\n[7/7] Generating charts...")
    plot_roc_curves(fitted, X_test, y_test)
    plot_feature_importance(imp_df, best_name)

    for name, model in fitted.items():
        plot_confusion_matrix(model, X_test, y_test, name)
        bt = backtest_signal(model, X_test, test_df)
        plot_backtest(bt, name)

    bt_best = backtest_signal(best_model, X_test, test_df)
    plot_signal_distribution(bt_best)

    # ── Save CSVs ─────────────────────────────────────────
    print("\nSaving CSV outputs...")
    eval_df.to_csv(f"{OUTPUT_DIR}/model_evaluation.csv", index=False)
    cv_results.to_csv(f"{OUTPUT_DIR}/cv_results.csv", index=False)
    imp_df.to_csv(f"{OUTPUT_DIR}/feature_importance.csv")
    bt_best[["date","ticker","signal","signal_prob","actual_ret",
             "strategy_ret","strategy_cum"]].to_csv(
        f"{OUTPUT_DIR}/backtest_signals.csv", index=False
    )
    print(f"  All CSVs saved to {OUTPUT_DIR}/")

    # ── Summary ───────────────────────────────────────────
    best_row = eval_df[eval_df["model"] == best_name].iloc[0]
    print("\n" + "="*55)
    print("  MODULE 3 COMPLETE")
    print("="*55)
    print(f"\n  Best model:     {best_name}")
    print(f"  ROC-AUC:        {best_row['roc_auc']}")
    print(f"  Accuracy:       {best_row['accuracy']}")
    print(f"  Precision (Up): {best_row['precision_up']}")
    print(f"  Recall (Up):    {best_row['recall_up']}")
    print(f"\n  Top 3 predictive features:")
    for i, (feat, row) in enumerate(imp_df.head(3).iterrows(), 1):
        print(f"  {i}. {feat:<28} {row['permutation_importance']:.4f}")
    print(f"\n  Charts + CSVs saved to: {OUTPUT_DIR}/")
    print("\n  NOTE: AUC 0.53–0.58 is meaningful for daily prediction.")
    print("  Frame as a signal generator — not a crystal ball.")
    print("\n  Next: Module 4 — Dashboard & Investment Memo")
    print("="*55 + "\n")


if __name__ == "__main__":
    run()
