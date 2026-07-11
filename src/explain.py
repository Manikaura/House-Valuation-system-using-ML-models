"""Generate the report figures used by the README and the slide deck.

Produces (in reports/figures/):
  model_comparison.png     - CV MAE across the model zoo
  actual_vs_predicted.png  - held-out predictions vs truth
  residuals.png            - residual distribution
  error_by_price.png       - MAPE across price bands (where the model is weak/strong)
  shap_importance.png      - global SHAP feature importance (log-price model)
  shap_beeswarm.png        - SHAP value distribution per feature
"""
from __future__ import annotations

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import build_frame

HERE = os.path.dirname(__file__)
REPORTS = os.path.join(HERE, "..", "reports")
FIG = os.path.join(REPORTS, "figures")
ACCENT = "#2563eb"      # bright blue
ACCENT2 = "#f59e0b"     # amber
GRID = "#e5e7eb"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#cbd5e1", "axes.grid": True, "grid.color": GRID,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "savefig.dpi": 130, "savefig.bbox": "tight",
})


def _load():
    art = joblib.load(os.path.join(HERE, "..", "models", "avm_pipeline.joblib"))
    X, y = build_frame()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    return art, X_tr, X_te, y_tr, y_te


def fig_model_comparison():
    df = pd.read_csv(os.path.join(REPORTS, "model_comparison.csv")).sort_values("CV_MAE")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = [ACCENT if m == "LightGBM" else "#94a3b8" for m in df["model"]]
    ax.barh(df["model"], df["CV_MAE"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("5-fold CV Mean Absolute Error (AUD)")
    ax.set_title("Model comparison — lower is better")
    for y_, v in zip(range(len(df)), df["CV_MAE"]):
        ax.text(v, y_, f"  ${v/1000:,.0f}k", va="center", fontsize=10)
    fig.savefig(os.path.join(FIG, "model_comparison.png"))
    plt.close(fig)


def fig_predictions(art, X_te, y_te):
    pred = art["model"].predict(X_te)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_te / 1e6, pred / 1e6, s=8, alpha=0.35, color=ACCENT)
    lim = [0, max(y_te.max(), pred.max()) / 1e6]
    ax.plot(lim, lim, "--", color="#111827", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual price ($M)"); ax.set_ylabel("Predicted price ($M)")
    ax.set_title(f"Actual vs predicted  (R² = {art['test_metrics']['R2']:.3f})")
    fig.savefig(os.path.join(FIG, "actual_vs_predicted.png"))
    plt.close(fig)

    resid = (pred - y_te) / 1e3
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(resid.clip(-800, 800), bins=60, color=ACCENT, alpha=0.85)
    ax.axvline(0, color="#111827", lw=1)
    ax.set_xlabel("Prediction error (thousand $)"); ax.set_ylabel("Count")
    ax.set_title("Residual distribution (held-out)")
    fig.savefig(os.path.join(FIG, "residuals.png"))
    plt.close(fig)

    # error across price bands
    ape = np.abs((pred - y_te.values) / y_te.values) * 100
    bands = pd.qcut(y_te, 5)
    tbl = pd.DataFrame({"ape": ape, "band": bands}).groupby("band", observed=True)["ape"].mean()
    labels = [f"${int(iv.left/1000)}–{int(iv.right/1000)}k" for iv in tbl.index]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, tbl.values, color=ACCENT2)
    ax.set_ylabel("Mean abs. % error"); ax.set_title("Accuracy across price bands")
    plt.xticks(rotation=20)
    fig.savefig(os.path.join(FIG, "error_by_price.png"))
    plt.close(fig)


def fig_shap(art, X_tr):
    import shap

    reg = art["model"].regressor_
    prep, model = reg.named_steps["prep"], reg.named_steps["model"]
    sample = X_tr.sample(min(2000, len(X_tr)), random_state=42)
    Xt = prep.transform(sample)
    names = art["feature_names_out"]

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xt)

    shap.summary_plot(sv, Xt, feature_names=names, plot_type="bar", show=False, max_display=15)
    plt.title("Global feature importance (SHAP)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "shap_importance.png")); plt.close()

    shap.summary_plot(sv, Xt, feature_names=names, show=False, max_display=15)
    plt.title("SHAP value distribution")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "shap_beeswarm.png")); plt.close()


def main():
    os.makedirs(FIG, exist_ok=True)
    art, X_tr, X_te, y_tr, y_te = _load()
    print("model_comparison"); fig_model_comparison()
    print("predictions/residuals/error-bands"); fig_predictions(art, X_te, y_te)
    print("shap"); fig_shap(art, X_tr)
    print("figures written to reports/figures/")


if __name__ == "__main__":
    main()
