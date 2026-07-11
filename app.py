"""Melbourne AVM — interactive Automated Valuation Model demo.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import featurize_raw, suburb_reference, load_raw

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "models", "avm_pipeline.joblib")

st.set_page_config(page_title="Melbourne AVM", page_icon="🏠", layout="wide")

# ----------------------------------------------------------------------------- style
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.metric-card {background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;
  padding:1.1rem 1.3rem;box-shadow:0 1px 2px rgba(0,0,0,.04);}
.big {font-size:2.1rem;font-weight:800;color:#1e3a8a;line-height:1.1;}
.sub {color:#64748b;font-size:.85rem;text-transform:uppercase;letter-spacing:.04em;}
.pill {display:inline-block;background:#dbeafe;color:#1e40af;border-radius:999px;
  padding:.15rem .7rem;font-size:.8rem;font-weight:600;margin-right:.4rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference():
    ref = suburb_reference()
    raw = load_raw()
    return ref, raw


art = load_artifact()
ref, raw = load_reference()
reg = art["model"].regressor_
prep, model = reg.named_steps["prep"], reg.named_steps["model"]


def predict_with_interval(row: pd.DataFrame, level: str = "0.90"):
    price = float(art["model"].predict(row)[0])
    q = art["conformal_q"][level]
    log_p = np.log1p(price)
    lo, hi = np.expm1(log_p - q), np.expm1(log_p + q)
    return price, lo, hi


def top_drivers(row: pd.DataFrame, k: int = 8):
    """Per-prediction SHAP, expressed as multiplicative price effects (log model)."""
    import shap
    Xt = prep.transform(row)
    sv = shap.TreeExplainer(model).shap_values(Xt)[0]
    names = art["feature_names_out"]
    order = np.argsort(np.abs(sv))[::-1][:k]
    return [(names[i].split("__")[-1], float(np.expm1(sv[i]) * 100)) for i in order]


# ----------------------------------------------------------------------------- header
st.title("🏠 Melbourne Automated Valuation Model")
st.markdown(
    "<span class='pill'>LightGBM + Optuna</span>"
    "<span class='pill'>Conformal intervals</span>"
    "<span class='pill'>SHAP explainability</span>"
    "&nbsp; Instant, explainable property valuations with calibrated uncertainty.",
    unsafe_allow_html=True)

page = st.sidebar.radio("View", ["💰 Value a property", "📊 Model & performance"])
m = art["test_metrics"]

# ===================================================================== VALUATION PAGE
if page.startswith("💰"):
    st.sidebar.header("Property details")
    suburbs = sorted(ref["Suburb"].unique())
    default_ix = suburbs.index("Reservoir") if "Reservoir" in suburbs else 0
    suburb = st.sidebar.selectbox("Suburb", suburbs, index=default_ix)
    ptype = st.sidebar.selectbox("Type", ["h — House", "u — Unit", "t — Townhouse"])
    type_code = ptype.split(" ")[0]

    rooms = st.sidebar.slider("Rooms", 1, 8, 3)
    bedroom2 = st.sidebar.slider("Bedrooms", 0, 8, 3)
    bathroom = st.sidebar.slider("Bathrooms", 1, 5, 1)
    car = st.sidebar.slider("Car spaces", 0, 6, 1)
    landsize = st.sidebar.number_input("Land size (m²)", 0, 3000, 400, step=10)
    building = st.sidebar.number_input("Building area (m²)", 0, 800, 130, step=10)
    year_built = st.sidebar.number_input("Year built", 1850, 2018, 1970, step=1)
    level = st.sidebar.select_slider("Confidence interval", ["0.80", "0.90"], "0.90")

    r = ref[ref["Suburb"] == suburb].iloc[0]
    raw_row = pd.DataFrame([{
        "Suburb": suburb, "Type": type_code, "Method": "S",
        "Rooms": rooms, "Bedroom2": bedroom2, "Bathroom": bathroom, "Car": car,
        "Landsize": landsize or np.nan, "BuildingArea": building or np.nan,
        "YearBuilt": year_built, "Date": "01-06-2017",
        "Distance": r["Distance"], "Postcode": int(r["Postcode"]),
        "CouncilArea": r["CouncilArea"], "Regionname": r["Regionname"],
        "Lattitude": r["Lattitude"], "Longtitude": r["Longtitude"],
        "Propertycount": r["Propertycount"],
    }])
    row = featurize_raw(raw_row)
    price, lo, hi = predict_with_interval(row, level)

    c1, c2, c3 = st.columns([1.2, 1, 1])
    c1.markdown(f"<div class='metric-card'><div class='sub'>Estimated value</div>"
                f"<div class='big'>${price:,.0f}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='sub'>{int(float(level)*100)}% range</div>"
                f"<div class='big' style='font-size:1.3rem'>${lo:,.0f}<br>– ${hi:,.0f}</div></div>",
                unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='sub'>Typical error (MedAPE)</div>"
                f"<div class='big'>{m['MedAPE_%']:.1f}%</div></div>", unsafe_allow_html=True)

    st.subheader("Why this valuation?")
    st.caption("Each factor's multiplicative effect on the estimate (SHAP, log-price model).")
    drivers = top_drivers(row)
    labels = [d[0] for d in drivers][::-1]
    vals = [d[1] for d in drivers][::-1]
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=colors,
                           text=[f"{v:+.1f}%" for v in vals], textposition="outside"))
    fig.update_layout(height=380, margin=dict(l=10, r=30, t=10, b=10),
                      xaxis_title="Effect on price (%)", plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📍 Location"):
        st.map(pd.DataFrame({"lat": [r["Lattitude"]], "lon": [r["Longtitude"]]}), zoom=11)

# ================================================================== PERFORMANCE PAGE
else:
    st.subheader("Held-out test performance")
    cols = st.columns(4)
    cards = [("R²", f"{m['R2']:.3f}"), ("MAPE", f"{m['MAPE_%']:.1f}%"),
             ("Median APE", f"{m['MedAPE_%']:.1f}%"), ("MAE", f"${m['MAE']:,.0f}")]
    for col, (lab, val) in zip(cols, cards):
        col.markdown(f"<div class='metric-card'><div class='sub'>{lab}</div>"
                     f"<div class='big'>{val}</div></div>", unsafe_allow_html=True)
    st.caption(f"90% prediction-interval coverage on unseen data: "
               f"**{art['pi_coverage_90']:.1%}** (target 90%).")

    fig_dir = os.path.join(HERE, "reports", "figures")
    a, b = st.columns(2)
    a.image(os.path.join(fig_dir, "model_comparison.png"), use_container_width=True)
    b.image(os.path.join(fig_dir, "actual_vs_predicted.png"), use_container_width=True)
    c, d = st.columns(2)
    c.image(os.path.join(fig_dir, "shap_importance.png"), use_container_width=True)
    d.image(os.path.join(fig_dir, "error_by_price.png"), use_container_width=True)

    st.subheader("Model comparison (5-fold CV)")
    st.dataframe(pd.read_csv(os.path.join(HERE, "reports", "model_comparison.csv")),
                 use_container_width=True, hide_index=True)

st.sidebar.caption("Data: Melbourne housing (Kaggle). Educational demo, not financial advice.")
