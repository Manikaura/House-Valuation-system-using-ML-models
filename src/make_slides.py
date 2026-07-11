"""Generate the 5-part slide deck: reports/AVM_house_valuation.pptx.

Structure: Title → Problem → Approach & tech stack → Key findings →
Model performance → Recommendations & impact. Bright, warm, minimal.
Run:  python -m src.make_slides
"""
from __future__ import annotations

import json
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

HERE = os.path.dirname(__file__)
REPORTS = os.path.join(HERE, "..", "reports")
FIG = os.path.join(REPORTS, "figures")
ASSETS = os.path.join(HERE, "..", "assets")

INK = RGBColor(0x0F, 0x17, 0x2A)
BLUE = RGBColor(0x1E, 0x3A, 0x8A)
ACCENT = RGBColor(0xF5, 0x9E, 0x0B)
GREY = RGBColor(0x64, 0x74, 0x8B)
LIGHT = RGBColor(0xFB, 0xF7, 0xEE)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

S = json.load(open(os.path.join(REPORTS, "summary.json")))
M = S["test_metrics"]


def slide(bg=WHITE):
    s = prs.slides.add_slide(BLANK)
    f = s.background.fill; f.solid(); f.fore_color.rgb = bg
    return s


def box(s, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for i, (txt, size, color, bold) in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = txt
        r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold
        r.font.name = "Calibri"
    return tb


def bullets(s, x, y, w, h, items, size=16, gap=8, color=INK):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        r = p.add_run(); r.text = "•  " + it
        r.font.size = Pt(size); r.font.color.rgb = color; r.font.name = "Calibri"
    return tb


def header(s, kicker, title):
    b = s.shapes.add_shape(1, Inches(0), Inches(0.55), Inches(0.22), Inches(1.05))
    b.fill.solid(); b.fill.fore_color.rgb = ACCENT; b.line.fill.background()
    box(s, 0.55, 0.5, 12, 0.4, [(kicker.upper(), 13, ACCENT, True)])
    box(s, 0.55, 0.85, 12.2, 1.0, [(title, 27, BLUE, True)])


# ---------------------------------------------------------------- 1 title
s = slide(INK)
box(s, 0.9, 2.3, 11.6, 1.3, [("🏠  Automated Valuation Model", 40, WHITE, True)])
box(s, 0.95, 3.7, 11.4, 1.0,
    [("Explainable, uncertainty-aware residential property valuation — the AVM class "
      "behind Zestimate, Redfin and lender pricing engines.", 18, RGBColor(0xE2, 0xE8, 0xF0), False)])
box(s, 0.95, 5.3, 11.4, 0.5,
    [("LightGBM + Optuna · SHAP · conformal prediction intervals · Streamlit", 15, ACCENT, True)])
box(s, 0.95, 6.4, 11.4, 0.5, [("Manik Kaura — Data Science portfolio", 12, GREY, False)])

# ---------------------------------------------------------------- 2 problem
s = slide()
header(s, "Problem statement", "Value millions of homes, instantly")
bullets(s, 0.7, 1.95, 12.0, 4.4, [
    "Banks, insurers, iBuyers and marketplaces need an instant price for every home — no human "
    "appraiser per property.",
    "Accuracy must hold across a wide range (this market: $85k → $9M).",
    "A single number is unsafe for underwriting — decisions need a calibrated range.",
    "Regulators and customers require knowing WHY a home was valued as it was.",
], size=18, gap=15)

# ---------------------------------------------------------------- 3 approach
s = slide()
header(s, "Approach · logic · tech stack", "One pipeline: raw row → $ + interval + why")
bullets(s, 0.7, 1.95, 7.4, 4.6, [
    "Model log(price) — proportional error for a skewed, multiplicative quantity.",
    "Median impute + missingness flags (a missing BuildingArea usually means a unit).",
    "One-hot for type/method/region; cross-fitted TargetEncoder for suburb/postcode (no leakage).",
    "LightGBM won a 5-model zoo (vs XGBoost, CatBoost, RandomForest, Ridge).",
    "Optuna TPE tuning over 8 hyperparameters; held-out test evaluation.",
    "Split-conformal prediction intervals + SHAP explanations.",
], size=14, gap=9)
box(s, 8.4, 2.0, 4.4, 0.4, [("TECH STACK", 12, ACCENT, True)])
bullets(s, 8.4, 2.4, 4.5, 4, [
    "Python · pandas · NumPy", "scikit-learn pipelines", "LightGBM / XGBoost / CatBoost",
    "Optuna", "SHAP", "Streamlit · Plotly",
], size=14, gap=9)

# ---------------------------------------------------------------- 4 key findings
s = slide()
header(s, "Key findings", "What drives a valuation")
bullets(s, 0.7, 1.9, 5.7, 4.5, [
    "Property type (house vs unit) and distance to the CBD dominate.",
    "Postcode & suburb encode fine-grained location value.",
    "Land size and bathrooms add property-level detail.",
    "Missingness is itself signal — flagged, not dropped.",
    "Per-prediction SHAP shows each factor as a ± % effect on that home.",
], size=15, gap=11)
s.shapes.add_picture(os.path.join(FIG, "shap_importance.png"),
                     Inches(6.7), Inches(1.9), height=Inches(4.9))

# ---------------------------------------------------------------- 5 model performance
s = slide()
header(s, "Model performance", "Held-out test set")
cards = [("R²", f"{M['R2']:.3f}"), ("MAPE", f"{M['MAPE_%']:.1f}%"),
         ("Median APE", f"{M['MedAPE_%']:.1f}%"), ("MAE", f"${M['MAE']:,.0f}"),
         ("90% PI coverage", f"{S['pi_coverage_90']*100:.1f}%")]
x = 0.7
for lab, val in cards:
    c = s.shapes.add_shape(5, Inches(x), Inches(2.0), Inches(2.3), Inches(1.4))
    c.fill.solid(); c.fill.fore_color.rgb = LIGHT; c.line.color.rgb = RGBColor(0xE2, 0xE8, 0xF0)
    tf = c.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = lab; r.font.size = Pt(12); r.font.color.rgb = GREY
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run(); r2.text = val; r2.font.size = Pt(23); r2.font.bold = True; r2.font.color.rgb = BLUE
    x += 2.48
s.shapes.add_picture(os.path.join(FIG, "actual_vs_predicted.png"),
                     Inches(4.0), Inches(3.7), height=Inches(3.4))

# ---------------------------------------------------------------- 6 recommendations & impact
s = slide()
header(s, "Recommendations & impact", "From valuation to decision")
bullets(s, 0.7, 1.95, 12.0, 4.4, [
    "Instant, explainable valuations for pricing, lending and portfolio monitoring.",
    "Prediction intervals let underwriting act on risk, not just a point estimate.",
    "SHAP explanations support regulatory/customer transparency requirements.",
    "Median error ~10% with calibrated 90% coverage — decision-ready out of the box.",
    "Extend with time (sale-date) trends, geospatial features and comparable-sales signals.",
], size=17, gap=13)

out = os.path.join(REPORTS, "AVM_house_valuation.pptx")
prs.save(out)
print("saved", out, "—", len(prs.slides._sldIdLst), "slides")
