"""Generate the project slide deck: reports/AVM_house_valuation.pptx.

A clean, bright 16:9 deck: problem & market -> data & complexity -> approach &
tech stack -> how we tackled it -> results -> explainability -> what we built.
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
ACCENT = RGBColor(0x25, 0x63, 0xEB)
GREY = RGBColor(0x64, 0x74, 0x8B)
LIGHT = RGBColor(0xF1, 0xF5, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

with open(os.path.join(REPORTS, "summary.json")) as f:
    S = json.load(f)
M = S["test_metrics"]


def slide():
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = WHITE
    return s


def bar(s, color=ACCENT):
    b = s.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(0.18))
    b.fill.solid(); b.fill.fore_color.rgb = color; b.line.fill.background()


def text(s, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    for i, (txt, size, color, bold) in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = txt
        r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold
        r.font.name = "Calibri"
    return tb


def bullets(s, x, y, w, h, items, size=16, gap=6):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        r = p.add_run(); r.text = "•  " + it
        r.font.size = Pt(size); r.font.color.rgb = INK; r.font.name = "Calibri"
    return tb


def title(s, t, sub=None):
    bar(s)
    text(s, 0.6, 0.45, 12, 1, [(t, 30, BLUE, True)])
    if sub:
        text(s, 0.6, 1.25, 12, 0.6, [(sub, 15, GREY, False)])


def pill_row(s, x, y, labels):
    cx = x
    for lab in labels:
        w = 0.28 + 0.11 * len(lab)
        box = s.shapes.add_shape(5, Inches(cx), Inches(y), Inches(w), Inches(0.42))
        box.fill.solid(); box.fill.fore_color.rgb = RGBColor(0xDB, 0xEA, 0xFE)
        box.line.fill.background()
        tf = box.text_frame; tf.word_wrap = False
        r = tf.paragraphs[0].add_run(); r.text = lab
        r.font.size = Pt(12); r.font.bold = True; r.font.color.rgb = BLUE
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        cx += w + 0.18


# ---------------------------------------------------------------- 1 title
s = slide()
band = s.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
band.fill.solid(); band.fill.fore_color.rgb = INK; band.line.fill.background()
text(s, 0.8, 2.3, 11.7, 1.4, [("🏠  Melbourne Automated Valuation Model", 40, WHITE, True)])
text(s, 0.85, 3.7, 11.5, 1,
     [("Explainable, uncertainty-aware property pricing — the AVM class behind "
       "Zestimate, Redfin and lender valuation engines.", 18,
       RGBColor(0xCB, 0xD5, 0xE1), False)])
text(s, 0.85, 5.4, 11, 0.6,
     [("LightGBM + Optuna   ·   Conformal prediction intervals   ·   "
       "SHAP explainability   ·   Streamlit", 15, RGBColor(0x93, 0xC5, 0xFD), True)])
text(s, 0.85, 6.5, 11, 0.5, [("Manik Kaura — Data Science portfolio", 13, GREY, False)])

# ---------------------------------------------------------------- 2 problem
s = slide(); title(s, "The problem — valuation at scale",
                   "Banks, insurers, iBuyers and marketplaces must value millions of homes instantly.")
bullets(s, 0.7, 1.9, 12, 4, [
    "An Automated Valuation Model (AVM) prices a property from its attributes — no human appraiser per home.",
    "Accurate across a huge range: this market spans $85k to $9M.",
    "Honest about uncertainty: a single number is unsafe for underwriting — decisions need a calibrated range.",
    "Explainable: regulators and customers must see WHY a home was valued as it was.",
], size=18, gap=14)
pill_row(s, 0.7, 5.9, ["Proptech", "Mortgage risk", "iBuying", "Insurance"])

# ---------------------------------------------------------------- 3 data & complexity
s = slide(); title(s, "The data — and why it's hard",
                   "Melbourne housing: 13,580 sales × 21 fields. Exactly as messy as real estate data.")
bullets(s, 0.7, 1.9, 5.9, 4.6, [
    "BuildingArea 47% missing, YearBuilt 40%, CouncilArea 10%.",
    "Right-skewed target across two orders of magnitude.",
    "314 suburbs & 200+ postcodes — strongest signal, but a leakage trap.",
    "Outliers & data-entry errors: 0 m² land, impossible build years.",
    "Must serve predictions from RAW inputs, not pre-baked arrays.",
], size=16, gap=12)
s.shapes.add_picture(os.path.join(FIG, "error_by_price.png"),
                     Inches(6.9), Inches(2.2), width=Inches(5.9))

# ---------------------------------------------------------------- 4 approach
s = slide(); title(s, "Approach & tech-stack logic",
                   "One scikit-learn pipeline: raw property row → dollars + interval + explanation.")
text(s, 0.7, 1.8, 12, 0.6,
     [("raw row → ColumnTransformer → LightGBM (log-target) → "
       "TransformedTargetRegressor → $  +  interval  +  SHAP", 15, ACCENT, True)])
bullets(s, 0.7, 2.6, 12, 4, [
    "log1p(Price) target — proportional error for a skewed, multiplicative quantity.",
    "Median impute + explicit missingness flags (a missing BuildingArea usually means a unit).",
    "One-hot for Type/Method/Region; cross-fitted TargetEncoder for Suburb/Postcode/Council (no leakage).",
    "LightGBM won a 5-model zoo (vs XGBoost, CatBoost, RandomForest, Ridge).",
    "Optuna TPE search — 40 trials over 8 hyperparameters, optimising CV MAE.",
    "Split-conformal intervals for distribution-free, verified-coverage uncertainty.",
], size=15, gap=10)

# ---------------------------------------------------------------- 5 how tackled
s = slide(); title(s, "How each challenge was tackled")
bullets(s, 0.7, 1.9, 12, 4.6, [
    "Missingness → median impute + *_missing indicators (turn a gap into signal).",
    "Skew → log-target modelling; report both dollar and percentage error.",
    "Location leakage → TargetEncoder cross-fitting; all encoding lives inside the CV pipeline.",
    "Outliers → 0-values treated as missing; land/building capped at the 99.5th percentile.",
    "Train/serve skew → a single Pipeline does clean→encode→predict, identical in training and app.",
    "Trust → conformal range on every quote + SHAP explanation of every driver.",
], size=17, gap=14)

# ---------------------------------------------------------------- 6 results
s = slide(); title(s, "Results — held-out test set",
                   "Untouched 20% split; intervals calibrated on a separate fold.")
cards = [("R²", f"{M['R2']:.3f}"), ("MAPE", f"{M['MAPE_%']:.1f}%"),
         ("Median APE", f"{M['MedAPE_%']:.1f}%"), ("MAE", f"${M['MAE']:,.0f}"),
         ("90% PI coverage", f"{S['pi_coverage_90']*100:.1f}%")]
cw, gap0 = 2.3, 0.18
x = 0.7
for lab, val in cards:
    box = s.shapes.add_shape(5, Inches(x), Inches(2.0), Inches(cw), Inches(1.4))
    box.fill.solid(); box.fill.fore_color.rgb = LIGHT; box.line.color.rgb = RGBColor(0xE2, 0xE8, 0xF0)
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = lab; r.font.size = Pt(12); r.font.color.rgb = GREY
    p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run(); r2.text = val; r2.font.size = Pt(24); r2.font.bold = True; r2.font.color.rgb = BLUE
    x += cw + gap0
s.shapes.add_picture(os.path.join(FIG, "actual_vs_predicted.png"),
                     Inches(3.9), Inches(3.7), height=Inches(3.4))

# ---------------------------------------------------------------- 7 explainability
s = slide(); title(s, "Explainability — SHAP",
                   "Global drivers and a per-property breakdown power the demo's 'why this valuation?'")
s.shapes.add_picture(os.path.join(FIG, "shap_importance.png"),
                     Inches(0.7), Inches(1.9), height=Inches(4.9))
bullets(s, 7.1, 2.3, 5.6, 4, [
    "Type (house vs unit) and distance to the CBD dominate.",
    "Postcode & suburb encode fine-grained location value.",
    "Land size and bathrooms add property-level detail.",
    "Per-prediction: each factor shown as a ± % effect on that home's price.",
], size=16, gap=14)

# ---------------------------------------------------------------- 8 what we built
s = slide(); title(s, "What we built",
                   "A bright, interactive Streamlit demo + a reproducible modelling pipeline.")
if os.path.exists(os.path.join(ASSETS, "demo_valuation.png")):
    s.shapes.add_picture(os.path.join(ASSETS, "demo_valuation.png"),
                         Inches(0.6), Inches(1.9), width=Inches(7.6))
bullets(s, 8.5, 2.2, 4.4, 4.5, [
    "Enter a property → instant valuation.",
    "Calibrated 80/90% price range.",
    "'Why this valuation?' SHAP drivers.",
    "Model-performance dashboard page.",
    "Modular src/ pipeline, one-command retrain.",
], size=15, gap=12)

out = os.path.join(REPORTS, "AVM_house_valuation.pptx")
prs.save(out)
print("saved", out, "—", len(prs.slides._sldIdLst), "slides")
