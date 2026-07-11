"""Data loading, cleaning and feature engineering for the Melbourne AVM.

The raw Melbourne housing extract is messy: heavy missingness in `BuildingArea`
and `YearBuilt`, `Landsize`/`BuildingArea` outliers, and high-cardinality location
fields. This module turns it into a clean, leakage-safe modelling frame and defines
which columns are numeric / low-cardinality / high-cardinality so the training
pipeline can encode each appropriately.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "melbourne_housing.csv")

TARGET = "Price"

# Columns dropped: identifiers / free-text / leakage-prone high-cardinality strings.
DROP_COLS = ["Address", "SellerG", "Date"]

# Feature groups consumed by the ColumnTransformer in train.py.
NUMERIC_FEATURES = [
    "Rooms", "Bedroom2", "Bathroom", "Car", "Landsize", "BuildingArea",
    "Distance", "Propertycount", "Lattitude", "Longtitude",
    "property_age", "rooms_per_bedroom", "bath_per_room",
    "land_to_building", "buildingarea_missing", "yearbuilt_missing",
]
LOWCARD_CATEGORICAL = ["Type", "Method", "Regionname"]
HIGHCARD_CATEGORICAL = ["Suburb", "CouncilArea", "Postcode"]

ALL_FEATURES = NUMERIC_FEATURES + LOWCARD_CATEGORICAL + HIGHCARD_CATEGORICAL


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Ratio that returns NaN (not inf) when the denominator is zero/missing."""
    b = b.replace(0, np.nan)
    return a / b


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic sanity cleaning: drop unusable rows, fix implausible values."""
    df = df.copy()

    # Target must exist and be positive.
    df = df[df[TARGET].notna() & (df[TARGET] > 0)]

    # Landsize == 0 is "unknown", not "zero square metres" -> treat as missing.
    df.loc[df["Landsize"] == 0, "Landsize"] = np.nan
    df.loc[df["BuildingArea"] == 0, "BuildingArea"] = np.nan

    # Clip extreme land/building outliers (data-entry errors) at the 99.5th pct.
    for col in ["Landsize", "BuildingArea"]:
        cap = df[col].quantile(0.995)
        df.loc[df[col] > cap, col] = cap

    # Implausible year built.
    df.loc[(df["YearBuilt"] < 1840) | (df["YearBuilt"] > 2018), "YearBuilt"] = np.nan

    return df.reset_index(drop=True)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Kept purely row-local (no target use) to avoid leakage."""
    df = df.copy()

    sale_year = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.year
    df["property_age"] = (sale_year - df["YearBuilt"]).clip(lower=0)

    df["rooms_per_bedroom"] = _safe_div(df["Rooms"], df["Bedroom2"])
    df["bath_per_room"] = _safe_div(df["Bathroom"], df["Rooms"])
    df["land_to_building"] = _safe_div(df["Landsize"], df["BuildingArea"])

    # Missingness is informative (units rarely report BuildingArea/YearBuilt).
    df["buildingarea_missing"] = df["BuildingArea"].isna().astype(int)
    df["yearbuilt_missing"] = df["YearBuilt"].isna().astype(int)

    # Postcode is categorical location info, not a magnitude.
    df["Postcode"] = df["Postcode"].astype(str)
    return df


def build_frame(path: str = RAW_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) ready for the modelling pipeline."""
    df = engineer(clean(load_raw(path)))
    y = df[TARGET].astype(float)
    X = df[ALL_FEATURES].copy()
    return X, y


def featurize_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the same engineering used at train time to raw rows for inference.

    Unlike `build_frame`, this does not filter on the target, so it works for new
    properties that have no known price.
    """
    df = engineer(df_raw.copy())
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    return df[ALL_FEATURES].copy()


def suburb_reference(path: str = RAW_PATH) -> pd.DataFrame:
    """Per-suburb location defaults used to auto-fill the demo form.

    Returns typical Distance / Postcode / CouncilArea / Regionname / lat-long /
    Propertycount for each suburb so the user only picks a suburb + property traits.
    """
    df = load_raw(path)
    agg = (df.groupby("Suburb")
             .agg(Distance=("Distance", "median"),
                  Postcode=("Postcode", lambda s: s.mode().iloc[0]),
                  CouncilArea=("CouncilArea", lambda s: s.dropna().mode().iloc[0]
                               if not s.dropna().empty else np.nan),
                  Regionname=("Regionname", lambda s: s.mode().iloc[0]),
                  Lattitude=("Lattitude", "median"),
                  Longtitude=("Longtitude", "median"),
                  Propertycount=("Propertycount", "median"))
             .reset_index())
    return agg


if __name__ == "__main__":
    X, y = build_frame()
    print("X shape:", X.shape)
    print("y range: {:,.0f} – {:,.0f}  median {:,.0f}".format(y.min(), y.max(), y.median()))
    print("\nnull counts per feature:")
    print(X.isnull().sum()[X.isnull().sum() > 0])
