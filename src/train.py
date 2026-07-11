"""Train and evaluate the Automated Valuation Model (AVM).

Pipeline design (leakage-safe, serve-safe):
  raw DataFrame
    -> ColumnTransformer  (median-impute numerics, one-hot low-card cats,
                           cross-fitted TargetEncoder for high-card location cats)
    -> gradient-boosted regressor
  wrapped in TransformedTargetRegressor(log1p / expm1) so `.predict` returns dollars
  from *raw* inputs — the same object is used at train and serve time, so there is
  no train/serve skew.

Steps: compare a model zoo with 5-fold CV -> tune the strongest family with Optuna
-> evaluate on an untouched test split -> calibrate split-conformal prediction
intervals -> persist one artifact for the app.
"""
from __future__ import annotations

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from src.data import (
    HIGHCARD_CATEGORICAL,
    LOWCARD_CATEGORICAL,
    NUMERIC_FEATURES,
    build_frame,
)

warnings.filterwarnings("ignore")

HERE = os.path.dirname(__file__)
REPORTS = os.path.join(HERE, "..", "reports")
MODELS = os.path.join(HERE, "..", "models")
RANDOM_STATE = 42


def build_preprocessor(scale: bool = False) -> ColumnTransformer:
    """Impute numerics, one-hot low-card cats, target-encode high-card location cats."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:  # only the linear baseline needs scaling
        num_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), NUMERIC_FEATURES),
            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=25,
                                  sparse_output=False), LOWCARD_CATEGORICAL),
            # TargetEncoder cross-fits internally -> no target leakage.
            ("tgt", TargetEncoder(random_state=RANDOM_STATE), HIGHCARD_CATEGORICAL),
        ],
        remainder="drop",
    )


def wrap(estimator, scale: bool = False) -> TransformedTargetRegressor:
    """Full raw->dollars estimator with log-target transform."""
    pipe = Pipeline([("prep", build_preprocessor(scale)), ("model", estimator)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def model_zoo() -> dict:
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor

    return {
        "Ridge (baseline)": wrap(Ridge(alpha=1.0), scale=True),
        "RandomForest": wrap(RandomForestRegressor(
            n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE)),
        "XGBoost": wrap(xgb.XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, random_state=RANDOM_STATE)),
        "CatBoost": wrap(CatBoostRegressor(
            iterations=600, learning_rate=0.05, depth=6, verbose=0,
            random_state=RANDOM_STATE)),
        "LightGBM": wrap(lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.05, num_leaves=48, subsample=0.8,
            colsample_bytree=0.8, n_jobs=-1, random_state=RANDOM_STATE, verbose=-1)),
    }


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE_%": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "MedAPE_%": float(np.median(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def compare_zoo(X_tr, y_tr) -> pd.DataFrame:
    """5-fold CV comparison of all candidate models on the training split."""
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
        "MAPE": "neg_mean_absolute_percentage_error",
    }
    rows = []
    for name, est in model_zoo().items():
        print(f"  CV: {name} ...", flush=True)
        res = cross_validate(est, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
        rows.append({
            "model": name,
            "CV_MAE": -res["test_MAE"].mean(),
            "CV_RMSE": -res["test_RMSE"].mean(),
            "CV_R2": res["test_R2"].mean(),
            "CV_MAPE_%": -res["test_MAPE"].mean() * 100,
        })
    return pd.DataFrame(rows).sort_values("CV_MAE").reset_index(drop=True)


def tune_lightgbm(X_tr, y_tr, n_trials: int = 40):
    """Optuna search over LightGBM hyperparameters, optimising CV MAE."""
    import lightgbm as lgb
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = KFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 400, 1500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 20, 120),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
        est = wrap(lgb.LGBMRegressor(
            n_jobs=-1, random_state=RANDOM_STATE, verbose=-1, **params))
        res = cross_validate(est, X_tr, y_tr, cv=cv,
                             scoring="neg_mean_absolute_error", n_jobs=1)
        return -res["test_score"].mean()

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


# Best hyperparameters from the last Optuna run. The app rebuilds from these so
# it never has to unpickle a model across scikit-learn/Python versions.
DEFAULT_PARAMS = {
    "n_estimators": 1489, "learning_rate": 0.022781267283140554, "num_leaves": 38,
    "max_depth": 8, "min_child_samples": 5, "subsample": 0.6817377407847849,
    "colsample_bytree": 0.6272842560912808, "reg_lambda": 2.900609395296397,
}


def split_data():
    X, y = build_frame()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    return X, X_tr, X_te, y_tr, y_te


def fit_final(X_tr, y_tr, best_params):
    """Fit the tuned LightGBM pipeline and calibrate split-conformal intervals."""
    import lightgbm as lgb
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=RANDOM_STATE)
    model = wrap(lgb.LGBMRegressor(
        n_jobs=-1, random_state=RANDOM_STATE, verbose=-1, **best_params))
    model.fit(X_fit, y_fit)
    # nonconformity = absolute residual in log space (matches the modelling target)
    log_res = np.abs(np.log1p(y_cal.values) - np.log1p(model.predict(X_cal)))
    conformal_q = {
        "0.80": float(np.quantile(log_res, 0.80)),
        "0.90": float(np.quantile(log_res, 0.90)),
    }
    return model, conformal_q


def evaluate(model, X_te, y_te, conformal_q):
    test_metrics = metrics(y_te.values, model.predict(X_te))
    log_pred = np.log1p(model.predict(X_te))
    lo = np.expm1(log_pred - conformal_q["0.90"])
    hi = np.expm1(log_pred + conformal_q["0.90"])
    coverage = float(((y_te.values >= lo) & (y_te.values <= hi)).mean())
    return test_metrics, coverage


def build_artifact(best_params=None):
    """Rebuild the full serving artifact from scratch — version-proof for the app.

    The Streamlit app calls this at startup instead of unpickling, so the model is
    always constructed with whatever scikit-learn / LightGBM the runtime installed.
    This removes the cross-version pickle errors that break cloud deployments.
    """
    best_params = best_params or DEFAULT_PARAMS
    X, X_tr, X_te, y_tr, y_te = split_data()
    model, conformal_q = fit_final(X_tr, y_tr, best_params)
    test_metrics, coverage = evaluate(model, X_te, y_te, conformal_q)
    return {
        "model": model,
        "conformal_q": conformal_q,
        "test_metrics": test_metrics,
        "pi_coverage_90": coverage,
        "best_params": best_params,
        "feature_names_out": list(
            model.regressor_.named_steps["prep"].get_feature_names_out()),
        "columns": list(X.columns),
    }


def main(n_trials: int = 40):
    os.makedirs(REPORTS, exist_ok=True)
    os.makedirs(os.path.join(REPORTS, "figures"), exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)

    X, X_tr, X_te, y_tr, y_te = split_data()

    print("1) Model-zoo cross-validation")
    comparison = compare_zoo(X_tr, y_tr)
    comparison.to_csv(os.path.join(REPORTS, "model_comparison.csv"), index=False)
    print(comparison.to_string(index=False))

    print("\n2) Optuna tuning of LightGBM")
    best_params, best_cv_mae = tune_lightgbm(X_tr, y_tr, n_trials=n_trials)
    print("  best CV MAE: {:,.0f}".format(best_cv_mae))
    print("  best params:", best_params)

    print("\n3) Fit tuned model, calibrate intervals, evaluate")
    artifact = build_artifact(best_params)
    test_metrics, coverage = artifact["test_metrics"], artifact["pi_coverage_90"]
    for k, v in test_metrics.items():
        print(f"  {k}: {v:,.3f}")
    print(f"  Empirical 90% PI coverage: {coverage:.1%}")
    pd.DataFrame([test_metrics]).to_csv(os.path.join(REPORTS, "metrics.csv"), index=False)

    print("\n4) Persist artifact + params")
    joblib.dump(artifact, os.path.join(MODELS, "avm_pipeline.joblib"))
    with open(os.path.join(MODELS, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    summary = {
        "test_metrics": test_metrics,
        "pi_coverage_90": coverage,
        "cv_best_mae": best_cv_mae,
        "best_model": "LightGBM (Optuna-tuned)",
    }
    with open(os.path.join(REPORTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("  saved models/avm_pipeline.joblib, models/best_params.json and reports/*")
    return artifact


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    main(n_trials=n)
