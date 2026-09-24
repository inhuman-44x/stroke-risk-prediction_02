"""Stroke risk models with clinical features, CV selection, and calibration.

Uses the existing processed train/test split so results are comparable to
notebooks/model.ipynb. The held-out test set is scored only after the model
is chosen by cross-validated PR-AUC.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"

SMOKE_ORDER = {
    "Unknown": 0,
    "Never smoked": 1,
    "Formerly smoked": 2,
    "Smokes": 3,
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    glucose = out["avg_glucose_level"]
    out["log_glucose"] = np.log1p(glucose)
    out["glucose_mid"] = ((glucose >= 100) & (glucose < 126)).astype(int)
    out["glucose_high"] = (glucose >= 126).astype(int)
    out["age_x_hypertension"] = out["age"] * out["hypertension"]
    out["age_x_heart_disease"] = out["age"] * out["heart_disease"]
    out["age_x_log_glucose"] = out["age"] * out["log_glucose"]
    out["bmi_obese"] = (out["bmi"] >= 30).astype(int)
    out["smoking_ord"] = out["smoking_status"].map(SMOKE_ORDER).astype(float)
    return out


def metrics(y_true, proba) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def main() -> None:
    train = pd.read_parquet(DATA / "train_df.parquet")
    test = pd.read_parquet(DATA / "test_df.parquet")
    train = add_features(train)
    test = add_features(test)

    y_train = train["stroke"].astype(int)
    y_test = test["stroke"].astype(int)
    drop_cols = ["id", "stroke"]
    X_train = train.drop(columns=drop_cols)
    X_test = test.drop(columns=drop_cols)

    cat_cols = ["gender", "ever_married", "work_type", "residence_type", "smoking_status"]
    num_base = ["age", "avg_glucose_level", "bmi", "bmi_missing", "hypertension", "heart_disease"]
    num_extra = [
        "log_glucose",
        "glucose_mid",
        "glucose_high",
        "age_x_hypertension",
        "age_x_heart_disease",
        "age_x_log_glucose",
        "bmi_obese",
        "smoking_ord",
    ]
    num_cols = num_base + num_extra

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    def encoder():
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)

    def baseline_pre():
        return ColumnTransformer(
            [
                ("num", StandardScaler(), num_base),
                ("cat", encoder(), cat_cols),
            ]
        )

    def improved_lr_pre():
        return ColumnTransformer(
            [
                (
                    "age_spline",
                    SplineTransformer(n_knots=5, degree=3, include_bias=False),
                    ["age"],
                ),
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    [c for c in num_cols if c != "age"],
                ),
                ("cat", encoder(), cat_cols),
            ]
        )

    def tree_pre():
        return ColumnTransformer(
            [
                ("num", "passthrough", num_cols),
                ("cat", encoder(), cat_cols),
            ]
        )

    baseline = Pipeline(
        [
            ("preprocessor", baseline_pre()),
            (
                "model",
                LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            ),
        ]
    )

    lr = Pipeline(
        [
            ("preprocessor", improved_lr_pre()),
            (
                "model",
                LogisticRegression(max_iter=2000, random_state=42),
            ),
        ]
    )
    lr_search = RandomizedSearchCV(
        lr,
        param_distributions={"model__C": np.logspace(-3, 2, 12)},
        n_iter=12,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        refit=True,
    )

    xgb = Pipeline(
        [
            ("preprocessor", tree_pre()),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=1,
                    scale_pos_weight=pos_weight,
                ),
            ),
        ]
    )
    xgb_search = RandomizedSearchCV(
        xgb,
        param_distributions={
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__subsample": [0.7, 0.9],
            "model__colsample_bytree": [0.7, 0.9],
            "model__min_child_weight": [3, 5, 10],
            "model__reg_lambda": [1.0, 5.0, 10.0],
            "model__gamma": [0.0, 0.5, 1.0],
        },
        n_iter=24,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        refit=True,
    )

    rf = Pipeline(
        [
            ("preprocessor", tree_pre()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=6,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print(f"Train n={len(y_train)} positives={int(y_train.sum())} scale_pos_weight={pos_weight:.2f}")
    print("Fitting baseline logistic regression...")
    baseline.fit(X_train, y_train)

    print("Tuning improved logistic regression...")
    lr_search.fit(X_train, y_train)
    print("LR", lr_search.best_params_, "CV PR-AUC", round(lr_search.best_score_, 4))

    print("Tuning XGBoost...")
    xgb_search.fit(X_train, y_train)
    print("XGB", xgb_search.best_params_, "CV PR-AUC", round(xgb_search.best_score_, 4))

    print("Fitting random forest...")
    rf.fit(X_train, y_train)
    rf_cv = []
    for train_idx, valid_idx in cv.split(X_train, y_train):
        fold = Pipeline(
            [
                ("preprocessor", tree_pre()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=6,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        fold.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        proba = fold.predict_proba(X_train.iloc[valid_idx])[:, 1]
        rf_cv.append(average_precision_score(y_train.iloc[valid_idx], proba))
    rf_cv_score = float(np.mean(rf_cv))
    print("RF CV PR-AUC", round(rf_cv_score, 4))

    candidates = {
        "baseline_logistic": (baseline, None),
        "spline_logistic": (lr_search.best_estimator_, float(lr_search.best_score_)),
        "xgboost": (xgb_search.best_estimator_, float(xgb_search.best_score_)),
        "random_forest": (rf, rf_cv_score),
    }

    # Baseline CV for a fair row in the table.
    base_cv = []
    for train_idx, valid_idx in cv.split(X_train, y_train):
        fold = Pipeline(
            [
                ("preprocessor", baseline_pre()),
                ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
            ]
        )
        fold.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        proba = fold.predict_proba(X_train.iloc[valid_idx])[:, 1]
        base_cv.append(average_precision_score(y_train.iloc[valid_idx], proba))
    candidates["baseline_logistic"] = (baseline, float(np.mean(base_cv)))

    selectable = {name: pair for name, pair in candidates.items() if name != "baseline_logistic"}
    selected_name = max(selectable, key=lambda name: selectable[name][1])
    selected_model = selectable[selected_name][0]
    print("Selected by CV PR-AUC:", selected_name)

    print("Calibrating selected model...")
    calibrated = CalibratedClassifierCV(selected_model, method="isotonic", cv=cv)
    calibrated.fit(X_train, y_train)

    print("Stacking spline logistic regression and XGBoost...")
    oof_lr = cross_val_predict(
        lr_search.best_estimator_,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    oof_xgb = cross_val_predict(
        xgb_search.best_estimator_,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    stack_X = np.column_stack([oof_lr, oof_xgb])
    stacker = LogisticRegression(max_iter=1000, random_state=42)
    stacker.fit(stack_X, y_train)
    stack_cv = average_precision_score(y_train, stacker.predict_proba(stack_X)[:, 1])

    rows = []
    for name, (model, cv_score) in candidates.items():
        proba = model.predict_proba(X_test)[:, 1]
        row = {"model": name, "cv_pr_auc": cv_score, **metrics(y_test, proba)}
        rows.append(row)
        print(name, row)

    cal_proba = calibrated.predict_proba(X_test)[:, 1]
    cal_row = {
        "model": f"{selected_name}_isotonic",
        "cv_pr_auc": None,
        **metrics(y_test, cal_proba),
    }
    rows.append(cal_row)
    print(cal_row)

    test_stack = np.column_stack(
        [
            lr_search.best_estimator_.predict_proba(X_test)[:, 1],
            xgb_search.best_estimator_.predict_proba(X_test)[:, 1],
        ]
    )
    stack_proba = stacker.predict_proba(test_stack)[:, 1]
    stack_row = {
        "model": "stack_lr_xgb",
        "cv_pr_auc": float(stack_cv),
        **metrics(y_test, stack_proba),
    }
    rows.append(stack_row)
    print(stack_row)

    OUT.mkdir(exist_ok=True)
    payload = {
        "selected_by_cv": selected_name,
        "scale_pos_weight": pos_weight,
        "lr_params": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in lr_search.best_params_.items()},
        "xgb_params": xgb_search.best_params_,
        "results": rows,
    }
    (OUT / "improved_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote", OUT / "improved_results.json")


if __name__ == "__main__":
    main()
