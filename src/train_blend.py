"""Second pass: unweighted XGBoost and a logistic blend with the baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

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
    out["glucose_high"] = (glucose >= 126).astype(int)
    out["age_x_hypertension"] = out["age"] * out["hypertension"]
    out["age_x_heart_disease"] = out["age"] * out["heart_disease"]
    out["age_x_log_glucose"] = out["age"] * out["log_glucose"]
    out["smoking_ord"] = out["smoking_status"].map(SMOKE_ORDER).astype(float)
    return out


def score(name, y, proba, cv_pr=None):
    row = {
        "model": name,
        "cv_pr_auc": None if cv_pr is None else float(cv_pr),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
    }
    print(row)
    return row


def main() -> None:
    train = add_features(pd.read_parquet(DATA / "train_df.parquet"))
    test = add_features(pd.read_parquet(DATA / "test_df.parquet"))
    y_train = train["stroke"].astype(int)
    y_test = test["stroke"].astype(int)
    X_train = train.drop(columns=["id", "stroke"])
    X_test = test.drop(columns=["id", "stroke"])

    cat_cols = ["gender", "ever_married", "work_type", "residence_type", "smoking_status"]
    num_base = ["age", "avg_glucose_level", "bmi", "bmi_missing", "hypertension", "heart_disease"]
    num_extra = num_base + [
        "log_glucose",
        "glucose_high",
        "age_x_hypertension",
        "age_x_heart_disease",
        "age_x_log_glucose",
        "smoking_ord",
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def enc():
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)

    def lr_pipe():
        return Pipeline(
            [
                (
                    "preprocessor",
                    ColumnTransformer(
                        [
                            ("num", StandardScaler(), num_base),
                            ("cat", enc(), cat_cols),
                        ]
                    ),
                ),
                ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
            ]
        )

    def xgb_pipe():
        return Pipeline(
            [
                (
                    "preprocessor",
                    ColumnTransformer(
                        [
                            ("num", "passthrough", num_extra),
                            ("cat", enc(), cat_cols),
                        ]
                    ),
                ),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_weight=5,
                        reg_lambda=5.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def rf_pipe():
        return Pipeline(
            [
                (
                    "preprocessor",
                    ColumnTransformer(
                        [
                            ("num", "passthrough", num_extra),
                            ("cat", enc(), cat_cols),
                        ]
                    ),
                ),
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

    print("Out-of-fold predictions...")
    oof_lr = cross_val_predict(lr_pipe(), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    oof_xgb = cross_val_predict(xgb_pipe(), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    oof_rf = cross_val_predict(rf_pipe(), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

    rows = []
    for name, oof in [("logistic", oof_lr), ("xgboost_unweighted", oof_xgb), ("random_forest", oof_rf)]:
        rows.append({"model": name, "cv_pr_auc": float(average_precision_score(y_train, oof))})
        print(name, "OOF PR-AUC", rows[-1]["cv_pr_auc"])

    stack_X = np.column_stack([oof_lr, oof_xgb, oof_rf])
    # Score the blend with a nested-style loop so the meta-learner is not fit on its own labels.
    blend_oof = np.zeros(len(y_train))
    for tr, va in cv.split(stack_X, y_train):
        meta = LogisticRegression(max_iter=1000, random_state=42)
        meta.fit(stack_X[tr], y_train.iloc[tr])
        blend_oof[va] = meta.predict_proba(stack_X[va])[:, 1]
    blend_cv = float(average_precision_score(y_train, blend_oof))
    print("blend OOF PR-AUC", blend_cv)

    lr = lr_pipe().fit(X_train, y_train)
    xgb = xgb_pipe().fit(X_train, y_train)
    rf = rf_pipe().fit(X_train, y_train)
    meta = LogisticRegression(max_iter=1000, random_state=42)
    meta.fit(stack_X, y_train)

    rows = [
        score("logistic", y_test, lr.predict_proba(X_test)[:, 1], average_precision_score(y_train, oof_lr)),
        score("xgboost_unweighted", y_test, xgb.predict_proba(X_test)[:, 1], average_precision_score(y_train, oof_xgb)),
        score("random_forest", y_test, rf.predict_proba(X_test)[:, 1], average_precision_score(y_train, oof_rf)),
    ]
    test_stack = np.column_stack(
        [
            lr.predict_proba(X_test)[:, 1],
            xgb.predict_proba(X_test)[:, 1],
            rf.predict_proba(X_test)[:, 1],
        ]
    )
    rows.append(score("blend_lr_xgb_rf", y_test, meta.predict_proba(test_stack)[:, 1], blend_cv))

    out = ROOT / "outputs" / "blend_results.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
