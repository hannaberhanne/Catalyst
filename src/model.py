# model.py
import json
import os
from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
import joblib

DATA_PATH = "data/processed/tickets_cleaned.csv"
MODEL_PATH = "models/random_forest.pkl"
MODEL_META_PATH = "models/model_metadata.json"
FEATURE_IMPORTANCE_PATH = "models/feature_importance.csv"
DEFAULT_THRESHOLD = 0.35
CALIBRATION_SPLIT_RANDOM_STATE = 2024
CALIBRATION_SPLIT_SIZE = 0.2


def _ensure_model_dir():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def _write_metadata(metadata):
    _ensure_model_dir()
    with open(MODEL_META_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def _select_threshold(y_true, y_proba, min_recall=0.6, fallback=DEFAULT_THRESHOLD):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_candidates = [
        (p, r, t)
        for p, r, t in zip(
            precision[:-1], recall[:-1], thresholds
        )
        if r >= min_recall
    ]
    if not pr_candidates:
        return fallback
    return max(pr_candidates, key=lambda x: x[0])[2]


def main():
    df = pd.read_csv(DATA_PATH)

    # Split features + target
    X = df.drop(columns=["SLA_exceeded"])
    y = df["SLA_exceeded"]

    LEAKAGE = [
        "Resolution_days",
        "To Resolution (Elapsed)",
        "To Resolution (Business)",
        "Service Time To Closure",
        "Closed At",
        "Resolved At"
    ]
    X = X.drop(columns=[c for c in LEAKAGE if c in X.columns], errors="ignore")

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Base model configuration
    base_model = RandomForestClassifier(
        n_estimators=600,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight={0: 1, 1: 5},
        n_jobs=-1,
        random_state=42
    )

    # Run cross-validation on the training fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        base_model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision_macro",
            "recall": "recall_macro"
        },
        n_jobs=-1
    )

    X_train_main, X_calib, y_train_main, y_calib = train_test_split(
        X_train,
        y_train,
        test_size=CALIBRATION_SPLIT_SIZE,
        random_state=CALIBRATION_SPLIT_RANDOM_STATE,
        stratify=y_train
    )

    base_model.fit(X_train_main, y_train_main)

    fi = pd.DataFrame({
        "feature": X.columns,
        "importance": base_model.feature_importances_
    }).sort_values("importance", ascending=False)
    fi.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    calibrator = CalibratedClassifierCV(base_model, cv="prefit", method="sigmoid")
    calibrator.fit(X_calib, y_calib)

    y_proba = calibrator.predict_proba(X_test)[:, 1]
    selected_threshold = _select_threshold(y_test, y_proba)
    y_pred = (y_proba >= selected_threshold).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nReport:")
    print(classification_report(y_test, y_pred))
    print(f"\nSelected threshold (recall>=0.6 max precision): {selected_threshold:.2f}")

    # Save trained model
    _ensure_model_dir()
    model_payload = {"calibrator": calibrator, "base": base_model}
    joblib.dump(model_payload, MODEL_PATH)
    print("\nModel saved →", MODEL_PATH)

    # Save metadata for reproducibility
    metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data_path": DATA_PATH,
        "data_rows": len(df),
        "feature_count": X.shape[1],
        "model_path": MODEL_PATH,
        "cv": {
            metric: {
                "mean": float(cv_results[f"test_{metric}"].mean()),
                "std": float(cv_results[f"test_{metric}"].std())
            }
            for metric in ("accuracy", "precision", "recall")
        },
        "test_metrics": {
            "accuracy": float(report["accuracy"]),
            "precision_macro": float(report["macro avg"]["precision"]),
            "recall_macro": float(report["macro avg"]["recall"])
        }
    }
    metadata["threshold"] = selected_threshold
    metadata["calibrated"] = True
    metadata["calibration_split"] = CALIBRATION_SPLIT_SIZE
    metadata["calibration_random_state"] = CALIBRATION_SPLIT_RANDOM_STATE
    metadata["top_features"] = fi.head(10).to_dict(orient="records")
    _write_metadata(metadata)


if __name__ == "__main__":
    main()
