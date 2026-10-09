import json
import os

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score
)

DATA_PATH = "../data/processed/tickets_cleaned.csv"
MODEL_PATH = "../models/random_forest.pkl"
MODEL_META_PATH = "../models/model_metadata.json"
DEFAULT_THRESHOLD = 0.35

REPORTS_DIR = "../reports/"
FIG_DIR = "../reports/figures/"

LEAKAGE = [
    "Resolution_days",
    "To Resolution (Elapsed)",
    "To Resolution (Business)",
    "Service Time To Closure",
    "Closed At",
    "Resolved At"
]


def _ensure_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def _load_threshold():
    try:
        with open(MODEL_META_PATH, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
            return metadata.get("threshold", DEFAULT_THRESHOLD)
    except FileNotFoundError:
        return DEFAULT_THRESHOLD


def main():
    _ensure_dirs()
    threshold = _load_threshold()

    df = pd.read_csv(DATA_PATH)
    loaded_model = joblib.load(MODEL_PATH)
    if isinstance(loaded_model, dict) and "calibrator" in loaded_model:
        model = loaded_model["calibrator"]
        base_estimator = loaded_model.get("base")
    else:
        model = loaded_model
        base_estimator = getattr(
            model,
            "base_estimator_",
            getattr(model, "base_estimator", model)
        )

    X = df.drop(columns=["SLA_exceeded"], errors="ignore")
    X = X.drop(columns=[c for c in LEAKAGE if c in X.columns], errors="ignore")
    y = df["SLA_exceeded"]

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
        y, y_proba
    )
    pr_candidates = [
        (p, r, t)
        for p, r, t in zip(
            precision_curve[:-1], recall_curve[:-1], pr_thresholds
        )
        if r >= 0.6
    ]
    best_candidate = max(
        pr_candidates,
        key=lambda x: x[0],
        default=(precision_curve[0], recall_curve[0], threshold),
    )
    recommended_precision, recommended_recall, recommended_threshold = best_candidate
    print(f"Recommended threshold (recall>=0.6, max precision): {recommended_threshold:.2f}")

    pr_fig, pr_ax = plt.subplots(figsize=(6, 5))
    pr_ax.plot(recall_curve, precision_curve, label="Precision / Recall")
    pr_ax.scatter(
        recommended_recall,
        recommended_precision,
        color="crimson",
        label=f"Threshold {recommended_threshold:.2f}",
    )
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_title("Precision / Recall Curve")
    pr_ax.legend()
    pr_fig.tight_layout()
    pr_fig.savefig(FIG_DIR + "precision_recall_curve.png", dpi=300)
    plt.close(pr_fig)
    print("Saved: precision_recall_curve.png")

    report_text = classification_report(y, y_pred)
    with open(REPORTS_DIR + "classification_report.txt", "w") as f:
        f.write(report_text)
    print("Saved: classification_report.txt")

    metrics_summary = f"""
Accuracy:  {accuracy_score(y, y_pred):.4f}
Precision: {precision_score(y, y_pred):.4f}
Recall:    {recall_score(y, y_pred):.4f}
F1 Score:  {f1_score(y, y_pred):.4f}
AUC:       {roc_auc_score(y, y_proba):.4f}
"""
    with open(REPORTS_DIR + "metrics_summary.txt", "w") as f:
        f.write(metrics_summary)
    print("Saved: metrics_summary.txt")

    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
    plt.title("Confusion Matrix — Catalyst Model", fontsize=16)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIG_DIR + "confusion_matrix.png", dpi=300)
    plt.close()
    print("Saved: confusion_matrix.png")

    feature_importances = getattr(base_estimator, "feature_importances_", None)
    importance_path = os.path.join("..", "models", "feature_importance.csv")
    if feature_importances is None and os.path.exists(importance_path):
        fi = pd.read_csv(importance_path)
    elif feature_importances is not None:
        fi = pd.DataFrame({
            "feature": X.columns,
            "importance": feature_importances
        }).sort_values("importance", ascending=False)
    else:
        raise RuntimeError("feature importances unavailable (no base estimator importances and no file)")
    fi.to_csv(REPORTS_DIR + "feature_importance.csv", index=False)
    print("Saved: feature_importance.csv")

    grouped_n = fi.head(30).copy()
    category_keywords = {
        "Team routing bottlenecks": ["Assignee Name", "Approval", "Routing", "Team"],
        "Slow response-time indicators": ["ToFirstResponse", "ToFirstResponse_min", "ToAssignment", "Service Time To Assignment"],
        "Reassignment / handoff friction": ["Reassign", "Assignment History"],
        "Category complexity": ["Category_", "Subcategory_"],
    }

    def categorize(feat: str) -> str:
        for label, keywords in category_keywords.items():
            for kw in keywords:
                if kw in feat:
                    return label
        return "Other"

    grouped_n["category"] = grouped_n["feature"].apply(categorize)
    palette_map = {
        "Team routing bottlenecks": "#1f77b4",
        "Slow response-time indicators": "#2ca02c",
        "Reassignment / handoff friction": "#ff7f0e",
        "Category complexity": "#9467bd",
        "Other": "#7f7f7f"
    }

    grouped_n = grouped_n.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = grouped_n["category"].map(palette_map)
    ax.barh(grouped_n["feature"], grouped_n["importance"], color=colors)
    ax.set_title("Feature Importances by Operational Category", fontsize=16)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("")
    handles = [
        plt.Line2D([0], [0], color=color, marker="s", linestyle="", markersize=10)
        for color in palette_map.values()
    ]
    ax.legend(handles, palette_map.keys(), title="Feature Group", loc="lower right")

    team_top5 = grouped_n[grouped_n["category"] == "Team routing bottlenecks"].nlargest(5, "importance")
    for _, row in team_top5.iterrows():
        ax.text(row["importance"] + 0.001, row["feature"], "Top bottleneck", va="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIG_DIR + "feature_importance_top15.png", dpi=300)
    plt.close()
    print("Saved: feature_importance_top15.png")

    out = pd.DataFrame({
        "Actual": y,
        "Predicted": y_pred,
        "RiskScore": y_proba
    })
    out.to_csv(REPORTS_DIR + "predictions_vs_actual.csv", index=False)
    print("Saved: predictions_vs_actual.csv")

    shap_sample = X.sample(n=min(len(X), 1000), random_state=42)
    base_for_shap = base_estimator or getattr(
        model,
        "base_estimator_",
        getattr(model, "base_estimator", model)
    )
    try:
        explainer = shap.TreeExplainer(base_for_shap)
        shap_vals = explainer.shap_values(shap_sample)
        shap_target = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_target, shap_sample, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "shap_summary.png"), dpi=300)
        plt.close()
        print("Saved: shap_summary.png")
    except Exception as exc:
        print("SHAP summary skipped:", exc)


if __name__ == "__main__":
    main()
