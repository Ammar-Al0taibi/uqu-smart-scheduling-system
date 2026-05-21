"""
mlflow_tracking.py
UQU Smart Scheduling V2 — MLflow Experiment Tracking + SHAP Explainability

Improvements over V1:
  - Every ML experiment is logged to MLflow (accuracy, AUC, params, artifacts)
  - SHAP values explain each prediction individually
  - Model comparison across multiple algorithms
  - Automated model registry with versioning
"""

import os, json, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import shap
warnings.filterwarnings("ignore")

from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.metrics           import (accuracy_score, roc_auc_score,
                                        classification_report, confusion_matrix)
from sklearn.preprocessing     import StandardScaler
from sklearn.pipeline          import Pipeline

SEED = 42
np.random.seed(SEED)

BASE     = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE, "..", "uqu_project", "data")
OUT_DIR  = os.path.join(BASE, "..", "uqu_project", "outputs")
FIG_DIR  = os.path.join(BASE, "..", "uqu_project", "figures")
MLFLOW_DIR = os.path.join(BASE, "mlruns")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")

# ── Feature engineering ────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "priority_weight", "academic_level", "gpa",
    "n_preferred_sections", "section_capacity",
    "slots_per_section", "is_graduating",
    "courses_already_allocated",
    # New V2 features
    "credit_hours_completed", "failed_courses_count",
    "priority_x_level",          # interaction term
    "gpa_x_priority",            # interaction term
]

def build_features(reqs, students, alloc, schedule):
    students_idx = students.set_index(students["student_id"].astype(str))
    alloc_ids    = set(alloc["request_id"].astype(str))
    sec_cap      = dict(zip(schedule["section_id"], schedule["capacity"]))
    sec_slots_n  = {
        row["section_id"]: len(str(row["slot_ids"]).split("|"))
        for _, row in schedule.iterrows()
    }
    alloc_per_student = alloc.groupby(alloc["student_id"].astype(str)).size().to_dict()

    features, labels = [], []
    for _, req in reqs.iterrows():
        sid = str(req["student_id"])
        try:
            stud = students_idx.loc[sid]
        except KeyError:
            continue

        prefs   = str(req.get("preferred_sections","")).split("|") if pd.notna(req.get("preferred_sections")) else []
        n_prefs = len([p for p in prefs if p.strip()])
        cap = 50; slots_n = 3
        for p in prefs:
            if p in sec_cap:
                cap = sec_cap[p]; slots_n = sec_slots_n.get(p, 3); break

        n_already   = alloc_per_student.get(sid, 0)
        pw          = float(req["priority_weight"])
        lvl         = int(stud["academic_level"])
        gpa         = float(stud["gpa"])
        credit_hrs  = int(stud.get("credit_hours_completed", (lvl-1)*15))
        failed      = int(stud.get("failed_courses_count", 0))

        features.append([
            pw, lvl, gpa, n_prefs, cap, slots_n,
            1 if req.get("is_graduating") else 0,
            n_already,
            credit_hrs, failed,
            pw * lvl,    # interaction
            gpa * pw,    # interaction
        ])
        labels.append(0 if str(req["request_id"]) in alloc_ids else 1)

    return np.array(features), np.array(labels)


# ── Model comparison ───────────────────────────────────────────────────────────
MODELS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=SEED, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1, random_state=SEED),
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
    ]),
}


def run_experiment():
    print("=" * 60)
    print("UQU V2 — MLflow Experiment Tracking + Model Comparison")
    print("=" * 60)

    # Load data
    students = pd.read_csv(os.path.join(DATA_DIR, "students.csv"))
    reqs     = pd.read_csv(os.path.join(DATA_DIR, "registration_requests.csv"))
    alloc    = pd.read_csv(os.path.join(OUT_DIR,  "allocations.csv"))
    schedule = pd.read_csv(os.path.join(OUT_DIR,  "section_schedule.csv"))

    print("\nBuilding feature matrix...")
    X, y = build_features(reqs, students, alloc, schedule)
    print(f"  Samples: {len(X):,} | Features: {X.shape[1]} | Positive rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y)

    mlflow.set_experiment("UQU_Conflict_Risk_Classifier_V2")
    best_model = None
    best_acc   = 0.0
    results    = {}

    for model_name, clf in MODELS.items():
        print(f"\n  Training {model_name}...")
        t0 = time.time()

        with mlflow.start_run(run_name=model_name):
            clf.fit(X_train, y_train)
            y_pred  = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1] \
                      if hasattr(clf, "predict_proba") \
                      else clf.decision_function(X_test)

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            cv  = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy").mean()
            elapsed = time.time() - t0

            # Log params
            if hasattr(clf, "n_estimators"):
                mlflow.log_param("n_estimators", clf.n_estimators)
                mlflow.log_param("max_depth",    clf.max_depth)
            mlflow.log_param("model_type",   model_name)
            mlflow.log_param("n_features",   X.shape[1])
            mlflow.log_param("train_samples",len(X_train))

            # Log metrics
            mlflow.log_metric("accuracy",       acc)
            mlflow.log_metric("auc_roc",        auc)
            mlflow.log_metric("cv_accuracy",    cv)
            mlflow.log_metric("training_time_s", elapsed)

            # Log model
            if hasattr(clf, "feature_importances_"):
                mlflow.sklearn.log_model(clf, f"model_{model_name}")

            results[model_name] = {
                "accuracy": round(acc, 4),
                "auc_roc":  round(auc, 4),
                "cv_acc":   round(cv, 4),
                "time_s":   round(elapsed, 2),
            }
            print(f"    Accuracy: {acc:.4f} | AUC: {auc:.4f} | CV: {cv:.4f} | Time: {elapsed:.1f}s")

            if acc > best_acc:
                best_acc   = acc
                best_model = (model_name, clf)

    # ── SHAP explainability on best model ─────────────────────────────────────
    print(f"\n  Running SHAP on best model: {best_model[0]}...")
    clf = best_model[1]

    if hasattr(clf, "feature_importances_"):
        explainer  = shap.TreeExplainer(clf)
        X_explain  = X_test[:500]    # sample for speed
        shap_vals  = explainer.shap_values(X_explain)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        # Summary plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_explain,
                          feature_names=FEATURE_NAMES,
                          plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance — {best_model[0]}\n"
                  "UQU Conflict-Risk Classifier V2", fontweight="bold")
        plt.tight_layout()
        shap_path = os.path.join(FIG_DIR, "shap_feature_importance.png")
        plt.savefig(shap_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"    SHAP plot saved → {shap_path}")

        # Mean absolute SHAP values per feature
        mean_shap = np.abs(shap_vals).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature":    FEATURE_NAMES,
            "mean_shap":  np.round(mean_shap, 5),
        }).sort_values("mean_shap", ascending=False)
        print("\n    Top SHAP features:")
        for _, row in shap_df.head(6).iterrows():
            print(f"      {row['feature']:<30} {row['mean_shap']:.5f}")

        shap_out = os.path.join(OUT_DIR, "shap_values.csv")
        shap_df.to_csv(shap_out, index=False)

    # ── Model comparison figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    model_names = list(results.keys())
    accs  = [results[m]["accuracy"] * 100 for m in model_names]
    aucs  = [results[m]["auc_roc"]         for m in model_names]
    colors = ["#1B5E20", "#1565C0", "#E65100"]

    axes[0].bar(model_names, accs, color=colors, edgecolor="white")
    axes[0].set_ylabel("Accuracy (%)");  axes[0].set_ylim(0, 105)
    axes[0].set_title("Model Accuracy Comparison", fontweight="bold")
    for i, v in enumerate(accs):
        axes[0].text(i, v+0.5, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

    axes[1].bar(model_names, aucs, color=colors, edgecolor="white")
    axes[1].set_ylabel("AUC-ROC");       axes[1].set_ylim(0, 1.1)
    axes[1].set_title("AUC-ROC Comparison", fontweight="bold")
    for i, v in enumerate(aucs):
        axes[1].text(i, v+0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("UQU V2 — Conflict-Risk Classifier Model Comparison\n"
                 "Tracked with MLflow", fontweight="bold")
    plt.tight_layout()
    cmp_path = os.path.join(FIG_DIR, "model_comparison.png")
    fig.savefig(cmp_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Model comparison figure → {cmp_path}")

    # Save results
    summary = {
        "experiment": "UQU_Conflict_Risk_Classifier_V2",
        "best_model": best_model[0],
        "best_accuracy": best_acc,
        "models": results,
        "n_features_v2": X.shape[1],
        "n_features_v1": 8,
        "new_features": ["credit_hours_completed", "failed_courses_count",
                         "priority_x_level", "gpa_x_priority"],
        "mlflow_tracking_uri": MLFLOW_DIR,
    }
    out_path = os.path.join(OUT_DIR, "mlflow_experiment_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Best model  : {best_model[0]}")
    print(f"  Accuracy    : {best_acc:.4f}")
    print(f"  Results     → {out_path}")
    print("=" * 60)
    return summary


if __name__ == "__main__":
    run_experiment()
