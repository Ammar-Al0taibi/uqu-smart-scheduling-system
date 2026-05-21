"""
05_ml_module.py
UQU Smart Scheduling System – Machine Learning Module
Project: UQU-DS-2025-M09

Three ML sub-systems:
  A. Collaborative-Filtering Course Recommender
  B. Demand Forecaster (level-adjusted growth model)
  C. Random Forest Conflict-Risk Classifier (target accuracy 94.7%)

Outputs (saved to outputs/):
  sample_recommendations.json
  course_demand_forecast.csv
  conflict_risk_predictions.csv
  conflict_prediction_features.png  (saved to figures/)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.metrics           import (classification_report, confusion_matrix,
                                        roc_auc_score, accuracy_score)
from sklearn.preprocessing     import LabelEncoder

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR  = os.path.join(BASE, "outputs")
FIG_DIR  = os.path.join(BASE, "figures")
SEED     = 42
np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ── Loaders ───────────────────────────────────────────────────────────────────
def load():
    students  = pd.read_csv(os.path.join(DATA_DIR, "students.csv"))
    courses   = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))
    sections  = pd.read_csv(os.path.join(DATA_DIR, "sections.csv"))
    reqs      = pd.read_csv(os.path.join(DATA_DIR, "registration_requests.csv"))
    alloc     = pd.read_csv(os.path.join(OUT_DIR,  "allocations.csv"))
    schedule  = pd.read_csv(os.path.join(OUT_DIR,  "section_schedule.csv"))
    return students, courses, sections, reqs, alloc, schedule


# ═══════════════════════════════════════════════════════════════════════════════
# A. Collaborative-Filtering Recommender
# ═══════════════════════════════════════════════════════════════════════════════

def build_recommender(alloc, students, courses):
    """
    Item-based CF using cosine similarity on a student × course binary matrix.
    Returns top-5 recommendations for a sample of 10 students.
    """
    print("  [A] Building collaborative-filtering recommender...")

    # Build binary matrix (students × courses)
    alloc["student_id"] = alloc["student_id"].astype(str)
    pivot = (
        alloc.groupby(["student_id", "course_id"])
        .size()
        .unstack(fill_value=0)
        .clip(upper=1)
    )

    if pivot.empty:
        print("    ⚠ Not enough allocation data for recommender.")
        return {}

    # Item-item cosine similarity
    course_matrix = pivot.values.T.astype(float)  # shape: (n_courses, n_students)
    norms = np.linalg.norm(course_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    course_norm = course_matrix / norms
    sim_matrix = course_norm @ course_norm.T   # (n_courses, n_courses)

    course_ids  = list(pivot.columns)
    course_name = dict(zip(courses["course_id"], courses["course_name"]))
    course_code = dict(zip(courses["course_id"], courses["course_code"]))

    def recommend(student_id, top_n=5):
        if student_id not in pivot.index:
            return []
        student_vec = pivot.loc[student_id].values.astype(float)
        scores = sim_matrix @ student_vec
        # Zero out already-enrolled courses
        scores[student_vec > 0] = -1
        top_indices = np.argsort(scores)[::-1][:top_n]
        recs = []
        for idx in top_indices:
            cid = course_ids[idx]
            recs.append({
                "course_id":   cid,
                "course_code": course_code.get(cid, cid),
                "course_name": course_name.get(cid, ""),
                "score":       round(float(scores[idx]), 4),
            })
        return recs

    # Sample 10 students
    sample_students = list(pivot.index[:10])
    student_name_map = dict(zip(students["student_id"].astype(str), students["name"]))
    results = {}
    for sid in sample_students:
        results[sid] = {
            "student_name": student_name_map.get(sid, sid),
            "recommendations": recommend(sid),
        }

    path = os.path.join(OUT_DIR, "sample_recommendations.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"    ✓ Saved → {path}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# B. Demand Forecaster
# ═══════════════════════════════════════════════════════════════════════════════

def build_forecaster(reqs, courses, students):
    """
    Historical demand = number of requests per course.
    Apply level-adjusted growth model:
      projected_demand = current_demand × (1 + growth_rate)
    Growth rates are differentiated by course level and program relevance.
    """
    print("  [B] Building demand forecaster...")

    demand = reqs.groupby("course_id").size().reset_index(name="current_requests")
    course_info = courses[["course_id", "course_code", "course_name", "level", "category"]].copy()
    demand = demand.merge(course_info, on="course_id", how="left")

    level_growth = {1: 0.05, 2: 0.04, 3: 0.06, 4: 0.07, 5: 0.08, 6: 0.09, 7: 0.10, 8: 0.12}
    cat_adjustment = {"Core": 1.0, "University": 0.95, "Elective": 1.05, "Lab": 0.9}

    def forecast_row(row):
        base_growth = level_growth.get(row["level"], 0.06)
        cat_adj     = cat_adjustment.get(row["category"], 1.0)
        growth_rate = base_growth * cat_adj
        forecast_1  = int(row["current_requests"] * (1 + growth_rate))
        forecast_2  = int(row["current_requests"] * (1 + growth_rate) ** 2)
        sections_needed = max(1, forecast_2 // 40)
        return pd.Series({
            "growth_rate_pct":     round(growth_rate * 100, 2),
            "forecast_next_sem":   forecast_1,
            "forecast_2_sem":      forecast_2,
            "recommended_sections": sections_needed,
        })

    demand[["growth_rate_pct", "forecast_next_sem", "forecast_2_sem", "recommended_sections"]] = \
        demand.apply(forecast_row, axis=1)

    demand = demand.sort_values("forecast_2_sem", ascending=False)
    path = os.path.join(OUT_DIR, "course_demand_forecast.csv")
    demand.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"    ✓ Saved → {path}  ({len(demand):,} courses)")

    # Figure: top 15 forecasted demand
    top15 = demand.head(15)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(top15))
    width = 0.35
    ax.bar(x - width/2, top15["current_requests"], width, label="Current Semester",
           color="#1B5E20", edgecolor="white")
    ax.bar(x + width/2, top15["forecast_2_sem"], width, label="2-Semester Forecast",
           color="#42A5F5", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(top15["course_code"], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Registration Requests")
    ax.set_title("KPI – Course Demand Forecast (Top 15)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fpath = os.path.join(FIG_DIR, "forecast_demand.png")
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Figure → {fpath}")

    return demand


# ═══════════════════════════════════════════════════════════════════════════════
# C. Random Forest Conflict-Risk Classifier
# ═══════════════════════════════════════════════════════════════════════════════

def build_conflict_classifier(reqs, students, alloc, schedule):
    """
    Features per request:
      - student priority_weight
      - student academic_level
      - student gpa (via students table)
      - n_preferred_sections
      - section_capacity (of most-preferred section)
      - scheduled_slots_count
      - is_graduating
      - n_courses_already_allocated (for this student up to this request)

    Label: 1 = request was eventually waitlisted (conflict/full), 0 = allocated

    Train Random Forest → report accuracy + feature importance figure.
    """
    print("  [C] Building conflict-risk classifier...")

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

        prefs = str(req["preferred_sections"]).split("|") if pd.notna(req["preferred_sections"]) else []
        n_prefs = len([p for p in prefs if p.strip()])

        # Capacity of first preferred section that is scheduled
        cap = 50  # default
        slots_n = 3
        for p in prefs:
            if p in sec_cap:
                cap = sec_cap[p]
                slots_n = sec_slots_n.get(p, 3)
                break

        n_already = alloc_per_student.get(sid, 0)

        features.append([
            float(req["priority_weight"]),
            int(stud["academic_level"]),
            float(stud["gpa"]),
            n_prefs,
            cap,
            slots_n,
            1 if req["is_graduating"] else 0,
            n_already,
        ])
        labels.append(0 if str(req["request_id"]) in alloc_ids else 1)

    X = np.array(features)
    y = np.array(labels)

    if len(X) < 100:
        print("    ⚠ Not enough data for classifier.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    print(f"    Accuracy : {acc:.4f}  |  AUC-ROC: {auc:.4f}")
    print("    " + classification_report(y_test, y_pred, target_names=["Allocated", "Waitlisted"],
                                         zero_division=0).replace("\n", "\n    "))

    # Save predictions on all requests
    preds_proba = clf.predict_proba(X)[:, 1]
    preds_class = clf.predict(X)
    risk_df = reqs[["request_id", "student_id", "course_id", "priority_weight"]].copy().reset_index(drop=True)
    risk_df["conflict_risk_score"]   = np.round(preds_proba, 4)
    risk_df["predicted_waitlisted"]  = preds_class
    risk_df["actual_waitlisted"]     = [1 - (1 if str(r) in alloc_ids else 0)
                                         for r in risk_df["request_id"].astype(str)]

    path = os.path.join(OUT_DIR, "conflict_risk_predictions.csv")
    risk_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"    ✓ Predictions saved → {path}")

    # Feature importance figure
    feat_names = [
        "Priority Weight", "Academic Level", "GPA",
        "# Preferred Sections", "Section Capacity",
        "Slots per Section", "Is Graduating", "Courses Already Allocated",
    ]
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(feat_names)), importances[order],
           color=["#1B5E20" if i == order[0] else "#42A5F5" for i in range(len(feat_names))],
           edgecolor="white")
    ax.set_xticks(range(len(feat_names)))
    ax.set_xticklabels([feat_names[i] for i in order], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Feature Importance (Gini)")
    ax.set_title("Conflict-Risk Classifier – Feature Importance\n"
                 f"Random Forest · Accuracy {acc:.3f} · AUC {auc:.3f}",
                 fontweight="bold")
    plt.tight_layout()
    fpath = os.path.join(FIG_DIR, "conflict_prediction_features.png")
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Figure → {fpath}")

    return clf, acc, auc


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("UQU Smart Scheduling – ML Module")
    print("=" * 60)

    print("\nLoading data...")
    students, courses, sections, reqs, alloc, schedule = load()
    print(f"  Loaded: {len(students):,} students | {len(reqs):,} requests | {len(alloc):,} allocations")

    print("\n── A. Collaborative-Filtering Recommender ──────────────────")
    build_recommender(alloc, students, courses)

    print("\n── B. Demand Forecaster ────────────────────────────────────")
    build_forecaster(reqs, courses, students)

    print("\n── C. Conflict-Risk Classifier (Random Forest) ─────────────")
    result = build_conflict_classifier(reqs, students, alloc, schedule)

    print("\n" + "=" * 60)
    print("ML Module complete.")
    if result:
        _, acc, auc = result
        print(f"  Classifier Accuracy : {acc:.4f}")
        print(f"  Classifier AUC-ROC  : {auc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
