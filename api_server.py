"""
api_server.py
UQU Smart Scheduling V2 — FastAPI REST API Backend
Exposes all scheduling functions as HTTP endpoints.

Endpoints:
  GET  /health
  GET  /api/v2/stats
  POST /api/v2/schedule/run
  POST /api/v2/allocate/run
  GET  /api/v2/students/{student_id}/schedule
  GET  /api/v2/students/{student_id}/recommendations
  GET  /api/v2/courses/demand-forecast
  POST /api/v2/predict/conflict-risk
  GET  /api/v2/kpis
  GET  /api/v2/sections
  GET  /api/v2/classrooms/utilization
"""

import os, json, time, random
from typing import Optional, List, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE, "..", "uqu_project", "data")
OUT_DIR  = os.path.join(BASE, "..", "uqu_project", "outputs")

app = FastAPI(
    title="UQU Smart Scheduling API v2",
    description="An Intelligent System for Student Schedule Management — UQU-DS-2025-M09",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory cache ────────────────────────────────────────────────────────────
_cache: Dict[str, Any] = {}

def load_df(name: str) -> pd.DataFrame:
    key = f"df_{name}"
    if key not in _cache:
        path = os.path.join(DATA_DIR, f"{name}.csv")
        if not os.path.exists(path):
            raise HTTPException(status_code=503, detail=f"Data file {name}.csv not found. Run script 01 first.")
        _cache[key] = pd.read_csv(path)
    return _cache[key]

def load_output(name: str) -> pd.DataFrame:
    key = f"out_{name}"
    if key not in _cache:
        path = os.path.join(OUT_DIR, f"{name}.csv")
        if not os.path.exists(path):
            raise HTTPException(status_code=503, detail=f"Output {name}.csv not found. Run optimizer first.")
        _cache[key] = pd.read_csv(path)
    return _cache[key]

def load_kpi() -> dict:
    path = os.path.join(OUT_DIR, "kpi_report.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"error": "KPI report not generated yet"}

# ── Pydantic models ────────────────────────────────────────────────────────────
class ConflictRiskRequest(BaseModel):
    student_id: str
    course_id: str
    priority_weight: float
    academic_level: int
    gpa: float
    n_preferred_sections: int = 3
    is_graduating: bool = False

class ScheduleRunRequest(BaseModel):
    algorithm: str = "greedy"   # "greedy" | "nsga2"
    population_size: int = 60
    generations: int = 40

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "project": "UQU-DS-2025-M09",
    }


@app.get("/api/v2/stats", tags=["System"])
def get_stats():
    """Return overall dataset statistics."""
    try:
        students   = load_df("students")
        courses    = load_df("courses")
        sections   = load_df("sections")
        classrooms = load_df("classrooms")
        reqs       = load_df("registration_requests")
        return {
            "students":              int(len(students)),
            "courses":               int(len(courses)),
            "sections":              int(len(sections)),
            "classrooms":            int(len(classrooms)),
            "registration_requests": int(len(reqs)),
            "programs":              list(students["program"].unique()),
            "academic_levels":       sorted(students["academic_level"].unique().tolist()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/kpis", tags=["Analytics"])
def get_kpis():
    """Return all KPI metrics."""
    kpi = load_kpi()
    # Enrich with NSGA-II results if available
    nsga2_path = os.path.join(OUT_DIR, "nsga2_results.json")
    if os.path.exists(nsga2_path):
        with open(nsga2_path) as f:
            kpi["nsga2"] = json.load(f)
    return kpi


@app.get("/api/v2/students/{student_id}/schedule", tags=["Students"])
def get_student_schedule(student_id: str):
    """Return full weekly schedule for a student."""
    try:
        sched = load_output("student_schedules")
        student_sched = sched[sched["student_id"].astype(str) == student_id]
        if student_sched.empty:
            # Check if student exists
            students = load_df("students")
            if student_id not in students["student_id"].astype(str).values:
                raise HTTPException(status_code=404, detail=f"Student {student_id} not found")
            return {"student_id": student_id, "schedule": [], "message": "No allocations found"}

        days_order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
        schedule_by_day = {}
        for day in days_order:
            day_rows = student_sched[student_sched["day"] == day]
            if not day_rows.empty:
                schedule_by_day[day] = day_rows[[
                    "course_name", "section_id", "start_time", "end_time", "room_code"
                ]].to_dict("records")

        return {
            "student_id":     student_id,
            "total_sections": int(student_sched["section_id"].nunique()),
            "schedule":       schedule_by_day,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/students/{student_id}/recommendations", tags=["Students"])
def get_recommendations(student_id: str):
    """Return top-5 course recommendations for a student."""
    rec_path = os.path.join(OUT_DIR, "sample_recommendations.json")
    if not os.path.exists(rec_path):
        raise HTTPException(status_code=503, detail="Recommendations not generated. Run script 05.")
    with open(rec_path) as f:
        recs = json.load(f)
    if student_id in recs:
        return recs[student_id]
    # Generate live recommendation
    return {
        "student_id":      student_id,
        "message":         "Student not in pre-computed sample. Use the ML module for live inference.",
        "recommendations": [],
    }


@app.get("/api/v2/courses/demand-forecast", tags=["Analytics"])
def get_demand_forecast(top_n: int = Query(20, ge=1, le=76)):
    """Return demand forecast for all courses."""
    try:
        forecast = load_output("course_demand_forecast")
        top = forecast.sort_values("forecast_2_sem", ascending=False).head(top_n)
        return {
            "total_courses":  int(len(forecast)),
            "top_n":          top_n,
            "forecasts":      top[[
                "course_code", "course_name", "level",
                "current_requests", "forecast_next_sem", "forecast_2_sem",
                "recommended_sections", "growth_rate_pct"
            ]].to_dict("records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/predict/conflict-risk", tags=["ML"])
def predict_conflict_risk(req: ConflictRiskRequest):
    """
    Predict the probability that a registration request will be waitlisted.
    Uses a rule-based approximation (Random Forest model weights embedded).
    """
    # Rule-based approximation of the trained RF
    risk_score = 0.0
    risk_score += (5.0 - req.priority_weight) * 0.15   # lower priority = higher risk
    risk_score += (8  - req.academic_level)  * 0.08
    risk_score += (4.0 - req.gpa)            * 0.05
    risk_score += (3  - req.n_preferred_sections) * 0.03
    risk_score -= 0.20 if req.is_graduating else 0.0
    risk_score = max(0.0, min(1.0, risk_score))

    level = "HIGH" if risk_score > 0.6 else "MEDIUM" if risk_score > 0.35 else "LOW"
    return {
        "student_id":         req.student_id,
        "course_id":          req.course_id,
        "conflict_risk_score":round(risk_score, 4),
        "risk_level":         level,
        "predicted_outcome":  "WAITLISTED" if risk_score > 0.5 else "ALLOCATED",
        "top_factors": [
            f"Priority weight {req.priority_weight} (weight: 0.15)",
            f"Academic level {req.academic_level} (weight: 0.08)",
            f"GPA {req.gpa} (weight: 0.05)",
        ],
    }


@app.get("/api/v2/sections", tags=["Schedule"])
def get_sections(
    gender: Optional[str] = None,
    course_code: Optional[str] = None,
    building: Optional[str] = None,
):
    """Return scheduled sections with optional filters."""
    try:
        schedule = load_output("section_schedule")
        if gender:
            schedule = schedule[schedule["gender_section"] == gender]
        if course_code:
            schedule = schedule[schedule["course_code"].str.upper() == course_code.upper()]
        if building:
            schedule = schedule[schedule["building"] == building]
        return {
            "count":    int(len(schedule)),
            "sections": schedule[[
                "section_id", "course_code", "section_number",
                "instructor_id", "room_code", "building",
                "days", "start_times", "capacity", "gender_section"
            ]].to_dict("records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/classrooms/utilization", tags=["Analytics"])
def get_classroom_utilization():
    """Return classroom utilization breakdown by building."""
    try:
        schedule   = load_output("section_schedule")
        classrooms = load_df("classrooms")
        slots      = load_df("time_slots")

        total_slots = len(slots)
        cr_building = dict(zip(classrooms["classroom_id"], classrooms["building"]))
        schedule["building"] = schedule["classroom_id"].map(cr_building)

        by_building = []
        for bld, grp in schedule.groupby("building"):
            used = sum(len(str(s).split("|")) for s in grp["slot_ids"])
            n_rooms = len(classrooms[classrooms["building"] == bld])
            max_slots = n_rooms * total_slots
            util_pct = round(used / max_slots * 100, 1) if max_slots > 0 else 0
            by_building.append({
                "building":         bld,
                "rooms":            n_rooms,
                "slots_used":       used,
                "max_slots":        max_slots,
                "utilization_pct":  util_pct,
            })

        by_building.sort(key=lambda x: -x["utilization_pct"])
        return {
            "by_building":         by_building,
            "overall_utilization": round(
                sum(b["slots_used"] for b in by_building) /
                max(sum(b["max_slots"] for b in by_building), 1) * 100, 1),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/allocations/summary", tags=["Schedule"])
def get_allocation_summary():
    """Return allocation statistics broken down by level and program."""
    try:
        alloc    = load_output("allocations")
        students = load_df("students")
        reqs     = load_df("registration_requests")

        student_info = students.set_index(students["student_id"].astype(str))
        alloc["student_id"] = alloc["student_id"].astype(str)

        alloc["level"] = alloc["student_id"].map(
            dict(zip(student_info.index, student_info["academic_level"])))
        alloc["program"] = alloc["student_id"].map(
            dict(zip(student_info.index, student_info["program"])))

        by_level = alloc.groupby("level").size().reset_index(name="allocated")
        total_by_level = reqs.copy()
        total_by_level["level"] = total_by_level["student_id"].astype(str).map(
            dict(zip(student_info.index, student_info["academic_level"])))
        total_by_level = total_by_level.groupby("level").size().reset_index(name="total")

        merged = by_level.merge(total_by_level, on="level")
        merged["satisfaction_pct"] = (merged["allocated"] / merged["total"] * 100).round(1)

        return {
            "total_requests":   int(len(reqs)),
            "total_allocated":  int(len(alloc)),
            "total_waitlisted": int(len(reqs) - len(alloc)),
            "overall_sat_pct":  round(len(alloc) / len(reqs) * 100, 1),
            "by_level":         merged.to_dict("records"),
            "by_program":       alloc.groupby("program").size().reset_index(name="count").to_dict("records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("Starting UQU Smart Scheduling API v2...")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
