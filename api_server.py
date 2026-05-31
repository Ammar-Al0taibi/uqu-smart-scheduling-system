"""
api_server.py
UQU Smart Scheduling V2 — FastAPI REST API Backend
"""

import os
import json
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE, "..", "..", "uqu_project", "scripts", "data"))
OUT_DIR  = os.path.abspath(os.path.join(BASE, "..", "..", "uqu_project", "scripts", "outputs"))

# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="UQU Smart Scheduling API",
    description="Intelligent student schedule management system — Umm Al-Qura University",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Cache ─────────────────────────────────────────────────────────
_cache: dict = {}

def load_data(name: str) -> pd.DataFrame:
    """Load a CSV file from the data directory with caching."""
    if name not in _cache:
        path = os.path.join(DATA_DIR, f"{name}.csv")
        if not os.path.exists(path):
            raise HTTPException(status_code=503, detail=f"File not found: {name}.csv")
        _cache[name] = pd.read_csv(path)
    return _cache[name]

def load_output(name: str) -> pd.DataFrame:
    """Load a CSV file from the outputs directory with caching."""
    key = f"out_{name}"
    if key not in _cache:
        path = os.path.join(OUT_DIR, f"{name}.csv")
        if not os.path.exists(path):
            raise HTTPException(status_code=503, detail=f"File not found: {name}.csv")
        _cache[key] = pd.read_csv(path)
    return _cache[key]

def load_kpi() -> dict:
    """Load the KPI report from a JSON file."""
    path = os.path.join(OUT_DIR, "kpi_report.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=503, detail="File not found: kpi_report.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ── Pydantic Models ───────────────────────────────────────────────
class ConflictRiskRequest(BaseModel):
    student_id: str
    course_id: str
    priority_weight: float
    academic_level: int
    gpa: float
    n_preferred_sections: int = 3
    is_graduating: bool = False

# ── Helpers ───────────────────────────────────────────────────────
LEVEL_COURSE_MAP = {
    1: ["MATH101", "CS101", "ENGL101", "PHYS101"],
    2: ["MATH201", "CS201", "DS201", "PHYS201"],
    3: ["CS301", "AI301", "DB301", "NET301"],
    4: ["CS401", "AI401", "SEC401", "PROJ401"],
}

def get_recommended_courses(student_id: str, level: int, gpa: float) -> dict:
    """
    Simple course recommendation logic based on academic level and GPA.
    Can be replaced later with a real ML model.
    """
    suitable    = []
    recommended = []

    level_courses = LEVEL_COURSE_MAP.get(level, [])
    next_level    = LEVEL_COURSE_MAP.get(level + 1, [])

    # Current level courses — always suitable
    for course in level_courses:
        suitable.append({
            "course_id": course,
            "reason":    f"Course suitable for your academic level ({level})",
            "priority":  "high"
        })

    # If GPA is high, recommend a course from the next level
    if gpa >= 3.5 and next_level:
        recommended.append({
            "course_id": next_level[0],
            "reason":    f"Your GPA of {gpa:.2f} qualifies you for an advanced course",
            "priority":  "medium"
        })

    # If GPA is low, recommend a support course
    if gpa < 2.5 and level_courses:
        recommended.append({
            "course_id": level_courses[0],
            "reason":    "We recommend focusing on this course to improve your GPA",
            "priority":  "high"
        })

    return {
        "student_id":          student_id,
        "academic_level":      level,
        "gpa":                 gpa,
        "suitable_courses":    suitable,
        "recommended_courses": recommended,
        "total_suggestions":   len(suitable) + len(recommended),
        "generated_at":        datetime.now().isoformat(),
    }

# ════════════════════════════════════════════════════════════════
#  Endpoints
# ════════════════════════════════════════════════════════════════

# ── Health ───────────────────────────────────────────────────────
@app.get("/health", tags=["General"], summary="Server health check")
def health():
    """Confirms the server is running."""
    return {"status": "ok", "time": datetime.now().isoformat()}


# ── Stats ─────────────────────────────────────────────────────────
@app.get("/api/v2/stats", tags=["General"], summary="System statistics")
def stats():
    """Number of students, courses, sections, and more."""
    students   = load_data("students")
    courses    = load_data("courses")
    sections   = load_data("sections")
    classrooms = load_data("classrooms")
    requests   = load_data("registration_requests")

    return {
        "students":   int(len(students)),
        "courses":    int(len(courses)),
        "sections":   int(len(sections)),
        "classrooms": int(len(classrooms)),
        "requests":   int(len(requests)),
        "programs":   list(students["program"].unique()),
    }


# ── KPIs ──────────────────────────────────────────────────────────
@app.get("/api/v2/kpis", tags=["General"], summary="Key Performance Indicators")
def kpis():
    """Main KPI report for the system."""
    return load_kpi()


# ── Students ──────────────────────────────────────────────────────
@app.get("/api/v2/students/{student_id}/schedule", tags=["Students"], summary="Student schedule")
def student_schedule(student_id: str):
    """
    Schedule for a specific student.

    - **student_id**: Student ID number
    """
    df     = load_output("student_schedules")
    result = df[df["student_id"].astype(str) == student_id]

    if result.empty:
        return {"student_id": student_id, "schedule": [], "message": "No schedule found for this student"}

    return {"student_id": student_id, "schedule": result.to_dict("records")}


@app.get("/api/v2/students/{student_id}/recommendations", tags=["Students"], summary="Smart recommendations for student")
def student_recommendations(
    student_id: str,
    academic_level: int = Query(..., ge=1, le=8, description="Academic level (1-8)"),
    gpa: float          = Query(..., ge=0.0, le=4.0, description="Cumulative GPA (0.0 - 4.0)"),
):
    """
    Smart recommendations for a student based on their level and GPA.

    - **student_id**: Student ID number
    - **academic_level**: Academic level from 1 to 8
    - **gpa**: Cumulative GPA from 0.0 to 4.0

    Returns:
    - **suitable_courses**: Courses appropriate for the current level
    - **recommended_courses**: Courses recommended based on GPA
    """
    return get_recommended_courses(student_id, academic_level, gpa)


# ── Courses ───────────────────────────────────────────────────────
@app.get("/api/v2/courses/demand-forecast", tags=["Courses"], summary="Course demand forecast")
def demand_forecast(top_n: int = Query(10, ge=1, le=50, description="Number of courses to display")):
    """Most in-demand courses with enrollment forecasts."""
    df = load_output("course_demand_forecast")
    return df.head(top_n).to_dict("records")


# ── Sections ──────────────────────────────────────────────────────
@app.get("/api/v2/sections", tags=["Sections"], summary="Section schedule")
def sections(
    page: int      = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Results per page"),
):
    """
    List of course sections with pagination support.

    - **page**: Page number
    - **page_size**: Number of results per page (max 200)
    """
    df    = load_output("section_schedule")
    total = len(df)
    start = (page - 1) * page_size
    end   = start + page_size

    return {
        "total":     total,
        "page":      page,
        "page_size": page_size,
        "pages":     (total + page_size - 1) // page_size,
        "data":      df.iloc[start:end].to_dict("records"),
    }


# ── Classrooms ────────────────────────────────────────────────────
@app.get("/api/v2/classrooms/utilization", tags=["Classrooms"], summary="Classroom utilization rate")
def classrooms_utilization():
    """Summary of classroom utilization."""
    df    = load_output("section_schedule")
    total = len(df)

    return {
        "total_sections": total,
        "message":        "Detailed data coming in the next update",
    }


# ── Allocations ───────────────────────────────────────────────────
@app.get("/api/v2/allocations/summary", tags=["Allocations"], summary="Allocation summary")
def allocations_summary():
    """Summary of student-to-section allocations."""
    df = load_output("allocations")
    return {
        "total_allocated": int(len(df)),
        "message":         "Detailed data coming in the next update",
    }


# ── Conflict Risk Prediction ───────────────────────────────────────
@app.post("/api/v2/predict/conflict-risk", tags=["Prediction"], summary="Schedule conflict probability")
def conflict_risk(req: ConflictRiskRequest):
    """
    Estimates the probability of a scheduling conflict for a student registering in a specific course.

    Note: Calculation is currently approximate. Will be replaced by an ML model in a future release.
    """
    score = round((5 - req.priority_weight) * 0.2, 3)

    return {
        "student_id": req.student_id,
        "course_id":  req.course_id,
        "risk_score": score,
        "risk_level": "high" if score > 0.7 else "medium" if score > 0.4 else "low",
        "note":       "Approximate calculation — ML model under development",
    }


# ── Run Server ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\nStarting server at http://127.0.0.1:8000")
    print("Interactive docs: http://127.0.0.1:8000/docs\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
