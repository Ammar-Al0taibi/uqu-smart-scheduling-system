# UQU Smart Scheduling System

**Project ID:** UQU-DS-2025-M09
**Title:** An Intelligent System for Student Schedule Management and Priority-Based Class Allocation
**Institution:** Umm Al-Qura University · College of Computing · Data Science Department
**Supervisor:** Dr. Ahmed Bukhari

**Team:**
- Raed Falah Alhelali (444006561)
- Mohammed Saad Alsarhani (444006145)
- Ammar Eid Alotaibi (444005392)
- Moayad Yousef Hawsawi (444004193)

---

## 1. What This Project Does

This system solves three connected academic-operations problems for a Saudi university at realistic scale:

1. **Schedules course sections** to time slots and classrooms without instructor or room conflicts.
2. **Allocates students to sections** in priority order — graduating students are protected first, freshmen last.
3. **Provides predictive analytics** so departments can plan section counts before registration opens.

The work delivered runs end-to-end on a synthetic dataset of:

| Asset | Count |
|---|---|
| Students | **5,000** |
| Courses | **76** |
| Sections | **390** |
| Classrooms | **92** in 7 buildings |
| Instructors | **80** |
| Time slots | **45** per week (Sun–Thu × 9 slots) |
| Registration requests | **29,215** |

---

## 2. Headline Results

| KPI | Value |
|---|---|
| Graduating-student allocation satisfaction | **85.8%** |
| Non-graduating allocation satisfaction | 48.9% |
| Final time conflicts in any student schedule | **0** (verified) |
| Improvement vs FIFO baseline (graduating) | **+15.5 percentage points** |
| Conflict-risk classifier accuracy | **94.7%** |
| Average idle gap per student-day | 2.64 hours |
| End-to-end pipeline runtime | ~30 seconds |

The priority system works as designed: **Level 8 students reach 96.9% satisfaction, Level 1 students 42.5%** — exactly the monotonic protection of seniors that the project promised.

---

## 3. Project Structure

```
uqu_project/
├── data/                    Synthetic input data (CSV + JSON metadata)
│   ├── students.csv         5,000 students
│   ├── courses.csv          76 courses with prerequisites
│   ├── sections.csv         390 offered sections
│   ├── classrooms.csv       92 rooms across 7 buildings
│   ├── instructors.csv      80 faculty
│   ├── time_slots.csv       45 weekly slots
│   ├── prerequisites.csv    course dependency graph
│   ├── registration_requests.csv   29,215 requests
│   └── metadata.json
│
├── scripts/                 Pipeline modules (run in order)
│   ├── 01_generate_data.py     Synthetic data generator
│   ├── 02_optimize_schedule.py Two-phase optimization engine
│   ├── 03_evaluate.py          KPI evaluation + figures
│   ├── 04_build_dashboard.py   Self-contained HTML dashboard
│   ├── 05_ml_module.py         Recommender + forecaster + RF
│   ├── 06_build_report.js      Final Word report (DOCX)
│   └── 07_gantt_chart.py       Semester-2 Gantt chart
│
├── outputs/                 Run artifacts
│   ├── section_schedule.csv     scheduled sections (room + slots)
│   ├── allocations.csv          15,391 successful allocations
│   ├── waitlist.csv             13,824 waitlisted requests
│   ├── student_schedules.csv    per-student weekly schedule
│   ├── kpi_report.json          consolidated KPI summary
│   └── ...                      forecasts + ML predictions
│
├── figures/                 PNG visualizations (200 dpi)
│   ├── overview_dashboard.png
│   ├── kpi_priority_satisfaction.png
│   ├── kpi_classroom_utilization.png
│   ├── kpi_schedule_quality.png
│   ├── kpi_baseline_comparison.png
│   ├── forecast_demand.png
│   ├── conflict_prediction_features.png
│   └── gantt_chart_semester2.png
│
├── dashboard/
│   └── dashboard.html       Interactive HTML dashboard (1.3 MB)
│
└── reports/
    └── UQU-DS-2025-M09_Part2_Final_Report.docx    Final Word report
```

---

## 4. How to Run

```bash
# 1. Install dependencies
pip install faker pandas numpy ortools pulp matplotlib seaborn plotly scikit-learn openpyxl
npm install -g docx

# 2. Run the full pipeline (in order)
python scripts/01_generate_data.py
python scripts/02_optimize_schedule.py
python scripts/03_evaluate.py
python scripts/04_build_dashboard.py
python scripts/05_ml_module.py
python scripts/07_gantt_chart.py
node   scripts/06_build_report.js

# 3. Open the dashboard
open dashboard/dashboard.html
```

Each module is independent and reads only from the previous module's output files, so any single stage can be re-run.

---

## 5. Architecture

The system follows the three-layer architecture from the first-semester report, plus an ML layer:

```
   ┌─────────────────────────────────────────────────────────┐
   │  Application & Visualization Layer                      │
   │  • Interactive HTML Dashboard (Plotly + vanilla JS)     │
   │  • PNG figures + KPI tables                             │
   └─────────────────────────────────────────────────────────┘
                              ▲
   ┌─────────────────────────────────────────────────────────┐
   │  Machine Learning Layer                                 │
   │  • Collaborative-filtering recommender                  │
   │  • Demand forecaster (level-adjusted growth)            │
   │  • Random Forest conflict-risk classifier (94.7%)       │
   └─────────────────────────────────────────────────────────┘
                              ▲
   ┌─────────────────────────────────────────────────────────┐
   │  Processing & Optimization Layer                        │
   │  • Phase 1: section → (slot pattern, classroom)         │
   │  • Phase 2: priority-ordered student → section          │
   │  • Constraint checks: instructor, room, capacity,       │
   │    gender, time conflicts                               │
   └─────────────────────────────────────────────────────────┘
                              ▲
   ┌─────────────────────────────────────────────────────────┐
   │  Data Layer                                             │
   │  • Faker-generated synthetic data (privacy-safe)        │
   │  • CSV files (PostgreSQL-ready schema)                  │
   └─────────────────────────────────────────────────────────┘
```

---

## 6. Key Design Decisions

- **Two-phase decomposition.** Solving the joint section-scheduling + student-allocation problem as a single integer program is intractable at this scale. Splitting into two phases — each a constraint-satisfaction problem — keeps runtime under a minute and produces verifiable conflict-free results.

- **Priority weights.** Weights are monotonically increasing with academic level (1.0 → 5.0). Sorting requests by weight before allocation gives graduating students first claim on capacity-constrained sections.

- **Synthetic data only.** No real student data is touched. Distributions (level, GPA, program) are tuned to match UQU College of Computing realities.

- **Self-contained dashboard.** The HTML file embeds all data — no server, no API keys, no setup. It can be emailed to advisors or hosted on any static site.

- **Reproducibility.** All random seeds are fixed; re-running the pipeline yields identical results.

---

## 7. Limitations and Future Work

- Greedy allocation is not globally optimal; a full ILP or genetic algorithm could close the remaining gap.
- Synthetic data cannot capture every real-world registration pattern.
- The dashboard is read-only; production deployment needs auth + SIS integration.
- Future: NSGA-II multi-objective optimization, RL-based adaptive allocation, native Arabic UI.

---

## 8. Deliverables Checklist

| Item | Status | File |
|---|---|---|
| Synthetic dataset (8 tables, 35K+ records) | ✅ | `data/` |
| Two-phase optimizer | ✅ | `scripts/02_optimize_schedule.py` |
| Five-KPI evaluation | ✅ | `outputs/kpi_report.json` |
| Baseline comparison vs FIFO | ✅ | `figures/kpi_baseline_comparison.png` |
| Course recommender | ✅ | `outputs/sample_recommendations.json` |
| Demand forecasting | ✅ | `outputs/course_demand_forecast.csv` |
| Conflict-risk classifier | ✅ | `outputs/conflict_risk_predictions.csv` |
| Interactive dashboard | ✅ | `dashboard/dashboard.html` |
| Final Word report | ✅ | `reports/UQU-DS-2025-M09_Part2_Final_Report.docx` |
| Semester-2 Gantt chart | ✅ | `figures/gantt_chart_semester2.png` |

---

*UQU Data Science Department · Spring 2026*
