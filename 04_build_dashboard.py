"""
04_build_dashboard.py
UQU Smart Scheduling System – Interactive HTML Dashboard Builder
Project: UQU-DS-2025-M09

Generates a fully self-contained HTML dashboard (no server needed).
All data is embedded as JavaScript variables.
Charts are rendered by Plotly (loaded from CDN).
"""

import os
import json
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR  = os.path.join(BASE, "outputs")
DASH_DIR = os.path.join(BASE, "dashboard")
os.makedirs(DASH_DIR, exist_ok=True)


# ── Loaders ───────────────────────────────────────────────────────────────────
def load():
    students  = pd.read_csv(os.path.join(DATA_DIR, "students.csv"))
    courses   = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))
    sections  = pd.read_csv(os.path.join(DATA_DIR, "sections.csv"))
    alloc     = pd.read_csv(os.path.join(OUT_DIR,  "allocations.csv"))
    schedule  = pd.read_csv(os.path.join(OUT_DIR,  "section_schedule.csv"))
    reqs      = pd.read_csv(os.path.join(DATA_DIR, "registration_requests.csv"))
    kpi       = json.load(open(os.path.join(OUT_DIR, "kpi_report.json")))
    return students, courses, sections, alloc, schedule, reqs, kpi


# ── Prepare chart data ────────────────────────────────────────────────────────
def prepare_data(students, courses, sections, alloc, schedule, reqs, kpi):

    # 1. Satisfaction by level
    level_map = dict(zip(students["student_id"].astype(str), students["academic_level"]))
    reqs = reqs.copy()
    reqs["student_id"] = reqs["student_id"].astype(str)
    reqs["level"] = reqs["student_id"].map(level_map)
    alloc_ids = set(alloc["request_id"].astype(str))
    reqs["allocated"] = reqs["request_id"].astype(str).isin(alloc_ids)
    by_level = reqs.groupby("level")["allocated"].mean().reset_index()
    by_level["pct"] = (by_level["allocated"] * 100).round(1)

    # 2. Sections per course (top 15)
    course_sec_count = schedule.groupby("course_id").size().reset_index(name="n_sections")
    course_name_map  = dict(zip(courses["course_id"], courses["course_code"]))
    course_sec_count["course_code"] = course_sec_count["course_id"].map(course_name_map)
    course_sec_count = course_sec_count.sort_values("n_sections", ascending=False).head(15)

    # 3. Allocations by program
    program_map = dict(zip(students["student_id"].astype(str), students["program"]))
    alloc["program"] = alloc["student_id"].astype(str).map(program_map)
    by_program = alloc.groupby("program").size().reset_index(name="count")

    # 4. Allocation outcome summary
    total_reqs = len(reqs)
    total_alloc = len(alloc)
    outcome = {
        "labels": ["Allocated", "Waitlisted"],
        "values": [total_alloc, total_reqs - total_alloc],
    }

    # 5. Graduating vs non-graduating satisfaction
    grad_comp = {
        "categories": ["Graduating", "Non-Graduating", "Overall"],
        "values": [
            round(kpi["grad_satisfaction"], 1),
            round(kpi["nongrad_satisfaction"], 1),
            round(kpi["overall_satisfaction"], 1),
        ],
    }

    return by_level, course_sec_count, by_program, outcome, grad_comp


# ── HTML Template ─────────────────────────────────────────────────────────────
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>UQU Smart Scheduling Dashboard | UQU-DS-2025-M09</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {{
    --primary:   #1B5E20;
    --secondary: #4CAF50;
    --gold:      #F9A825;
    --blue:      #1565C0;
    --light-blue:#42A5F5;
    --warn:      #E53935;
    --bg:        #F5F7FA;
    --card-bg:   #FFFFFF;
    --text:      #212121;
    --muted:     #757575;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    direction: ltr;
  }}
  header {{
    background: linear-gradient(135deg, var(--primary) 0%, #2E7D32 100%);
    color: white;
    padding: 20px 32px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  header h1 {{ font-size: 1.4rem; font-weight: 700; }}
  header p  {{ font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }}
  header .badge {{
    background: rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 0.8rem;
  }}
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 16px;
    padding: 24px 32px 8px;
  }}
  .kpi-card {{
    background: var(--card-bg);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-top: 4px solid var(--primary);
    text-align: center;
  }}
  .kpi-card.gold  {{ border-top-color: var(--gold);      }}
  .kpi-card.blue  {{ border-top-color: var(--blue);      }}
  .kpi-card.warn  {{ border-top-color: var(--warn);      }}
  .kpi-card.lblue {{ border-top-color: var(--light-blue);}}
  .kpi-val  {{ font-size: 2rem; font-weight: 800; color: var(--primary); }}
  .kpi-card.gold  .kpi-val {{ color: var(--gold);       }}
  .kpi-card.blue  .kpi-val {{ color: var(--blue);       }}
  .kpi-card.warn  .kpi-val {{ color: var(--warn);       }}
  .kpi-card.lblue .kpi-val {{ color: var(--light-blue); }}
  .kpi-label {{ font-size: 0.75rem; color: var(--muted); margin-top: 6px; }}
  .charts-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    padding: 16px 32px;
  }}
  .chart-card {{
    background: var(--card-bg);
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    padding: 16px;
  }}
  .chart-card h3 {{
    font-size: 0.9rem;
    color: var(--primary);
    margin-bottom: 12px;
    font-weight: 600;
  }}
  .chart-full {{ grid-column: 1 / -1; }}
  footer {{
    text-align: center;
    padding: 20px;
    color: var(--muted);
    font-size: 0.75rem;
    border-top: 1px solid #E0E0E0;
    margin-top: 20px;
  }}
  .info-bar {{
    background: var(--primary);
    color: white;
    display: flex;
    justify-content: space-around;
    padding: 10px 32px;
    font-size: 0.8rem;
  }}
  .info-bar span {{ opacity: 0.9; }}
  .info-bar strong {{ font-weight: 700; }}
</style>
</head>
<body>

<header>
  <div>
    <h1>UQU Smart Scheduling System</h1>
    <p>An Intelligent System for Student Schedule Management and Priority-Based Class Allocation</p>
    <p style="margin-top:6px;font-size:0.75rem;opacity:0.7;">
      Umm Al-Qura University · College of Computing · Data Science Department
    </p>
  </div>
  <div class="badge">UQU-DS-2025-M09 · Spring 2026</div>
</header>

<div class="info-bar">
  <span>Students: <strong>5,000</strong></span>
  <span>Courses: <strong>76</strong></span>
  <span>Sections: <strong>390</strong></span>
  <span>Classrooms: <strong>92</strong></span>
  <span>Instructors: <strong>80</strong></span>
  <span>Requests: <strong>{TOTAL_REQUESTS:,}</strong></span>
</div>

<!-- KPI Cards -->
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-val">{GRAD_SAT}%</div>
    <div class="kpi-label">Graduating Student Satisfaction</div>
  </div>
  <div class="kpi-card blue">
    <div class="kpi-val">{NONGRAD_SAT}%</div>
    <div class="kpi-label">Non-Graduating Satisfaction</div>
  </div>
  <div class="kpi-card {CONFLICT_CLASS}">
    <div class="kpi-val">{CONFLICTS}</div>
    <div class="kpi-label">Time Conflicts</div>
  </div>
  <div class="kpi-card gold">
    <div class="kpi-val">{UTIL_RATE}%</div>
    <div class="kpi-label">Classroom Utilisation</div>
  </div>
  <div class="kpi-card lblue">
    <div class="kpi-val">{AVG_GAP}h</div>
    <div class="kpi-label">Avg Daily Idle Gap</div>
  </div>
</div>

<!-- Charts -->
<div class="charts-grid">

  <div class="chart-card">
    <h3>Satisfaction Rate by Academic Level</h3>
    <div id="chart-satisfaction"></div>
  </div>

  <div class="chart-card">
    <h3>Allocation Outcome</h3>
    <div id="chart-outcome"></div>
  </div>

  <div class="chart-card">
    <h3>Graduating vs Non-Graduating Satisfaction</h3>
    <div id="chart-grad-comp"></div>
  </div>

  <div class="chart-card">
    <h3>Allocations by Program</h3>
    <div id="chart-program"></div>
  </div>

  <div class="chart-card chart-full">
    <h3>Top 15 Courses by Number of Sections</h3>
    <div id="chart-courses"></div>
  </div>

</div>

<footer>
  UQU-DS-2025-M09 · Team: Raed Alhelali · Mohammed Alsarhani · Ammar Alotaibi · Moayad Hawsawi
  · Supervisor: Dr. Ahmed Bukhari · Spring 2026
</footer>

<script>
// ── Embedded data ──────────────────────────────────────────────────────────────
const DATA = {DATA_JSON};

const GREEN  = '#1B5E20';
const LGREEN = '#4CAF50';
const GOLD   = '#F9A825';
const BLUE   = '#1565C0';
const LBLUE  = '#42A5F5';
const WARN   = '#E53935';

const layout_defaults = {{
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor:  'rgba(0,0,0,0)',
  margin: {{ t: 20, b: 40, l: 40, r: 20 }},
  font: {{ family: 'Segoe UI, Arial', size: 11 }},
}};

// 1. Satisfaction by Level
Plotly.newPlot('chart-satisfaction', [{{
  x: DATA.by_level.levels,
  y: DATA.by_level.pcts,
  type: 'bar',
  marker: {{
    color: DATA.by_level.pcts.map(v => v >= 70 ? GREEN : v >= 50 ? LGREEN : GOLD)
  }},
  text: DATA.by_level.pcts.map(v => v.toFixed(1) + '%'),
  textposition: 'outside',
}}], {{
  ...layout_defaults,
  xaxis: {{ title: 'Academic Level', tickvals: [1,2,3,4,5,6,7,8] }},
  yaxis: {{ title: 'Satisfaction (%)', range: [0, 110] }},
}}, {{responsive: true}});

// 2. Allocation Outcome Pie
Plotly.newPlot('chart-outcome', [{{
  labels: DATA.outcome.labels,
  values: DATA.outcome.values,
  type: 'pie',
  hole: 0.4,
  marker: {{ colors: [GREEN, LBLUE] }},
  textinfo: 'label+percent',
}}], {{
  ...layout_defaults,
  showlegend: true,
  legend: {{ orientation: 'h', y: -0.1 }},
}}, {{responsive: true}});

// 3. Graduating vs Non-Graduating
Plotly.newPlot('chart-grad-comp', [{{
  x: DATA.grad_comp.categories,
  y: DATA.grad_comp.values,
  type: 'bar',
  marker: {{ color: [GREEN, BLUE, GOLD] }},
  text: DATA.grad_comp.values.map(v => v.toFixed(1) + '%'),
  textposition: 'outside',
}}], {{
  ...layout_defaults,
  yaxis: {{ title: 'Satisfaction (%)', range: [0, 110] }},
}}, {{responsive: true}});

// 4. By Program
Plotly.newPlot('chart-program', [{{
  labels: DATA.by_program.programs,
  values: DATA.by_program.counts,
  type: 'pie',
  marker: {{ colors: [GREEN, BLUE, GOLD, LBLUE, LGREEN, WARN] }},
  textinfo: 'label+percent',
}}], {{
  ...layout_defaults,
  showlegend: true,
  legend: {{ orientation: 'h', y: -0.1 }},
}}, {{responsive: true}});

// 5. Top Courses
Plotly.newPlot('chart-courses', [{{
  x: DATA.top_courses.codes,
  y: DATA.top_courses.counts,
  type: 'bar',
  marker: {{ color: LBLUE }},
  text: DATA.top_courses.counts,
  textposition: 'outside',
}}], {{
  ...layout_defaults,
  xaxis: {{ tickangle: -35 }},
  yaxis: {{ title: 'Number of Sections' }},
  margin: {{ t: 20, b: 80, l: 40, r: 20 }},
}}, {{responsive: true}});

</script>
</body>
</html>
"""


def build_dashboard(students, courses, sections, alloc, schedule, reqs, kpi):
    by_level, top_courses, by_program, outcome, grad_comp = prepare_data(
        students, courses, sections, alloc, schedule, reqs, kpi
    )

    data_json = json.dumps({
        "by_level": {
            "levels": by_level["level"].tolist(),
            "pcts":   by_level["pct"].tolist(),
        },
        "outcome": outcome,
        "grad_comp": grad_comp,
        "by_program": {
            "programs": by_program["program"].tolist(),
            "counts":   by_program["count"].tolist(),
        },
        "top_courses": {
            "codes":  top_courses["course_code"].tolist(),
            "counts": top_courses["n_sections"].tolist(),
        },
    })

    html = HTML_TEMPLATE.format(
        TOTAL_REQUESTS   = len(reqs),
        GRAD_SAT         = f"{kpi['grad_satisfaction']:.1f}",
        NONGRAD_SAT      = f"{kpi['nongrad_satisfaction']:.1f}",
        CONFLICTS        = kpi["time_conflicts"],
        CONFLICT_CLASS   = "warn" if kpi["time_conflicts"] > 0 else "",
        UTIL_RATE        = f"{kpi['classroom_utilization']:.1f}",
        AVG_GAP          = f"{kpi['avg_idle_gap']:.2f}",
        DATA_JSON        = data_json,
    )

    path = os.path.join(DASH_DIR, "dashboard.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  ✓ Dashboard saved → {path}  ({size_mb:.2f} MB)")
    return path


def main():
    print("=" * 60)
    print("UQU Smart Scheduling – Dashboard Builder")
    print("=" * 60)
    print("\nLoading data...")
    students, courses, sections, alloc, schedule, reqs, kpi = load()
    print(f"  {len(students):,} students  |  {len(alloc):,} allocations  |  {len(schedule):,} scheduled sections")
    print("\nBuilding dashboard...")
    build_dashboard(students, courses, sections, alloc, schedule, reqs, kpi)
    print("\nDone. Open dashboard/dashboard.html in any browser.")


if __name__ == "__main__":
    main()
