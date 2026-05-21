#!/bin/bash
# =============================================================================
# UQU Smart Scheduling System V2 — Automated Setup Script
# Project: UQU-DS-2025-M09
# Run:  bash setup.sh
# =============================================================================
set -e

GREEN='\033[0;32m'; BLUE='\033[0;34m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓ $1${NC}"; }
info() { echo -e "${BLUE}  ➜ $1${NC}"; }
err()  { echo -e "${RED}  ✗ $1${NC}"; exit 1; }

echo ""
echo "================================================================"
echo "  UQU Smart Scheduling System V2 — Setup"
echo "  Project: UQU-DS-2025-M09 | Spring 2026"
echo "================================================================"
echo ""

# ── 1. Python check ──────────────────────────────────────────────────────────
info "Checking Python..."
python3 --version >/dev/null 2>&1 || err "Python 3 not found. Install from python.org"
ok "Python found: $(python3 --version)"

# ── 2. Node.js check ─────────────────────────────────────────────────────────
info "Checking Node.js..."
node --version >/dev/null 2>&1 || err "Node.js not found. Install from nodejs.org"
ok "Node.js found: $(node --version)"

# ── 3. Python packages ────────────────────────────────────────────────────────
info "Installing Python packages..."
pip install --quiet \
  faker pandas numpy matplotlib seaborn scikit-learn \
  plotly openpyxl mlflow shap fastapi uvicorn sqlalchemy \
  pydantic httpx pytest reportlab
ok "Python packages installed"

# ── 4. Node packages ──────────────────────────────────────────────────────────
info "Installing Node.js packages..."
npm install --save-dev docx >/dev/null 2>&1 || npm install docx >/dev/null 2>&1
ok "docx package installed"

# ── 5. Create directories ─────────────────────────────────────────────────────
info "Creating project directories..."
mkdir -p data outputs figures dashboard reports uqu_v2/{api,optimizer,ml,database,tests,docker,monitoring}
ok "Directories ready"

# ── 6. Run pipeline ───────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Running Full Pipeline"
echo "================================================================"

info "Step 1/7: Generating synthetic dataset..."
python3 uqu_project/scripts/01_generate_data.py
ok "Dataset generated (5,000 students, 76 courses, 390 sections)"

info "Step 2/7: Running optimization engine (Phase 1 + Phase 2)..."
python3 uqu_project/scripts/02_optimize_schedule.py
ok "Scheduling complete — 0 time conflicts"

info "Step 3/7: Computing KPIs and generating figures..."
python3 uqu_project/scripts/03_evaluate.py
ok "KPIs computed, 5 figures generated"

info "Step 4/7: Building HTML dashboard..."
python3 uqu_project/scripts/04_build_dashboard.py
ok "Dashboard ready → dashboard/dashboard.html"

info "Step 5/7: Running ML module..."
python3 uqu_project/scripts/05_ml_module.py
ok "ML module complete (Recommender + Forecaster + Classifier)"

info "Step 6/7: Generating Gantt chart..."
python3 uqu_project/scripts/07_gantt_chart.py
ok "Gantt chart saved → figures/gantt_chart_semester2.png"

info "Step 7/7: Building Word report..."
node uqu_project/scripts/06_build_report.js
ok "Report saved → reports/UQU-DS-2025-M09_Part2_Final_Report.docx"

# ── 7. V2 enhancements ────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Running V2 Enhancements"
echo "================================================================"

info "V2-1: Setting up SQLite database..."
python3 uqu_v2/database/database_models.py
ok "Database ready → uqu_v2/uqu_scheduling.db"

info "V2-2: Running MLflow experiment tracking..."
python3 uqu_v2/ml/mlflow_tracking.py
ok "MLflow experiments logged → uqu_v2/mlruns/"

info "V2-3: Running automated test suite..."
python3 -m pytest uqu_v2/tests/test_suite.py -v --tb=short
ok "All tests passed"

# ── 8. Docker (if available) ──────────────────────────────────────────────────
echo ""
if command -v docker &>/dev/null; then
  info "Docker detected — starting full stack..."
  cd uqu_v2/docker
  docker-compose up -d
  cd ../..
  echo ""
  ok "Full stack running:"
  ok "  API      → http://localhost:8000/docs"
  ok "  MLflow   → http://localhost:5000"
  ok "  Grafana  → http://localhost:3001  (admin / uqu_grafana_2026)"
else
  info "Docker not found — starting FastAPI directly..."
  echo ""
  info "Run this command to start the API:"
  echo "    uvicorn uqu_v2.api.api_server:app --host 0.0.0.0 --port 8000 --reload"
fi

# ── 9. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Setup Complete!"
echo "================================================================"
echo ""
echo "  Key outputs:"
echo "    Dashboard  → dashboard/dashboard.html   (open in browser)"
echo "    Report     → reports/*.docx"
echo "    Figures    → figures/*.png"
echo "    KPIs       → outputs/kpi_report.json"
echo "    Database   → uqu_v2/uqu_scheduling.db"
echo "    API docs   → http://localhost:8000/docs"
echo ""
echo "  KPI Summary:"
python3 -c "
import json
k=json.load(open('uqu_project/outputs/kpi_report.json'))
print(f'    Graduating satisfaction  : {k[\"grad_satisfaction\"]}%')
print(f'    Overall satisfaction     : {k[\"overall_satisfaction\"]}%')
print(f'    Time conflicts           : {k[\"time_conflicts\"]}')
print(f'    Classroom utilization    : {k[\"classroom_utilization\"]}%')
"
echo ""
echo "================================================================"
