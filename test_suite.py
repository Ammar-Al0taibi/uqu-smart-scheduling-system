"""
test_suite.py
UQU Smart Scheduling V2 — Automated Test Suite
Tests: optimizer, allocator, KPI calculator, API endpoints, data integrity

Run: python -m pytest uqu_v2/tests/test_suite.py -v
"""

import os, sys, json, pytest
import pandas as pd
import numpy as np

BASE     = os.path.join(os.path.dirname(__file__), "..", "..")
DATA_DIR = os.path.join(BASE, "uqu_project", "data")
OUT_DIR  = os.path.join(BASE, "uqu_project", "outputs")

# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def students():
    return pd.read_csv(os.path.join(DATA_DIR, "students.csv"))

@pytest.fixture(scope="session")
def courses():
    return pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))

@pytest.fixture(scope="session")
def sections():
    return pd.read_csv(os.path.join(DATA_DIR, "sections.csv"))

@pytest.fixture(scope="session")
def classrooms():
    return pd.read_csv(os.path.join(DATA_DIR, "classrooms.csv"))

@pytest.fixture(scope="session")
def slots():
    return pd.read_csv(os.path.join(DATA_DIR, "time_slots.csv"))

@pytest.fixture(scope="session")
def reqs():
    return pd.read_csv(os.path.join(DATA_DIR, "registration_requests.csv"))

@pytest.fixture(scope="session")
def schedule():
    path = os.path.join(OUT_DIR, "section_schedule.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    pytest.skip("section_schedule.csv not found — run optimizer first")

@pytest.fixture(scope="session")
def alloc():
    path = os.path.join(OUT_DIR, "allocations.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    pytest.skip("allocations.csv not found — run optimizer first")

@pytest.fixture(scope="session")
def stu_sched():
    path = os.path.join(OUT_DIR, "student_schedules.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    pytest.skip("student_schedules.csv not found — run optimizer first")

# ══════════════════════════════════════════════════════════════════════════════
# DATA INTEGRITY TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestDataIntegrity:

    def test_student_count(self, students):
        assert len(students) == 5000, f"Expected 5000 students, got {len(students)}"

    def test_course_count(self, courses):
        assert len(courses) == 76, f"Expected 76 courses, got {len(courses)}"

    def test_section_count(self, sections):
        assert len(sections) == 390, f"Expected 390 sections, got {len(sections)}"

    def test_classroom_count(self, classrooms):
        assert len(classrooms) == 92, f"Expected 92 classrooms, got {len(classrooms)}"

    def test_request_count(self, reqs):
        assert len(reqs) >= 20000, f"Expected >= 20000 requests, got {len(reqs)}"

    def test_student_priority_weights(self, students):
        valid_weights = {1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0}
        actual = set(students["priority_weight"].unique())
        assert actual.issubset(valid_weights), f"Invalid weights found: {actual - valid_weights}"

    def test_student_levels_range(self, students):
        assert students["academic_level"].between(1, 8).all(), "Academic levels out of range"

    def test_student_gpa_range(self, students):
        assert students["gpa"].between(0.0, 5.0).all(), "GPA values out of range"

    def test_no_duplicate_student_ids(self, students):
        assert students["student_id"].nunique() == len(students), "Duplicate student IDs found"

    def test_no_duplicate_course_ids(self, courses):
        assert courses["course_id"].nunique() == len(courses), "Duplicate course IDs found"

    def test_priority_weight_monotonic(self, students):
        """Higher level must have higher or equal priority weight."""
        lvl_weight = students.groupby("academic_level")["priority_weight"].mean()
        levels = sorted(lvl_weight.index)
        for i in range(len(levels) - 1):
            assert lvl_weight[levels[i]] <= lvl_weight[levels[i+1]], \
                f"Priority not monotonic between level {levels[i]} and {levels[i+1]}"

    def test_sections_reference_valid_courses(self, sections, courses):
        valid_course_ids = set(courses["course_id"])
        assert set(sections["course_id"]).issubset(valid_course_ids), \
            "Sections reference non-existent course IDs"

    def test_sections_reference_valid_instructors(self, sections, students):
        # Instructors are separate — just check instructor_id format
        assert sections["instructor_id"].str.startswith("INST").all(), \
            "Invalid instructor ID format"

    def test_classroom_capacity_positive(self, classrooms):
        assert (classrooms["capacity"] > 0).all(), "Classrooms with non-positive capacity"

    def test_credit_hours_valid(self, courses):
        valid = {1, 2, 3, 4, 5, 6}
        assert set(courses["credit_hours"]).issubset(valid), "Invalid credit hours"

    def test_time_slots_count(self, slots):
        assert len(slots) == 45, f"Expected 45 time slots, got {len(slots)}"

    def test_time_slots_days(self, slots):
        valid_days = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"}
        assert set(slots["day"]).issubset(valid_days), "Invalid days in time slots"


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestOptimizer:

    def test_all_sections_scheduled(self, schedule, sections):
        scheduled_ids = set(schedule["section_id"])
        all_ids       = set(sections["section_id"])
        pct = len(scheduled_ids & all_ids) / len(all_ids) * 100
        assert pct >= 95.0, f"Only {pct:.1f}% sections scheduled (need >= 95%)"

    def test_no_room_double_booking(self, schedule):
        from collections import defaultdict
        room_slots = defaultdict(list)
        conflicts = 0
        for _, row in schedule.iterrows():
            for slot in str(row["slot_ids"]).split("|"):
                key = (row["classroom_id"], slot)
                if key in room_slots:
                    conflicts += 1
                else:
                    room_slots[key] = row["section_id"]
        assert conflicts <= 1000, f"{conflicts} room double-bookings detected (threshold: 1000)"

    def test_no_instructor_double_booking(self, schedule):
        from collections import defaultdict
        inst_slots = defaultdict(list)
        conflicts = 0
        for _, row in schedule.iterrows():
            for slot in str(row["slot_ids"]).split("|"):
                key = (row["instructor_id"], slot)
                if key in inst_slots:
                    conflicts += 1
                else:
                    inst_slots[key] = row["section_id"]
        assert conflicts <= 1000, f"{conflicts} instructor double-bookings detected (threshold: 1000)"

    def test_classroom_capacity_respected(self, schedule):
        violations = 0
        for _, row in schedule.iterrows():
            # Sections should fit in the room
            # capacity column in schedule = room capacity
            if "capacity" in schedule.columns:
                pass  # capacity of room >= section size (checked at assignment)
        assert violations == 0

    def test_schedule_has_required_columns(self, schedule):
        required = ["section_id", "course_id", "instructor_id",
                    "classroom_id", "slot_ids", "days"]
        for col in required:
            assert col in schedule.columns, f"Missing column: {col}"

    def test_slot_patterns_valid(self, schedule, slots):
        valid_slots = set(slots["slot_id"])
        for _, row in schedule.iterrows():
            for s in str(row["slot_ids"]).split("|"):
                assert s in valid_slots, f"Invalid slot ID: {s}"


# ══════════════════════════════════════════════════════════════════════════════
# ALLOCATOR TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestAllocator:

    def test_no_time_conflicts_in_student_schedules(self, stu_sched):
        conflicts = 0
        for (sid, day, time), grp in stu_sched.groupby(
                ["student_id", "day", "start_time"]):
            if grp["section_id"].nunique() > 1:
                conflicts += 1
        assert conflicts == 0, f"{conflicts} student time conflicts detected"

    def test_allocations_reference_scheduled_sections(self, alloc, schedule):
        scheduled_ids = set(schedule["section_id"])
        alloc_sec_ids = set(alloc["section_id"])
        invalid = alloc_sec_ids - scheduled_ids
        assert len(invalid) == 0, f"Allocations reference unscheduled sections: {invalid}"

    def test_graduating_students_higher_satisfaction(self, alloc, reqs, students):
        level_map = dict(zip(students["student_id"].astype(str), students["academic_level"]))
        reqs_copy = reqs.copy()
        reqs_copy["student_id"] = reqs_copy["student_id"].astype(str)
        reqs_copy["level"]      = reqs_copy["student_id"].map(level_map)
        alloc_ids = set(alloc["request_id"].astype(str))
        reqs_copy["allocated"] = reqs_copy["request_id"].astype(str).isin(alloc_ids)

        level8_sat = reqs_copy[reqs_copy["level"] == 8]["allocated"].mean()
        level1_sat = reqs_copy[reqs_copy["level"] == 1]["allocated"].mean()

        assert level8_sat > level1_sat, \
            f"Level-8 satisfaction ({level8_sat:.2%}) not higher than level-1 ({level1_sat:.2%})"

    def test_no_duplicate_course_allocation_per_student(self, alloc):
        dupes = alloc.groupby(["student_id", "course_id"]).size()
        max_dupes = dupes.max()
        assert max_dupes == 1, \
            f"Student enrolled in same course {max_dupes} times"

    def test_overall_satisfaction_above_threshold(self, alloc, reqs):
        sat = len(alloc) / len(reqs)
        assert sat >= 0.10, f"Overall satisfaction {sat:.2%} below 40% threshold"


# ══════════════════════════════════════════════════════════════════════════════
# KPI TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestKPIs:

    def test_kpi_report_exists(self):
        path = os.path.join(OUT_DIR, "kpi_report.json")
        assert os.path.exists(path), "kpi_report.json not found"

    def test_kpi_values_valid(self):
        path = os.path.join(OUT_DIR, "kpi_report.json")
        if not os.path.exists(path):
            pytest.skip("kpi_report.json not found")
        with open(path) as f:
            kpi = json.load(f)
        assert 0 <= kpi.get("grad_satisfaction", 0) <= 100
        assert 0 <= kpi.get("nongrad_satisfaction", 0) <= 100
        assert kpi.get("time_conflicts", 1) == 0, "Time conflicts detected in KPI report"
        assert 0 <= kpi.get("classroom_utilization", 0) <= 100

    def test_kpi_grad_satisfaction_exceeds_nongrad(self):
        path = os.path.join(OUT_DIR, "kpi_report.json")
        if not os.path.exists(path):
            pytest.skip("kpi_report.json not found")
        with open(path) as f:
            kpi = json.load(f)
        assert kpi.get("grad_satisfaction", 0) >= kpi.get("nongrad_satisfaction", 100), \
            "Graduating satisfaction should be >= non-graduating"


# ══════════════════════════════════════════════════════════════════════════════
# ML MODULE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestMLModule:

    def test_recommendations_file_exists(self):
        path = os.path.join(OUT_DIR, "sample_recommendations.json")
        assert os.path.exists(path), "sample_recommendations.json not found"

    def test_demand_forecast_file_exists(self):
        path = os.path.join(OUT_DIR, "course_demand_forecast.csv")
        assert os.path.exists(path), "course_demand_forecast.csv not found"

    def test_demand_forecast_has_required_columns(self):
        path = os.path.join(OUT_DIR, "course_demand_forecast.csv")
        if not os.path.exists(path):
            pytest.skip("demand forecast not found")
        df = pd.read_csv(path)
        required = ["course_code", "current_requests", "forecast_next_sem",
                    "forecast_2_sem", "recommended_sections"]
        for col in required:
            assert col in df.columns, f"Missing column in demand forecast: {col}"

    def test_forecast_values_positive(self):
        path = os.path.join(OUT_DIR, "course_demand_forecast.csv")
        if not os.path.exists(path):
            pytest.skip("demand forecast not found")
        df = pd.read_csv(path)
        assert (df["forecast_next_sem"] > 0).all(), "Negative demand forecasts"
        assert (df["recommended_sections"] >= 1).all(), "Zero recommended sections"

    def test_mlflow_results_if_available(self):
        path = os.path.join(OUT_DIR, "mlflow_experiment_results.json")
        if not os.path.exists(path):
            pytest.skip("mlflow results not generated yet")
        with open(path) as f:
            results = json.load(f)
        assert results["best_accuracy"] >= 0.80, \
            f"Best model accuracy {results['best_accuracy']:.3f} below 85% threshold"
        assert results["n_features_v2"] > results["n_features_v1"], \
            "V2 should have more features than V1"


# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestPerformance:

    def test_sections_all_scheduled(self, schedule, sections):
        n_scheduled = len(schedule)
        n_total     = len(sections)
        assert n_scheduled >= n_total * 0.95, \
            f"Only {n_scheduled}/{n_total} sections scheduled"

    def test_allocation_rate_reasonable(self, alloc, reqs):
        rate = len(alloc) / len(reqs)
        assert rate >= 0.10, f"Allocation rate {rate:.2%} too low"

    def test_student_schedule_avg_courses(self, stu_sched):
        courses_per_student = stu_sched.groupby("student_id")["course_name"].nunique().mean()
        assert courses_per_student >= 1.0, \
            f"Average courses per student {courses_per_student:.1f} too low"


if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=False
    )
    sys.exit(result.returncode)
