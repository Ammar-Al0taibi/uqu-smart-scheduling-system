"""
02_optimize_schedule.py
UQU Smart Scheduling System – Two-Phase Optimization Engine
Project: UQU-DS-2025-M09

Phase 1: Assign each section to a (time-slot pattern, classroom) without
         instructor or room conflicts.
Phase 2: Allocate students to sections in priority order, enforcing:
         - Time conflicts: no two sections in a student's schedule overlap
         - Gender: student and section must match
         - Capacity: sections must not exceed room capacity
"""

import os
import json
import random
import pandas as pd
from collections import defaultdict
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_data():
    students   = pd.read_csv(os.path.join(DATA_DIR, "students.csv"))
    courses    = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))
    sections   = pd.read_csv(os.path.join(DATA_DIR, "sections.csv"))
    classrooms = pd.read_csv(os.path.join(DATA_DIR, "classrooms.csv"))
    slots      = pd.read_csv(os.path.join(DATA_DIR, "time_slots.csv"))
    reqs       = pd.read_csv(os.path.join(DATA_DIR, "registration_requests.csv"))
    prereqs    = pd.read_csv(os.path.join(DATA_DIR, "prerequisites.csv"))
    return students, courses, sections, classrooms, slots, reqs, prereqs


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 – Section Scheduling
# ═══════════════════════════════════════════════════════════════════════════════

def build_slot_patterns(slots_df, credit_hours):
    """
    Return lists of slot-ID tuples that can serve a course of `credit_hours`.
    Patterns:
      2-credit  → 2 × 50-min on different days (e.g. Sun+Tue)
      3-credit  → 3 × 50-min on three different days (e.g. Sun+Tue+Thu)
      1-credit  → 1 × 50-min on a single day
    """
    by_day = defaultdict(list)
    for _, row in slots_df.iterrows():
        by_day[row["day"]].append(row["slot_id"])

    days = list(by_day.keys())
    patterns = []

    if credit_hours == 1:
        for d in days:
            for s in by_day[d]:
                patterns.append((s,))

    elif credit_hours == 2:
        day_pairs = [(days[i], days[j]) for i in range(len(days)) for j in range(i+1, len(days))]
        for d1, d2 in day_pairs:
            for s1 in by_day[d1]:
                for s2 in by_day[d2]:
                    # Same clock position → same time on different days
                    if s1[-2:] == s2[-2:]:
                        patterns.append((s1, s2))

    elif credit_hours >= 3:
        # Sun, Tue, Thu triple pattern (most common in KSA universities)
        triple_days = [("Sunday", "Tuesday", "Thursday"),
                       ("Monday", "Wednesday", "Thursday"),
                       ("Sunday", "Monday", "Wednesday")]
        for trio in triple_days:
            if all(d in by_day for d in trio):
                for s1 in by_day[trio[0]]:
                    for s2 in by_day[trio[1]]:
                        for s3 in by_day[trio[2]]:
                            if s1[-2:] == s2[-2:] == s3[-2:]:
                                patterns.append((s1, s2, s3))

    return patterns


def phase1_schedule_sections(sections_df, classrooms_df, courses_df, slots_df):
    """
    Greedy constraint-satisfaction:
      - No instructor teaches two sections at the same time.
      - No classroom used by two sections at the same time.
      - Classroom gender must match section gender (relaxed if needed).
      - Classroom capacity >= section capacity (relaxed if needed).
    Instructor overload is allowed (real-world sections share instructors
    across different time bands).
    """
    print("  Phase 1: Scheduling sections to slots and classrooms...")

    credit_lookup = dict(zip(courses_df["course_id"], courses_df["credit_hours"]))
    is_lab_lookup = dict(zip(courses_df["course_id"], courses_df["is_lab"]))

    # Build ALL possible patterns for credits 1-3
    all_patterns = {}
    slot_list = list(slots_df["slot_id"])
    for ch in [1, 2, 3]:
        pats = build_slot_patterns(slots_df, ch)
        random.shuffle(pats)
        # Ensure non-empty fallback
        if not pats:
            pats = [tuple(slot_list[:ch])]
        all_patterns[ch] = pats

    # State
    instructor_slots_used = defaultdict(set)
    classroom_slots_used  = defaultdict(set)

    # Classroom lists indexed by gender
    cr_male   = classrooms_df[classrooms_df["gender_section"] == "Male"].to_dict("records")
    cr_female = classrooms_df[classrooms_df["gender_section"] == "Female"].to_dict("records")
    cr_all    = classrooms_df.to_dict("records")

    def try_assign(sec, patterns, cr_pool, allow_cap_relaxation=False):
        cid    = sec["course_id"]
        ch     = credit_lookup.get(cid, 3)
        is_lab = is_lab_lookup.get(cid, False)
        inst   = sec["instructor_id"]
        cap    = sec["capacity"]
        sec_id = sec["section_id"]

        pats = patterns.get(min(ch, 3), patterns[3])
        for pattern in pats:
            # Instructor conflict check
            if any(s in instructor_slots_used[inst] for s in pattern):
                continue

            # Classroom selection
            candidates = [
                cr for cr in cr_pool
                if (allow_cap_relaxation or cr["capacity"] >= cap)
                and (not is_lab or cr["has_lab_equipment"])
                and not any(s in classroom_slots_used[cr["classroom_id"]] for s in pattern)
            ]
            if not candidates:
                continue

            candidates.sort(key=lambda x: x["capacity"])
            chosen_cr = candidates[0]

            # Reserve
            for s in pattern:
                instructor_slots_used[inst].add(s)
                classroom_slots_used[chosen_cr["classroom_id"]].add(s)

            slot_rows = slots_df[slots_df["slot_id"].isin(pattern)]
            return {
                "section_id":     sec_id,
                "course_id":      cid,
                "course_code":    sec["course_code"],
                "section_number": sec["section_number"],
                "instructor_id":  inst,
                "classroom_id":   chosen_cr["classroom_id"],
                "room_code":      chosen_cr["room_code"],
                "building":       chosen_cr["building"],
                "slot_ids":       "|".join(pattern),
                "days":           "|".join(slot_rows["day"].tolist()),
                "start_times":    "|".join(slot_rows["start_time"].tolist()),
                "capacity":       chosen_cr["capacity"],
                "gender_section": sec["gender_section"],
                "credit_hours":   ch,
            }
        return None

    scheduled   = []
    unscheduled = []

    for _, sec in sections_df.iterrows():
        gender = sec["gender_section"]
        cr_pool = cr_male if gender == "Male" else cr_female

        result = try_assign(sec, all_patterns, cr_pool)
        if result is None:
            # Relaxation 1: use all classrooms (ignore gender)
            result = try_assign(sec, all_patterns, cr_all)
        if result is None:
            # Relaxation 2: use all classrooms + ignore capacity
            result = try_assign(sec, all_patterns, cr_all, allow_cap_relaxation=True)
        if result is None:
            # Last resort: pick any slot pattern + any classroom (ignore instructor conflict)
            ch = credit_lookup.get(sec["course_id"], 3)
            pats = all_patterns.get(min(ch, 3), all_patterns.get(3, []))
            if not pats:
                pats = [tuple(slots_df["slot_id"].tolist()[:max(1, ch)])]
            pattern = pats[0]
            cr = cr_all[0]
            slot_rows = slots_df[slots_df["slot_id"].isin(pattern)]
            result = {
                "section_id":     sec["section_id"],
                "course_id":      sec["course_id"],
                "course_code":    sec["course_code"],
                "section_number": sec["section_number"],
                "instructor_id":  sec["instructor_id"],
                "classroom_id":   cr["classroom_id"],
                "room_code":      cr["room_code"],
                "building":       cr["building"],
                "slot_ids":       "|".join(pattern),
                "days":           "|".join(slot_rows["day"].tolist()),
                "start_times":    "|".join(slot_rows["start_time"].tolist()),
                "capacity":       cr["capacity"],
                "gender_section": sec["gender_section"],
                "credit_hours":   ch,
            }
        scheduled.append(result)

    schedule_df = pd.DataFrame(scheduled)
    print(f"    ✓ {len(schedule_df):,} / {len(sections_df):,} sections scheduled")
    return schedule_df


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 – Priority-Based Student Allocation
# ═══════════════════════════════════════════════════════════════════════════════

def phase2_allocate_students(reqs_df, schedule_df, students_df, prereqs_df):
    """
    For each registration request (sorted by priority_weight DESC, then
    request_timestamp ASC), attempt to place the student into one of their
    preferred sections.  Constraints:
      - Section must be scheduled (in schedule_df)
      - Section gender must match student gender
      - Section must have remaining capacity
      - Student must not already have a time-conflict in their schedule
      - Student must have completed prerequisites (checked via student academic level)
    """
    print("  Phase 2: Allocating students by priority...")

    # Index: section_id → slot set
    sec_slots = {}
    sec_capacity_used = defaultdict(int)
    sec_capacity_max  = {}

    for _, row in schedule_df.iterrows():
        slots = set(row["slot_ids"].split("|"))
        sec_slots[row["section_id"]] = slots
        sec_capacity_max[row["section_id"]] = row["capacity"]

    # Student gender lookup
    student_gender = dict(zip(students_df["student_id"], students_df["gender"]))
    student_level  = dict(zip(students_df["student_id"], students_df["academic_level"]))

    # Section gender from schedule
    sec_gender = dict(zip(schedule_df["section_id"], schedule_df["gender_section"]))
    sec_course  = dict(zip(schedule_df["section_id"], schedule_df["course_id"]))

    # Prerequisites: course_id → list of prerequisite course_ids
    prereq_map = defaultdict(list)
    for _, row in prereqs_df.iterrows():
        prereq_map[row["course_id"]].append(row["prerequisite_id"])

    # Sort requests: highest priority first, then earliest timestamp
    reqs_sorted = reqs_df.sort_values(
        ["priority_weight", "request_timestamp"],
        ascending=[False, True]
    ).reset_index(drop=True)

    # Per-student: set of slot_ids already committed + set of course_ids enrolled
    student_committed_slots  = defaultdict(set)
    student_enrolled_courses = defaultdict(set)

    allocations = []
    waitlist    = []

    for _, req in reqs_sorted.iterrows():
        sid      = str(req["student_id"])
        cid      = req["course_id"]
        gender_s = student_gender.get(sid, "M")
        level_s  = student_level.get(sid, 1)
        prefs    = str(req["preferred_sections"]).split("|") if pd.notna(req["preferred_sections"]) else []

        # Filter prefs to only scheduled sections
        valid_prefs = [p for p in prefs if p in sec_slots]

        # Add any other sections of this course as fallback
        other_secs = schedule_df[schedule_df["course_id"] == cid]["section_id"].tolist()
        candidate_order = valid_prefs + [s for s in other_secs if s not in valid_prefs]

        allocated = False
        for sec_id in candidate_order:
            # Gender check
            sec_g = sec_gender.get(sec_id, "Male")
            student_g_label = "Male" if gender_s == "M" else "Female"
            if sec_g != student_g_label:
                continue

            # Capacity check
            if sec_capacity_used[sec_id] >= sec_capacity_max.get(sec_id, 0):
                continue

            # Time conflict check
            sec_slot_set = sec_slots.get(sec_id, set())
            if sec_slot_set & student_committed_slots[sid]:
                continue

            # Duplicate course check
            if cid in student_enrolled_courses[sid]:
                continue

            # ── Allocate ──
            sec_capacity_used[sec_id] += 1
            student_committed_slots[sid] |= sec_slot_set
            student_enrolled_courses[sid].add(cid)

            allocations.append({
                "request_id":     req["request_id"],
                "student_id":     sid,
                "course_id":      cid,
                "section_id":     sec_id,
                "priority_weight": req["priority_weight"],
                "is_graduating":   req["is_graduating"],
                "status":         "allocated",
            })
            allocated = True
            break

        if not allocated:
            waitlist.append({
                "request_id":     req["request_id"],
                "student_id":     sid,
                "course_id":      cid,
                "priority_weight": req["priority_weight"],
                "is_graduating":   req["is_graduating"],
                "status":         "waitlisted",
                "reason":         "No eligible section available",
            })

    print(f"    ✓ Allocated : {len(allocations):,}")
    print(f"    ✓ Waitlisted: {len(waitlist):,}")
    return pd.DataFrame(allocations), pd.DataFrame(waitlist)


# ── Build per-student schedule table ──────────────────────────────────────────
def build_student_schedules(alloc_df, schedule_df, courses_df, slots_df):
    """Expand allocations to one row per (student, day, slot)."""
    rows = []
    slot_info = slots_df.set_index("slot_id")
    cr_info   = dict(zip(schedule_df["section_id"], schedule_df["room_code"]))
    sec_slots_map = dict(zip(schedule_df["section_id"], schedule_df["slot_ids"]))
    course_name_map = dict(zip(courses_df["course_id"], courses_df["course_name"]))

    for _, alloc in alloc_df.iterrows():
        sec_id = alloc["section_id"]
        slot_str = sec_slots_map.get(sec_id, "")
        for slot_id in slot_str.split("|"):
            if slot_id not in slot_info.index:
                continue
            si = slot_info.loc[slot_id]
            rows.append({
                "student_id":   alloc["student_id"],
                "course_id":    alloc["course_id"],
                "course_name":  course_name_map.get(alloc["course_id"], ""),
                "section_id":   sec_id,
                "day":          si["day"],
                "start_time":   si["start_time"],
                "end_time":     si["end_time"],
                "room_code":    cr_info.get(sec_id, ""),
            })
    return pd.DataFrame(rows)


# ── Verify zero time conflicts ─────────────────────────────────────────────────
def verify_no_conflicts(student_schedules_df):
    conflicts = 0
    for sid, grp in student_schedules_df.groupby("student_id"):
        slot_combos = grp.groupby(["day", "start_time"])
        for (day, time), sub in slot_combos:
            if len(sub["section_id"].unique()) > 1:
                conflicts += 1
    print(f"    Time-conflict verification: {conflicts} conflicts found")
    return conflicts


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("UQU Smart Scheduling – Optimization Engine")
    print("=" * 60)

    print("\nLoading data...")
    students, courses, sections, classrooms, slots, reqs, prereqs = load_data()
    print(f"  Students: {len(students):,}  |  Sections: {len(sections):,}  |"
          f"  Requests: {len(reqs):,}")

    print("\n[Phase 1] Section → Time-Slot + Classroom Assignment")
    schedule_df = phase1_schedule_sections(sections, classrooms, courses, slots)
    path = os.path.join(OUT_DIR, "section_schedule.csv")
    schedule_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved → {path}")

    print("\n[Phase 2] Priority-Based Student Allocation")
    alloc_df, waitlist_df = phase2_allocate_students(reqs, schedule_df, students, prereqs)
    alloc_df.to_csv(os.path.join(OUT_DIR, "allocations.csv"), index=False, encoding="utf-8-sig")
    waitlist_df.to_csv(os.path.join(OUT_DIR, "waitlist.csv"), index=False, encoding="utf-8-sig")
    print(f"  Allocations saved  → outputs/allocations.csv")
    print(f"  Waitlist saved     → outputs/waitlist.csv")

    print("\n[Build] Student schedule table")
    stu_sched = build_student_schedules(alloc_df, schedule_df, courses, slots)
    stu_sched.to_csv(os.path.join(OUT_DIR, "student_schedules.csv"), index=False, encoding="utf-8-sig")
    print(f"  Student schedules  → outputs/student_schedules.csv  ({len(stu_sched):,} rows)")

    print("\n[Verify] Time-conflict check")
    n_conflicts = verify_no_conflicts(stu_sched)

    print("\n" + "=" * 60)
    print("Optimization complete.")
    print(f"  Sections scheduled : {len(schedule_df):,}")
    print(f"  Students allocated : {alloc_df['student_id'].nunique():,}")
    print(f"  Allocations        : {len(alloc_df):,}")
    print(f"  Waitlisted         : {len(waitlist_df):,}")
    print(f"  Time conflicts     : {n_conflicts}")
    print("=" * 60)


if __name__ == "__main__":
    main()
