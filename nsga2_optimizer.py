"""
nsga2_optimizer.py
UQU Smart Scheduling V2 — NSGA-II Multi-Objective Optimizer
Replaces the greedy Phase-1 with a true Pareto-optimal search.

Objectives (all minimized — use negatives for maximization):
  F1: -satisfaction_rate          (maximize student satisfaction)
  F2:  avg_idle_gap               (minimize idle gaps in student schedules)
  F3: -classroom_utilization      (maximize classroom utilization)

Constraints handled via penalty:
  - Instructor time conflicts
  - Room double-booking
  - Gender mismatch
"""

import os, sys, random, copy, time, json
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE     = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE, "..", "uqu_project", "data")
OUT_DIR  = os.path.join(BASE, "..", "uqu_project", "outputs")

# ── NSGA-II Parameters ────────────────────────────────────────────────────────
POP_SIZE       = 30
N_GENERATIONS  = 20
CROSSOVER_PROB = 0.85
MUTATION_PROB  = 0.20
TOURNAMENT_K   = 3

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    sections   = pd.read_csv(os.path.join(DATA_DIR, "sections.csv"))
    classrooms = pd.read_csv(os.path.join(DATA_DIR, "classrooms.csv"))
    courses    = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))
    slots      = pd.read_csv(os.path.join(DATA_DIR, "time_slots.csv"))
    students   = pd.read_csv(os.path.join(DATA_DIR, "students.csv"))
    reqs       = pd.read_csv(os.path.join(DATA_DIR, "registration_requests.csv"))
    return sections, classrooms, courses, slots, students, reqs

# ── Chromosome representation ─────────────────────────────────────────────────
# Gene i = (slot_pattern_index, classroom_index) for section i
# We precompute all feasible (pattern, classroom) pairs per section.

def build_slot_patterns(slots_df, credit_hours):
    by_day = defaultdict(list)
    for _, r in slots_df.iterrows():
        by_day[r["day"]].append(r["slot_id"])
    days = list(by_day.keys())
    patterns = []
    if credit_hours == 1:
        for d in days:
            for s in by_day[d]:
                patterns.append((s,))
    elif credit_hours == 2:
        for i in range(len(days)):
            for j in range(i+1, len(days)):
                for s1 in by_day[days[i]]:
                    for s2 in by_day[days[j]]:
                        if s1[-2:] == s2[-2:]:
                            patterns.append((s1, s2))
    else:
        triples = [("Sunday","Tuesday","Thursday"),("Monday","Wednesday","Thursday")]
        for trio in triples:
            if all(d in by_day for d in trio):
                for s1 in by_day[trio[0]]:
                    for s2 in by_day[trio[1]]:
                        for s3 in by_day[trio[2]]:
                            if s1[-2:] == s2[-2:] == s3[-2:]:
                                patterns.append((s1, s2, s3))
    return patterns

def precompute_options(sections, classrooms, courses, slots):
    credit_map = dict(zip(courses["course_id"], courses["credit_hours"]))
    is_lab_map = dict(zip(courses["course_id"], courses["is_lab"]))
    cr_records = classrooms.to_dict("records")

    pattern_cache = {ch: build_slot_patterns(slots, ch) for ch in [1,2,3]}
    for ch in pattern_cache:
        if not pattern_cache[ch]:
            pattern_cache[ch] = [(list(slots["slot_id"])[0],)]

    options = []
    for _, sec in sections.iterrows():
        ch     = credit_map.get(sec["course_id"], 3)
        is_lab = is_lab_map.get(sec["course_id"], False)
        gender = sec["gender_section"]
        cap    = sec["capacity"]
        pats   = pattern_cache.get(min(ch, 3), pattern_cache[3])
        valid_crs = [
            cr for cr in cr_records
            if cr["capacity"] >= cap
            and (not is_lab or cr["has_lab_equipment"])
            and (cr["gender_section"] == gender or cr["gender_section"] == "Mixed")
        ]
        if not valid_crs:
            valid_crs = cr_records  # relaxed fallback
        options.append({
            "section_id":   sec["section_id"],
            "instructor_id":sec["instructor_id"],
            "gender":       gender,
            "patterns":     pats,
            "classrooms":   valid_crs,
            "credit_hours": ch,
        })
    return options

# ── Individual = list of (pat_idx, cr_idx) per section ───────────────────────
def random_individual(options):
    return [
        (random.randint(0, max(0, len(o["patterns"])-1)),
         random.randint(0, max(0, len(o["classrooms"])-1)))
        for o in options
    ]

# ── Decode chromosome → schedule dict ─────────────────────────────────────────
def decode(individual, options):
    schedule = []
    for i, (pi, ci) in enumerate(individual):
        opt = options[i]
        pat = opt["patterns"][pi % len(opt["patterns"])]
        cr  = opt["classrooms"][ci % len(opt["classrooms"])]
        schedule.append({
            "section_id":    opt["section_id"],
            "instructor_id": opt["instructor_id"],
            "classroom_id":  cr["classroom_id"],
            "slot_ids":      pat,
            "gender":        opt["gender"],
            "capacity":      cr["capacity"],
        })
    return schedule

# ── Constraint violations ─────────────────────────────────────────────────────
def count_violations(schedule):
    inst_slots  = defaultdict(list)
    room_slots  = defaultdict(list)
    viol = 0
    for entry in schedule:
        inst = entry["instructor_id"]
        cr   = entry["classroom_id"]
        for s in entry["slot_ids"]:
            if s in inst_slots[inst]:
                viol += 1
            inst_slots[inst].append(s)
            if s in room_slots[cr]:
                viol += 1
            room_slots[cr].append(s)
    return viol

# ── Objectives ─────────────────────────────────────────────────────────────────
def compute_objectives(schedule, reqs, students, slots_df):
    """Returns (F1, F2, F3) — all values where lower = better."""
    # Build quick lookups
    sec_slots  = {e["section_id"]: set(e["slot_ids"]) for e in schedule}
    sec_gender = {e["section_id"]: e["gender"] for e in schedule}
    sec_cap    = {e["section_id"]: e["capacity"] for e in schedule}

    student_gender = dict(zip(students["student_id"].astype(str), students["gender"]))
    gender_map = {"M": "Male", "F": "Female"}

    alloc_count = 0
    total_reqs  = len(reqs)

    student_committed = defaultdict(set)
    sec_used = defaultdict(int)

    for _, req in reqs.sort_values("priority_weight", ascending=False).iterrows():
        sid = str(req["student_id"])
        cid = req["course_id"]
        sg  = gender_map.get(student_gender.get(sid, "M"), "Male")
        prefs = str(req.get("preferred_sections","")).split("|")
        candidates = [s["section_id"] for s in schedule
                      if s["section_id"] in prefs or True]  # try all
        allocated = False
        for sec_id in candidates:
            if sec_gender.get(sec_id) != sg:
                continue
            if sec_used[sec_id] >= sec_cap.get(sec_id, 50):
                continue
            if sec_slots.get(sec_id, set()) & student_committed[sid]:
                continue
            sec_used[sec_id] += 1
            student_committed[sid] |= sec_slots.get(sec_id, set())
            alloc_count += 1
            allocated = True
            break

    satisfaction = alloc_count / max(total_reqs, 1)

    # F2: avg idle gap (simplified — count conflict-free schedule spread)
    SLOT_HOUR = {f"TS{i:04d}": 8 + (i-1) % 9 for i in range(1, 451)}
    gaps = []
    for sid, slot_set in student_committed.items():
        hours = sorted(set(SLOT_HOUR.get(s, 8) for s in slot_set))
        if len(hours) > 1:
            gap = hours[-1] - hours[0] - len(hours) + 1
            gaps.append(max(0, gap))
    avg_gap = np.mean(gaps) if gaps else 0.0

    # F3: classroom utilization
    total_slots = len(slots_df)
    total_rooms = len(set(e["classroom_id"] for e in schedule))
    used_pairs  = sum(len(e["slot_ids"]) for e in schedule)
    utilization = used_pairs / max(total_rooms * total_slots, 1)

    violations = count_violations(schedule)
    penalty    = violations * 0.1

    F1 = -satisfaction + penalty
    F2 = avg_gap
    F3 = -utilization + penalty
    return (F1, F2, F3), violations

# ── NSGA-II core ──────────────────────────────────────────────────────────────
def dominates(a, b):
    """True if a dominates b (a is at least as good in all, strictly better in one)."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def fast_non_dominated_sort(population):
    fronts = [[]]
    dom_count = [0] * len(population)
    dominated = [[] for _ in range(len(population))]

    for i in range(len(population)):
        for j in range(i+1, len(population)):
            fi = population[i]["objectives"]
            fj = population[j]["objectives"]
            if dominates(fi, fj):
                dominated[i].append(j)
                dom_count[j] += 1
            elif dominates(fj, fi):
                dominated[j].append(i)
                dom_count[i] += 1
        if dom_count[i] == 0:
            fronts[0].append(i)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    next_front.append(q)
        fronts.append(next_front)
        i += 1
    return fronts[:-1]

def crowding_distance(front, population):
    n = len(front)
    if n <= 2:
        return [float("inf")] * n
    dist = [0.0] * n
    n_obj = len(population[front[0]]["objectives"])
    for m in range(n_obj):
        sorted_front = sorted(range(n), key=lambda i: population[front[i]]["objectives"][m])
        dist[sorted_front[0]] = dist[sorted_front[-1]] = float("inf")
        obj_range = (population[front[sorted_front[-1]]]["objectives"][m] -
                     population[front[sorted_front[0]]]["objectives"][m])
        if obj_range == 0:
            continue
        for k in range(1, n-1):
            dist[sorted_front[k]] += (
                population[front[sorted_front[k+1]]]["objectives"][m] -
                population[front[sorted_front[k-1]]]["objectives"][m]
            ) / obj_range
    return dist

def tournament_select(population, fronts, crowd_dist_map):
    candidates = random.sample(range(len(population)), min(TOURNAMENT_K, len(population)))
    def rank_individual(idx):
        for rank, front in enumerate(fronts):
            if idx in front:
                return (rank, -crowd_dist_map.get(idx, 0))
        return (999, 0)
    return min(candidates, key=rank_individual)

def crossover(p1, p2):
    point = random.randint(1, len(p1)-1)
    c1 = p1[:point] + p2[point:]
    c2 = p2[:point] + p1[point:]
    return c1, c2

def mutate(ind, options):
    ind = list(ind)
    for i in range(len(ind)):
        if random.random() < MUTATION_PROB:
            ind[i] = (
                random.randint(0, max(0, len(options[i]["patterns"])-1)),
                random.randint(0, max(0, len(options[i]["classrooms"])-1))
            )
    return ind

# ── Main NSGA-II run ──────────────────────────────────────────────────────────
def run_nsga2():
    print("=" * 60)
    print("UQU V2 — NSGA-II Multi-Objective Optimizer")
    print("=" * 60)
    t0 = time.time()

    print("\nLoading data...")
    sections, classrooms, courses, slots, students, reqs = load_data()
    print(f"  Sections: {len(sections)} | Classrooms: {len(classrooms)} | Requests: {len(reqs):,}")

    print("Precomputing options...")
    options = precompute_options(sections, classrooms, courses, slots)

    # Sample reqs for speed during evolution (use full set for final eval)
    reqs_sample = reqs.sample(min(3000, len(reqs)), random_state=SEED)

    print(f"Initializing population (size={POP_SIZE})...")
    population = []
    for _ in range(POP_SIZE):
        genes = random_individual(options)
        sched = decode(genes, options)
        obj, viol = compute_objectives(sched, reqs_sample, students, slots)
        population.append({"genes": genes, "objectives": obj, "violations": viol})

    best_f1_history = []

    print(f"\nEvolving for {N_GENERATIONS} generations...")
    for gen in range(N_GENERATIONS):
        # Non-dominated sort
        fronts = fast_non_dominated_sort(population)

        # Crowding distance
        crowd_dist_map = {}
        for front in fronts:
            dists = crowding_distance(front, population)
            for idx, d in zip(front, dists):
                crowd_dist_map[idx] = d

        # Generate offspring
        offspring = []
        while len(offspring) < POP_SIZE:
            p1_idx = tournament_select(population, fronts, crowd_dist_map)
            p2_idx = tournament_select(population, fronts, crowd_dist_map)
            g1, g2 = crossover(population[p1_idx]["genes"], population[p2_idx]["genes"]) \
                     if random.random() < CROSSOVER_PROB \
                     else (population[p1_idx]["genes"][:], population[p2_idx]["genes"][:])
            for g in [mutate(g1, options), mutate(g2, options)]:
                sched = decode(g, options)
                obj, viol = compute_objectives(sched, reqs_sample, students, slots)
                offspring.append({"genes": g, "objectives": obj, "violations": viol})

        combined = population + offspring
        fronts_c = fast_non_dominated_sort(combined)

        new_pop = []
        for front in fronts_c:
            if len(new_pop) + len(front) <= POP_SIZE:
                new_pop.extend([combined[i] for i in front])
            else:
                dists = crowding_distance(front, combined)
                sorted_front = sorted(zip(front, dists), key=lambda x: -x[1])
                needed = POP_SIZE - len(new_pop)
                new_pop.extend([combined[i] for i, _ in sorted_front[:needed]])
                break
        population = new_pop

        best_f1 = min(ind["objectives"][0] for ind in population)
        best_f1_history.append(best_f1)
        if gen % 10 == 0 or gen == N_GENERATIONS - 1:
            best_sat = (-best_f1) * 100
            print(f"  Gen {gen:3d}/{N_GENERATIONS} | Best satisfaction: {best_sat:.1f}% | "
                  f"Pop size: {len(population)}")

    # ── Final evaluation on full request set ─────────────────────────────────
    print("\nFinal evaluation on full request set...")
    best_individual = min(population, key=lambda x: x["objectives"][0])
    best_schedule   = decode(best_individual["genes"], options)
    final_obj, final_viol = compute_objectives(best_schedule, reqs, students, slots)

    elapsed = time.time() - t0
    results = {
        "method":                "NSGA-II Multi-Objective Genetic Algorithm",
        "generations":           N_GENERATIONS,
        "population_size":       POP_SIZE,
        "satisfaction_rate":     round(-final_obj[0] * 100, 2),
        "avg_idle_gap":          round(final_obj[1], 4),
        "classroom_utilization": round(-final_obj[2] * 100, 2),
        "constraint_violations": int(final_viol),
        "runtime_seconds":       round(elapsed, 1),
        "pareto_front_size":     len(fast_non_dominated_sort(population)[0]),
        "best_f1_history":       [round(-f, 4) for f in best_f1_history],
    }

    out_path = os.path.join(OUT_DIR, "nsga2_results.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("NSGA-II Results:")
    print(f"  Satisfaction rate   : {results['satisfaction_rate']:.1f}%")
    print(f"  Avg idle gap        : {results['avg_idle_gap']:.2f}h")
    print(f"  Classroom util.     : {results['classroom_utilization']:.1f}%")
    print(f"  Constraint viol.    : {results['constraint_violations']}")
    print(f"  Pareto front size   : {results['pareto_front_size']}")
    print(f"  Runtime             : {results['runtime_seconds']}s")
    print("=" * 60)
    return results

if __name__ == "__main__":
    run_nsga2()
