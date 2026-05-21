"""
01_generate_data.py
UQU Smart Scheduling System – Synthetic Dataset Generator
Project: UQU-DS-2025-M09
Institution: Umm Al-Qura University · College of Computing · Data Science Department
"""

import random
import json
import os
from datetime import datetime, timedelta
import pandas as pd
from faker import Faker

# ── Configuration ──────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

fake = Faker("ar_SA")
fake.seed_instance(SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
PROGRAMS = ["CS", "DS", "CYS", "AI", "IS", "IT"]
LEVELS = list(range(1, 9))
DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]

PRIORITY_WEIGHTS = {8: 5.0, 7: 4.0, 6: 3.0, 5: 2.5, 4: 2.0, 3: 1.5, 2: 1.2, 1: 1.0}

BUILDINGS = {
    "A1": {"floors": 3, "gender": "Male"},
    "A2": {"floors": 3, "gender": "Male"},
    "B1": {"floors": 4, "gender": "Female"},
    "B2": {"floors": 4, "gender": "Female"},
    "C1": {"floors": 3, "gender": "Male"},
    "C2": {"floors": 3, "gender": "Female"},
    "D1": {"floors": 2, "gender": "Mixed"},
}

ROOM_TYPES = {
    "Lecture Hall": {"capacity_range": (60, 120), "has_lab": False},
    "Classroom": {"capacity_range": (30, 60), "has_lab": False},
    "Lab": {"capacity_range": (20, 40), "has_lab": True},
    "Seminar Room": {"capacity_range": (15, 30), "has_lab": False},
}

DEPARTMENTS = ["CS", "DS", "CYS", "AI", "IS", "IT", "Math", "English", "Islamic"]
TITLES = ["Prof.", "Assoc. Prof.", "Asst. Prof.", "Dr.", "TA"]

N_STUDENTS = 5000
N_INSTRUCTORS = 80
N_CLASSROOMS = 92
N_COURSES = 76
N_SECTIONS_TARGET = 390


# ── Helpers ───────────────────────────────────────────────────────────────────
def arabic_name(gender="M"):
    male_names = [
        "محمد", "عبدالله", "عمر", "أحمد", "خالد", "عبدالرحمن", "سلطان",
        "فيصل", "إبراهيم", "يوسف", "علي", "حسن", "أنس", "راشد", "نايف",
        "بندر", "سعود", "منصور", "تركي", "وليد", "عادل", "ماجد", "طارق",
        "زياد", "رائد", "ضياء", "جلاء", "معاذ", "رؤوف", "عمار",
    ]
    female_names = [
        "نورة", "سارة", "ريم", "هند", "لجين", "دانة", "رهف", "جواهر",
        "أريج", "أميرة", "هيا", "لمى", "شهد", "بيان", "زينب", "ميرة",
        "غلا", "جلا", "رؤوف", "هناء", "حلا", "جالا", "الدكتورة جالا",
        "الأستاذة حلا",
    ]
    family_names = [
        "القحطاني", "الشمري", "الغامدي", "الزهراني", "العتيبي", "الحربي",
        "الدوسري", "المطيري", "السبيعي", "الرشيدي", "البقمي", "العمري",
        "العسيري", "الجهني", "الصاعدي", "المالكي", "الأسمري", "السلمي",
        "آل سعود", "آل معيض", "آل قصير", "آل سرحاني", "الهلالي", "الحواساوي",
        "الخرافي", "الدباغ", "كانو",
    ]
    first = random.choice(male_names if gender == "M" else female_names)
    last = random.choice(family_names)
    return f"{first} {last}"


def save_csv(df, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  ✓ {name}.csv  ({len(df):,} rows)")


# ── 1. Time Slots ─────────────────────────────────────────────────────────────
def generate_time_slots():
    slots = []
    start_times = [
        "08:00", "09:00", "10:00", "11:00", "12:00",
        "13:00", "14:00", "15:00", "16:00",
    ]
    slot_id = 1
    for day in DAYS:
        for st in start_times:
            h, m = map(int, st.split(":"))
            end_h = h + 1 if m < 10 else h + 1
            end_time = f"{end_h:02d}:{m + 50 - 60 if m >= 10 else m + 50:02d}"
            end_time = f"{h:02d}:{m+50:02d}" if m + 50 < 60 else f"{h+1:02d}:{m+50-60:02d}"
            slots.append({
                "slot_id": f"TS{slot_id:04d}",
                "day": day,
                "start_time": st,
                "end_time": f"{h:02d}:50",
                "duration": 50,
            })
            slot_id += 1
    return pd.DataFrame(slots)


# ── 2. Classrooms ─────────────────────────────────────────────────────────────
def generate_classrooms():
    rooms = []
    room_id = 1
    room_num = 101
    for building, info in BUILDINGS.items():
        n_rooms = N_CLASSROOMS // len(BUILDINGS)
        for _ in range(n_rooms):
            rtype = random.choices(
                list(ROOM_TYPES.keys()), weights=[3, 5, 3, 1]
            )[0]
            rinfo = ROOM_TYPES[rtype]
            cap = random.randint(*rinfo["capacity_range"])
            floor = random.randint(1, info["floors"])
            gender_sec = info["gender"] if info["gender"] != "Mixed" else random.choice(["Male", "Female"])
            rooms.append({
                "classroom_id": f"CR{room_id:04d}",
                "room_code": f"{building}-{room_num}",
                "building": building,
                "floor": floor,
                "type": rtype,
                "capacity": cap,
                "has_projector": random.random() > 0.2,
                "has_smart_board": random.random() > 0.5,
                "has_lab_equipment": rinfo["has_lab"],
                "gender_section": gender_sec,
            })
            room_id += 1
            room_num += 1
    # Top up to N_CLASSROOMS
    while len(rooms) < N_CLASSROOMS:
        bld = random.choice(list(BUILDINGS.keys()))
        rtype = "Classroom"
        rooms.append({
            "classroom_id": f"CR{room_id:04d}",
            "room_code": f"{bld}-{room_num}",
            "building": bld,
            "floor": 1,
            "type": rtype,
            "capacity": 50,
            "has_projector": True,
            "has_smart_board": False,
            "has_lab_equipment": False,
            "gender_section": "Male",
        })
        room_id += 1
        room_num += 1
    return pd.DataFrame(rooms[:N_CLASSROOMS])


# ── 3. Instructors ────────────────────────────────────────────────────────────
def generate_instructors():
    instructors = []
    for i in range(1, N_INSTRUCTORS + 1):
        gender = random.choice(["M", "F"])
        name = arabic_name(gender)
        dept = random.choice(DEPARTMENTS)
        title = random.choices(TITLES, weights=[1, 2, 4, 3, 3])[0]
        exp = random.randint(1, 35)
        load = random.randint(9, 18)
        office_bld = random.choice(["A1", "A2", "C1"])
        office_room = random.randint(100, 299)
        instructors.append({
            "instructor_id": f"INST{i:04d}",
            "name": name,
            "title": title,
            "department": dept,
            "email": f"instructor{i}@uqu.edu.sa",
            "gender": gender,
            "max_teaching_load": load,
            "years_experience": exp,
            "office": f"{office_bld}-{office_room}",
        })
    return pd.DataFrame(instructors)


# ── 4. Courses ────────────────────────────────────────────────────────────────
COURSE_TEMPLATES = [
    ("UNV101", "Islamic Culture I", 2, 1, "University", "CS,DS,CYS,AI,IS,IT", False),
    ("UNV102", "Islamic Culture II", 2, 2, "University", "CS,DS,CYS,AI,IS,IT", False),
    ("UNV103", "Arabic Language", 2, 1, "University", "CS,DS,CYS,AI,IS,IT", False),
    ("UNV104", "English Communication", 3, 1, "University", "CS,DS,CYS,AI,IS,IT", False),
    ("MATH101", "Calculus I", 3, 1, "Core", "CS,DS,AI,CYS", False),
    ("MATH102", "Calculus II", 3, 2, "Core", "CS,DS,AI", False),
    ("MATH201", "Linear Algebra", 3, 2, "Core", "CS,DS,AI", False),
    ("MATH202", "Probability & Statistics", 3, 3, "Core", "CS,DS,AI,IS", False),
    ("MATH301", "Numerical Methods", 3, 4, "Elective", "CS,DS", False),
    ("CS101", "Introduction to Programming", 3, 1, "Core", "CS,DS,CYS,AI,IS,IT", False),
    ("CS102", "Data Structures", 3, 2, "Core", "CS,DS,CYS,AI", False),
    ("CS201", "Algorithms", 3, 3, "Core", "CS,DS,AI", False),
    ("CS202", "Database Systems", 3, 3, "Core", "CS,DS,IS,IT", False),
    ("CS203", "Operating Systems", 3, 4, "Core", "CS,CYS", False),
    ("CS204", "Computer Networks", 3, 4, "Core", "CS,CYS,IT", False),
    ("CS301", "Software Engineering", 3, 5, "Core", "CS,IS", False),
    ("CS302", "Computer Architecture", 3, 5, "Core", "CS", False),
    ("CS401", "Graduation Project I", 3, 7, "Core", "CS,DS,CYS,AI,IS,IT", False),
    ("CS402", "Graduation Project II", 3, 8, "Core", "CS,DS,CYS,AI,IS,IT", False),
    ("DS101", "Introduction to Data Science", 3, 1, "Core", "DS", False),
    ("DS201", "Data Wrangling", 3, 2, "Core", "DS", False),
    ("DS202", "Exploratory Data Analysis", 3, 3, "Core", "DS", False),
    ("DS301", "Machine Learning", 3, 4, "Core", "DS,AI,CS", False),
    ("DS302", "Deep Learning", 3, 5, "Core", "DS,AI", False),
    ("DS303", "Natural Language Processing", 3, 5, "Elective", "DS,AI", False),
    ("DS401", "Big Data Analytics", 3, 6, "Core", "DS", False),
    ("DS402", "Data Governance", 3, 6, "Elective", "DS,IS", False),
    ("DS403", "Capstone Project I", 3, 7, "Core", "DS", False),
    ("DS404", "Capstone Project II", 3, 8, "Core", "DS", False),
    ("AI201", "Logic & Knowledge Representation", 3, 3, "Core", "AI", False),
    ("AI301", "Computer Vision", 3, 5, "Core", "AI,DS", False),
    ("AI302", "Reinforcement Learning", 3, 6, "Elective", "AI", False),
    ("CYS201", "Information Security", 3, 3, "Core", "CYS", False),
    ("CYS202", "Cryptography", 3, 4, "Core", "CYS", False),
    ("CYS301", "Ethical Hacking", 3, 5, "Core", "CYS", False),
    ("CYS302", "Digital Forensics", 3, 6, "Core", "CYS", False),
    ("IS201", "Systems Analysis & Design", 3, 3, "Core", "IS", False),
    ("IS202", "ERP Systems", 3, 4, "Core", "IS", False),
    ("IS301", "Business Intelligence", 3, 5, "Core", "IS", False),
    ("IT201", "Network Administration", 3, 3, "Core", "IT", False),
    ("IT202", "Cloud Computing", 3, 4, "Core", "IT,CS", False),
    ("IT301", "IT Project Management", 3, 5, "Core", "IT,IS", False),
    ("LAB101", "Programming Lab", 1, 1, "Lab", "CS,DS,AI,CYS,IS,IT", True),
    ("LAB201", "Database Lab", 1, 3, "Lab", "CS,DS,IS,IT", True),
    ("LAB301", "Networks Lab", 1, 4, "Lab", "CS,CYS,IT", True),
    ("LAB401", "AI Lab", 1, 5, "Lab", "AI,DS,CS", True),
]

# Fill to N_COURSES
EXTRA_COURSES = [
    (f"ELEC{100+i}", f"Elective Course {i}", 3, random.randint(3, 7), "Elective",
     random.choice(["CS,DS", "AI,DS", "CYS", "IS,IT", "CS"]), False)
    for i in range(N_COURSES - len(COURSE_TEMPLATES))
]


def generate_courses():
    all_templates = COURSE_TEMPLATES + EXTRA_COURSES
    courses = []
    for i, t in enumerate(all_templates[:N_COURSES]):
        code, name, credits, level, cat, programs, is_lab = t
        courses.append({
            "course_id": f"C{i+1:04d}",
            "course_code": code,
            "course_name": name,
            "credit_hours": credits,
            "level": level,
            "category": cat,
            "programs": programs,
            "is_lab": is_lab,
            "requires_lab": is_lab,
            "has_prerequisite": False,  # updated later
        })
    return pd.DataFrame(courses)


# ── 5. Prerequisites ──────────────────────────────────────────────────────────
PREREQ_PAIRS = [
    ("CS102", "CS101"), ("CS201", "CS102"), ("CS202", "CS101"),
    ("CS203", "CS102"), ("CS204", "CS201"), ("CS301", "CS202"),
    ("CS302", "CS203"), ("CS401", "CS301"), ("CS402", "CS401"),
    ("MATH102", "MATH101"), ("MATH201", "MATH101"), ("MATH202", "MATH101"),
    ("MATH301", "MATH202"), ("DS201", "DS101"), ("DS202", "DS201"),
    ("DS301", "MATH202"), ("DS302", "DS301"), ("DS303", "DS301"),
    ("DS401", "DS302"), ("DS402", "DS301"), ("DS403", "DS401"),
    ("DS404", "DS403"), ("AI201", "CS102"), ("AI301", "DS301"),
    ("AI302", "AI301"), ("CYS202", "CYS201"), ("CYS301", "CYS202"),
    ("CYS302", "CYS301"), ("IS202", "IS201"), ("IS301", "IS202"),
    ("IT202", "IT201"), ("IT301", "IT202"),
    ("UNV102", "UNV101"), ("LAB201", "LAB101"), ("LAB301", "CS204"),
    ("LAB401", "DS301"),
]


def generate_prerequisites(courses_df):
    code_to_id = dict(zip(courses_df["course_code"], courses_df["course_id"]))
    prereqs = []
    pid = 1
    for course_code, prereq_code in PREREQ_PAIRS:
        if course_code in code_to_id and prereq_code in code_to_id:
            prereqs.append({
                "prereq_id": f"PR{pid:04d}",
                "course_id": code_to_id[course_code],
                "prerequisite_id": code_to_id[prereq_code],
            })
            pid += 1
    # Update has_prerequisite
    course_ids_with_prereq = set(p["course_id"] for p in prereqs)
    courses_df["has_prerequisite"] = courses_df["course_id"].isin(course_ids_with_prereq)
    return pd.DataFrame(prereqs), courses_df


# ── 6. Sections ───────────────────────────────────────────────────────────────
def generate_sections(courses_df, instructors_df, classrooms_df):
    sections = []
    sec_id = 1
    instructor_load = {iid: 0 for iid in instructors_df["instructor_id"]}
    male_instructors = list(instructors_df[instructors_df["gender"] == "M"]["instructor_id"])
    female_instructors = list(instructors_df[instructors_df["gender"] == "F"]["instructor_id"])
    male_classrooms = list(classrooms_df[classrooms_df["gender_section"] == "Male"]["classroom_id"])
    female_classrooms = list(classrooms_df[classrooms_df["gender_section"] == "Female"]["classroom_id"])

    for _, course in courses_df.iterrows():
        # Number of sections based on popularity
        n_sec = random.choices([1, 2, 3, 4, 5, 6], weights=[5, 10, 15, 20, 10, 5])[0]
        n_sec = min(n_sec, 6)
        for sec_num in range(1, n_sec + 1):
            gender_sec = random.choice(["Male", "Female"])
            inst_pool = male_instructors if gender_sec == "Male" else female_instructors
            if not inst_pool:
                inst_pool = list(instructors_df["instructor_id"])
            # Pick instructor with lowest load
            inst_pool_sorted = sorted(inst_pool, key=lambda x: instructor_load.get(x, 0))
            instructor_id = inst_pool_sorted[0]
            instructor_load[instructor_id] = instructor_load.get(instructor_id, 0) + course["credit_hours"]

            cr_pool = male_classrooms if gender_sec == "Male" else female_classrooms
            preferred_cr = random.choice(cr_pool) if cr_pool else random.choice(list(classrooms_df["classroom_id"]))

            cap = classrooms_df[classrooms_df["classroom_id"] == preferred_cr]["capacity"].values
            capacity = int(cap[0]) if len(cap) > 0 else 50

            sections.append({
                "section_id": f"SEC{sec_id:05d}",
                "course_id": course["course_id"],
                "course_code": course["course_code"],
                "section_number": sec_num,
                "instructor_id": instructor_id,
                "preferred_classroom": preferred_cr,
                "capacity": capacity,
                "gender_section": gender_sec,
                "enrolled_count": 0,
            })
            sec_id += 1
        if sec_id > N_SECTIONS_TARGET:
            break

    return pd.DataFrame(sections[:N_SECTIONS_TARGET])


# ── 7. Students ───────────────────────────────────────────────────────────────
def generate_students():
    students = []
    level_dist = {1: 800, 2: 750, 3: 700, 4: 650, 5: 600, 6: 550, 7: 500, 8: 450}
    for sid in range(1, N_STUDENTS + 1):
        # Assign level by distribution
        level = random.choices(
            LEVELS,
            weights=[level_dist[l] for l in LEVELS]
        )[0]
        gender = random.choice(["M", "F"])
        program = random.choice(PROGRAMS)
        gpa = round(random.uniform(2.0, 4.0), 2)
        credit_hours = (level - 1) * 15 + random.randint(0, 14)
        failed = random.choices([0, 1, 2, 3], weights=[70, 15, 10, 5])[0]
        is_graduating = level == 8
        weight = PRIORITY_WEIGHTS[level]
        enroll_year = 2026 - (level - 1)
        grad_year = enroll_year + 4
        students.append({
            "student_id": f"44400{sid:04d}",
            "name": arabic_name(gender),
            "gender": gender,
            "program": program,
            "academic_level": level,
            "gpa": gpa,
            "credit_hours_completed": credit_hours,
            "failed_courses_count": failed,
            "priority_weight": weight,
            "is_graduating": is_graduating,
            "enrollment_year": enroll_year,
            "expected_graduation": grad_year,
            "email": f"44400{sid:04d}@uqu.edu.sa",
            "phone": f"05{random.randint(10000000, 99999999)}",
        })
    return pd.DataFrame(students)


# ── 8. Registration Requests ──────────────────────────────────────────────────
def generate_registration_requests(students_df, courses_df, sections_df):
    reqs = []
    req_id = 1
    code_to_sections = {}
    for _, sec in sections_df.iterrows():
        cid = sec["course_id"]
        code_to_sections.setdefault(cid, []).append(sec["section_id"])

    base_date = datetime(2026, 1, 15)
    for _, student in students_df.iterrows():
        # Each student requests 4-8 courses
        n_courses = random.randint(4, 8)
        # Filter eligible courses (level <= student level + 1)
        eligible = courses_df[courses_df["level"] <= student["academic_level"] + 1]["course_id"].tolist()
        if len(eligible) < n_courses:
            eligible = list(courses_df["course_id"])
        chosen_courses = random.sample(eligible, min(n_courses, len(eligible)))
        for cid in chosen_courses:
            secs = code_to_sections.get(cid, [])
            # Filter by gender
            gender_match = sections_df[
                (sections_df["course_id"] == cid) &
                (sections_df["gender_section"].isin([
                    "Male" if student["gender"] == "M" else "Female"
                ]))
            ]["section_id"].tolist()
            if not gender_match:
                gender_match = secs
            preferred = "|".join(random.sample(gender_match, min(3, len(gender_match)))) if gender_match else ""
            ts_offset = random.randint(0, 60)
            ts = base_date + timedelta(days=ts_offset, hours=random.randint(8, 22),
                                       minutes=random.randint(0, 59))
            reqs.append({
                "request_id": f"REQ{req_id:07d}",
                "student_id": student["student_id"],
                "course_id": cid,
                "preferred_sections": preferred,
                "priority_weight": student["priority_weight"],
                "is_graduating": student["is_graduating"],
                "gender": student["gender"],
                "request_timestamp": ts.isoformat(),
            })
            req_id += 1

    random.shuffle(reqs)
    return pd.DataFrame(reqs)


# ── 9. Metadata ───────────────────────────────────────────────────────────────
def generate_metadata(stats):
    meta = {
        "project_id": "UQU-DS-2025-M09",
        "university": "Umm Al-Qura University",
        "college": "College of Computing",
        "campus": "Al-Abidiyah Campus",
        "semester": "Spring 2026",
        "generation_date": datetime.now().isoformat(),
        "data_statistics": stats,
        "priority_system": {str(k): v for k, v in PRIORITY_WEIGHTS.items()},
        "time_slots_per_week": 45,
        "days": DAYS,
        "programs": {
            "CS": "Computer Science",
            "DS": "Data Science",
            "CYS": "Cybersecurity",
            "AI": "Artificial Intelligence",
            "IS": "Information Systems",
            "IT": "Information Technology",
        },
    }
    path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("  ✓ metadata.json")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("UQU Smart Scheduling – Data Generator")
    print("=" * 60)

    print("\n[1/8] Generating time slots...")
    slots_df = generate_time_slots()
    save_csv(slots_df, "time_slots")

    print("\n[2/8] Generating classrooms...")
    classrooms_df = generate_classrooms()
    save_csv(classrooms_df, "classrooms")

    print("\n[3/8] Generating instructors...")
    instructors_df = generate_instructors()
    save_csv(instructors_df, "instructors")

    print("\n[4/8] Generating courses...")
    courses_df = generate_courses()

    print("\n[5/8] Generating prerequisites...")
    prereqs_df, courses_df = generate_prerequisites(courses_df)
    save_csv(courses_df, "courses")
    save_csv(prereqs_df, "prerequisites")

    print("\n[6/8] Generating sections...")
    sections_df = generate_sections(courses_df, instructors_df, classrooms_df)
    save_csv(sections_df, "sections")

    print("\n[7/8] Generating students...")
    students_df = generate_students()
    save_csv(students_df, "students")

    print("\n[8/8] Generating registration requests...")
    reqs_df = generate_registration_requests(students_df, courses_df, sections_df)
    save_csv(reqs_df, "registration_requests")

    stats = {
        "students": len(students_df),
        "instructors": len(instructors_df),
        "classrooms": len(classrooms_df),
        "time_slots": len(slots_df),
        "courses": len(courses_df),
        "prerequisites": len(prereqs_df),
        "sections": len(sections_df),
        "registration_requests": len(reqs_df),
    }
    generate_metadata(stats)

    print("\n" + "=" * 60)
    print("Dataset generation complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    for k, v in stats.items():
        print(f"  {k:<28} {v:>8,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
