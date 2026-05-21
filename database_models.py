"""
database_models.py
UQU Smart Scheduling V2 — SQLAlchemy ORM Models + PostgreSQL Schema
Replaces CSV flat files with a relational database.

Tables mirror the 8 CSV files but add:
  - Foreign key constraints
  - Indexes for fast querying
  - Audit timestamps (created_at, updated_at)
  - Allocation and schedule tables
"""

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean,
    DateTime, ForeignKey, Text, Index, CheckConstraint
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func
from datetime import datetime
import os, pandas as pd

Base = declarative_base()

# ── Models ────────────────────────────────────────────────────────────────────

class Student(Base):
    __tablename__ = "students"
    student_id          = Column(String(20), primary_key=True)
    name                = Column(String(120), nullable=False)
    gender              = Column(String(1),   nullable=False)
    program             = Column(String(10),  nullable=False)
    academic_level      = Column(Integer,     nullable=False)
    gpa                 = Column(Float,       nullable=False)
    credit_hours_completed = Column(Integer, default=0)
    failed_courses_count   = Column(Integer, default=0)
    priority_weight     = Column(Float,       nullable=False)
    is_graduating       = Column(Boolean,     default=False)
    enrollment_year     = Column(Integer)
    expected_graduation = Column(Integer)
    email               = Column(String(100))
    phone               = Column(String(20))
    created_at          = Column(DateTime, server_default=func.now())

    requests     = relationship("RegistrationRequest", back_populates="student")
    allocations  = relationship("Allocation",          back_populates="student")

    __table_args__ = (
        CheckConstraint("academic_level BETWEEN 1 AND 8", name="chk_level"),
        CheckConstraint("gpa BETWEEN 0.0 AND 5.0",        name="chk_gpa"),
        CheckConstraint("gender IN ('M','F')",             name="chk_gender"),
        Index("ix_students_program",  "program"),
        Index("ix_students_level",    "academic_level"),
        Index("ix_students_priority", "priority_weight"),
    )


class Course(Base):
    __tablename__ = "courses"
    course_id       = Column(String(10), primary_key=True)
    course_code     = Column(String(20), unique=True, nullable=False)
    course_name     = Column(String(150), nullable=False)
    credit_hours    = Column(Integer, nullable=False)
    level           = Column(Integer, nullable=False)
    category        = Column(String(30))
    programs        = Column(Text)    # comma-separated
    is_lab          = Column(Boolean, default=False)
    has_prerequisite= Column(Boolean, default=False)

    sections    = relationship("Section",      back_populates="course")
    prereq_for  = relationship("Prerequisite", foreign_keys="Prerequisite.course_id",
                               back_populates="course")

    __table_args__ = (
        Index("ix_courses_level", "level"),
        Index("ix_courses_code",  "course_code"),
    )


class Prerequisite(Base):
    __tablename__ = "prerequisites"
    prereq_id       = Column(String(10), primary_key=True)
    course_id       = Column(String(10), ForeignKey("courses.course_id"), nullable=False)
    prerequisite_id = Column(String(10), ForeignKey("courses.course_id"), nullable=False)

    course      = relationship("Course", foreign_keys=[course_id], back_populates="prereq_for")
    prereq_course = relationship("Course", foreign_keys=[prerequisite_id])


class Instructor(Base):
    __tablename__ = "instructors"
    instructor_id      = Column(String(10), primary_key=True)
    name               = Column(String(120), nullable=False)
    title              = Column(String(30))
    department         = Column(String(10))
    email              = Column(String(100))
    gender             = Column(String(1))
    max_teaching_load  = Column(Integer, default=12)
    years_experience   = Column(Integer, default=1)
    office             = Column(String(20))

    sections = relationship("Section", back_populates="instructor")


class Classroom(Base):
    __tablename__ = "classrooms"
    classroom_id    = Column(String(10), primary_key=True)
    room_code       = Column(String(20), unique=True)
    building        = Column(String(5),  nullable=False)
    floor           = Column(Integer)
    type            = Column(String(20))
    capacity        = Column(Integer,    nullable=False)
    has_projector   = Column(Boolean,    default=True)
    has_smart_board = Column(Boolean,    default=False)
    has_lab_equipment = Column(Boolean,  default=False)
    gender_section  = Column(String(10))

    __table_args__ = (
        Index("ix_classrooms_building", "building"),
        Index("ix_classrooms_gender",   "gender_section"),
    )


class TimeSlot(Base):
    __tablename__ = "time_slots"
    slot_id     = Column(String(10), primary_key=True)
    day         = Column(String(15), nullable=False)
    start_time  = Column(String(8),  nullable=False)
    end_time    = Column(String(8))
    duration    = Column(Integer, default=50)

    __table_args__ = (
        Index("ix_slots_day", "day"),
    )


class Section(Base):
    __tablename__ = "sections"
    section_id          = Column(String(10), primary_key=True)
    course_id           = Column(String(10), ForeignKey("courses.course_id"))
    course_code         = Column(String(20))
    section_number      = Column(Integer)
    instructor_id       = Column(String(10), ForeignKey("instructors.instructor_id"))
    preferred_classroom = Column(String(10))
    capacity            = Column(Integer, default=50)
    gender_section      = Column(String(10))
    enrolled_count      = Column(Integer, default=0)
    # Scheduling result columns (populated by optimizer)
    scheduled_classroom = Column(String(10), ForeignKey("classrooms.classroom_id"), nullable=True)
    slot_ids            = Column(Text, nullable=True)   # pipe-separated
    days                = Column(Text, nullable=True)
    start_times         = Column(Text, nullable=True)
    is_scheduled        = Column(Boolean, default=False)
    schedule_method     = Column(String(20))    # "greedy" or "nsga2"
    scheduled_at        = Column(DateTime)

    course      = relationship("Course",     back_populates="sections")
    instructor  = relationship("Instructor", back_populates="sections")
    allocations = relationship("Allocation", back_populates="section")

    __table_args__ = (
        Index("ix_sections_course",   "course_id"),
        Index("ix_sections_instructor","instructor_id"),
        Index("ix_sections_gender",   "gender_section"),
    )


class RegistrationRequest(Base):
    __tablename__ = "registration_requests"
    request_id          = Column(String(15), primary_key=True)
    student_id          = Column(String(20), ForeignKey("students.student_id"))
    course_id           = Column(String(10), ForeignKey("courses.course_id"))
    preferred_sections  = Column(Text)
    priority_weight     = Column(Float)
    is_graduating       = Column(Boolean, default=False)
    gender              = Column(String(1))
    request_timestamp   = Column(String(30))
    status              = Column(String(20), default="pending")  # pending/allocated/waitlisted
    created_at          = Column(DateTime, server_default=func.now())

    student = relationship("Student", back_populates="requests")

    __table_args__ = (
        Index("ix_requests_student",  "student_id"),
        Index("ix_requests_course",   "course_id"),
        Index("ix_requests_priority", "priority_weight"),
        Index("ix_requests_status",   "status"),
    )


class Allocation(Base):
    __tablename__ = "allocations"
    allocation_id    = Column(Integer, primary_key=True, autoincrement=True)
    request_id       = Column(String(15), unique=True)
    student_id       = Column(String(20), ForeignKey("students.student_id"))
    course_id        = Column(String(10), ForeignKey("courses.course_id"))
    section_id       = Column(String(10), ForeignKey("sections.section_id"))
    priority_weight  = Column(Float)
    is_graduating    = Column(Boolean)
    status           = Column(String(20), default="allocated")
    allocated_at     = Column(DateTime, server_default=func.now())

    student = relationship("Student",  back_populates="allocations")
    section = relationship("Section",  back_populates="allocations")

    __table_args__ = (
        Index("ix_alloc_student",  "student_id"),
        Index("ix_alloc_section",  "section_id"),
        Index("ix_alloc_status",   "status"),
    )


class KPISnapshot(Base):
    """Stores KPI snapshots per semester for trend analysis."""
    __tablename__ = "kpi_snapshots"
    snapshot_id              = Column(Integer, primary_key=True, autoincrement=True)
    semester                 = Column(String(20))
    algorithm_used           = Column(String(20))
    grad_satisfaction        = Column(Float)
    nongrad_satisfaction     = Column(Float)
    overall_satisfaction     = Column(Float)
    classroom_utilization    = Column(Float)
    avg_idle_gap             = Column(Float)
    time_conflicts           = Column(Integer)
    total_allocations        = Column(Integer)
    total_waitlisted         = Column(Integer)
    runtime_seconds          = Column(Float)
    created_at               = Column(DateTime, server_default=func.now())


# ── Database setup functions ───────────────────────────────────────────────────

def get_engine(db_url: str = None):
    """Get SQLAlchemy engine. Defaults to SQLite for demo."""
    if db_url is None:
        db_path = os.path.join(os.path.dirname(__file__), "..", "uqu_scheduling.db")
        db_url = f"sqlite:///{db_path}"
    return create_engine(db_url, echo=False)


def create_all_tables(engine):
    Base.metadata.create_all(engine)
    print("  All tables created successfully.")


def load_csv_to_db(engine, data_dir: str):
    """Load all CSV files into the database."""
    Session = sessionmaker(bind=engine)
    session = Session()

    table_map = {
        "time_slots":             TimeSlot,
        "classrooms":             Classroom,
        "instructors":            Instructor,
        "courses":                Course,
        "prerequisites":          Prerequisite,
        "sections":               Section,
        "students":               Student,
        "registration_requests":  RegistrationRequest,
    }

    for name, Model in table_map.items():
        path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(path):
            print(f"    ⚠ {name}.csv not found, skipping.")
            continue
        df = pd.read_csv(path)
        df = df.where(pd.notnull(df), None)
        existing = session.query(Model).count()
        if existing > 0:
            print(f"    ↩ {name}: {existing} rows already exist, skipping.")
            continue
        records = df.to_dict("records")
        valid_cols = {c.key for c in Model.__table__.columns}
        clean_records = [{k: v for k, v in r.items() if k in valid_cols} for r in records]
        session.bulk_insert_mappings(Model, clean_records)
        session.commit()
        print(f"    ✓ {name}: {len(clean_records):,} rows loaded")

    session.close()


def setup_database(data_dir: str, db_url: str = None):
    """Full database setup: create tables + load data."""
    print("Setting up database...")
    engine = get_engine(db_url)
    create_all_tables(engine)
    load_csv_to_db(engine, data_dir)
    print("  Database ready.")
    return engine


if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uqu_project", "data")
    engine = setup_database(DATA_DIR)

    # Quick verification
    Session = sessionmaker(bind=engine)
    session = Session()
    n_students = session.query(Student).count()
    n_sections = session.query(Section).count()
    n_reqs     = session.query(RegistrationRequest).count()
    print(f"\nVerification:")
    print(f"  Students  : {n_students:,}")
    print(f"  Sections  : {n_sections:,}")
    print(f"  Requests  : {n_reqs:,}")
    session.close()
