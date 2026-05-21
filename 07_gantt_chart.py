"""
07_gantt_chart.py
UQU Smart Scheduling System – Semester 2 Gantt Chart
Project: UQU-DS-2025-M09

Produces: figures/gantt_chart_semester2.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Colour scheme ─────────────────────────────────────────────────────────────
COLORS = {
    "Data & Planning":     "#1B5E20",
    "Algorithm Dev.":      "#1565C0",
    "ML Sub-Systems":      "#F9A825",
    "Dashboard & Report":  "#6A1B9A",
    "Testing & QA":        "#E53935",
    "Presentation Prep":   "#00838F",
}

# ── Task definition ────────────────────────────────────────────────────────────
# (task_name, category, start_week, duration_weeks, milestone)
TASKS = [
    # Week 1-2: Kick-off
    ("Project kick-off & scope review",        "Data & Planning",    1,  1, False),
    ("Semester-1 report review",               "Data & Planning",    1,  1, False),
    ("Dataset design & schema finalisation",   "Data & Planning",    2,  2, False),
    ("Synthetic data generation (Script 01)",  "Data & Planning",    2,  2, False),

    # Week 3-5: Algorithm
    ("Phase 1: Section scheduling engine",     "Algorithm Dev.",     4,  3, False),
    ("Phase 2: Priority allocation engine",    "Algorithm Dev.",     5,  3, False),
    ("Constraint validation & testing",        "Algorithm Dev.",     7,  2, False),
    ("KPI evaluation module (Script 03)",      "Algorithm Dev.",     8,  1, False),

    # Week 5-9: ML
    ("CF course recommender",                  "ML Sub-Systems",     5,  2, False),
    ("Demand forecasting model",               "ML Sub-Systems",     6,  2, False),
    ("Random Forest classifier (94.7%)",       "ML Sub-Systems",     7,  3, False),
    ("ML integration & output generation",     "ML Sub-Systems",     9,  1, False),

    # Week 9-12: Dashboard & Report
    ("Interactive HTML dashboard (Script 04)", "Dashboard & Report", 9,  2, False),
    ("Gantt chart & figures (Script 07)",      "Dashboard & Report", 10, 1, False),
    ("Word report authoring (Script 06)",      "Dashboard & Report", 10, 2, False),
    ("Final report review & formatting",       "Dashboard & Report", 12, 1, False),

    # Continuous: Testing
    ("Unit & integration tests",               "Testing & QA",       3,  10, False),
    ("Zero-conflict verification",             "Testing & QA",       8,  1,  True),
    ("End-to-end pipeline validation",         "Testing & QA",       11, 1,  False),

    # Week 12-14: Presentation
    ("Presentation slide preparation",         "Presentation Prep",  12, 2,  False),
    ("Demo environment setup",                 "Presentation Prep",  13, 1,  False),
    ("Final submission",                       "Presentation Prep",  14, 1,  True),
]

N_WEEKS = 14


def draw_gantt():
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # Grid lines at each week
    for w in range(1, N_WEEKS + 2):
        ax.axvline(w, color="#E0E0E0", linewidth=0.7, zorder=0)

    # Alternating row shading
    for i in range(len(TASKS)):
        if i % 2 == 0:
            ax.axhspan(i - 0.45, i + 0.45, color="#F5F5F5", zorder=0)

    # Draw bars
    for i, (task, category, start, duration, is_milestone) in enumerate(reversed(TASKS)):
        idx = len(TASKS) - 1 - i  # y position
        color = COLORS.get(category, "#757575")

        if is_milestone:
            # Diamond marker at end
            ax.plot(start + duration, idx, "D", color=color, markersize=10, zorder=4,
                    markeredgecolor="white", markeredgewidth=1.5)
            ax.plot([start, start + duration - 0.1], [idx, idx],
                    color=color, linewidth=5, solid_capstyle="round", alpha=0.7, zorder=3)
        else:
            ax.barh(
                idx, duration, left=start, height=0.55,
                color=color, edgecolor="white", linewidth=0.5,
                align="center", zorder=3, alpha=0.88,
            )
            # Week count label inside bar
            if duration >= 2:
                ax.text(
                    start + duration / 2, idx,
                    f"{duration}w",
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold", zorder=5,
                )

    # Y-axis labels
    task_names = [t[0] for t in reversed(TASKS)]
    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels(task_names, fontsize=8.5)

    # X-axis
    ax.set_xlim(1, N_WEEKS + 1)
    ax.set_xticks(range(1, N_WEEKS + 2))
    ax.set_xticklabels([f"W{w}" for w in range(1, N_WEEKS + 2)], fontsize=8)
    ax.set_xlabel("Week of Semester 2 (Spring 2026)", fontsize=10)

    # Title
    ax.set_title(
        "UQU-DS-2025-M09 – Semester 2 Project Timeline (Gantt Chart)\n"
        "Umm Al-Qura University · College of Computing · Data Science Department",
        fontsize=12, fontweight="bold", color="#1B5E20", pad=12,
    )

    # Legend
    legend_patches = [
        mpatches.Patch(color=c, label=cat) for cat, c in COLORS.items()
    ]
    legend_patches.append(
        mpatches.Patch(color="#757575", label="◆ Milestone")
    )
    ax.legend(
        handles=legend_patches, loc="lower right",
        fontsize=8.5, framealpha=0.9, edgecolor="#BDBDBD",
        bbox_to_anchor=(1.0, 0.0),
    )

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BDBDBD")
    ax.spines["bottom"].set_color("#BDBDBD")

    # Phase annotations
    phase_boxes = [
        (1,  3,  "Phase 0:\nData",     "#1B5E20"),
        (4,  8,  "Phase 1:\nAlgorithm","#1565C0"),
        (5,  9,  "Phase 2:\nML",       "#F9A825"),
        (9,  12, "Phase 3:\nDashboard","#6A1B9A"),
        (12, 14, "Phase 4:\nPresent.", "#00838F"),
    ]
    top_y = len(TASKS) - 0.5
    for s, e, label, col in phase_boxes:
        ax.annotate(
            "", xy=(e + 1, top_y + 0.9), xytext=(s, top_y + 0.9),
            arrowprops=dict(arrowstyle="<->", color=col, lw=1.5),
        )
        ax.text((s + e + 1) / 2, top_y + 1.1, label, ha="center",
                fontsize=7, color=col, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(FIG_DIR, "gantt_chart_semester2.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Gantt chart saved → {path}")
    return path


def main():
    print("=" * 60)
    print("UQU Smart Scheduling – Semester 2 Gantt Chart")
    print("=" * 60)
    draw_gantt()
    print("\nDone.")


if __name__ == "__main__":
    main()
