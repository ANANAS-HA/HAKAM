"""F5 — Pairwise 2×2 outcome between 3-epoch student and 27B teacher.

Heatmap-style 2×2 grid (rows = student, columns = teacher; correct/wrong each).
Diagonals (agreement) green-tinted, off-diagonals (disagreement) orange-tinted.
Cells annotated with absolute count and % of n=1000. Net student-vs-teacher
advantage (+2 items) called out below the grid.

Source: predictions_test_trained_3ep.jsonl + predictions_test_teacher.jsonl
joined on qa_id. Target report section: §4.8, replaces Table 4.7.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _metric import is_correct, load_jsonl
from _style import COLORS, FIGSIZE_SINGLE, apply_style


ROOT = Path(__file__).parent.parent
STUDENT_FILE = ROOT / "predictions_test_trained_3ep.jsonl"
TEACHER_FILE = ROOT / "predictions_test_teacher.jsonl"
OUT = Path(__file__).parent / "F5_pairwise.png"


def main() -> None:
    apply_style()

    student = {r["qa_id"]: r for r in load_jsonl(STUDENT_FILE)}
    teacher = {r["qa_id"]: r for r in load_jsonl(TEACHER_FILE)}
    shared = sorted(set(student) & set(teacher))
    if len(shared) < 990:
        print(f"FAIL: too few shared qa_ids: {len(shared)}")
        sys.exit(1)

    both_correct = student_only = teacher_only = both_wrong = 0
    for qid in shared:
        sc = is_correct(student[qid]["student_text"], student[qid]["gold"])
        tc = is_correct(teacher[qid]["student_text"], teacher[qid]["gold"])
        if   sc and tc:           both_correct += 1
        elif sc and not tc:       student_only += 1
        elif not sc and tc:       teacher_only += 1
        else:                     both_wrong += 1

    n = len(shared)

    # Drift check (report values: 333 / 66 / 64 / 537)
    expected = (333, 66, 64, 537)
    actual = (both_correct, student_only, teacher_only, both_wrong)
    if any(abs(a - e) > 2 for a, e in zip(actual, expected)):
        print(f"FAIL: pairwise counts drifted. expected={expected} got={actual}")
        sys.exit(1)

    # Build the 2x2 grid (rows = student, cols = teacher)
    grid = np.array([[both_correct, student_only],
                     [teacher_only, both_wrong]])
    # Color matrix: green for diagonal (agreement), orange for off-diagonal
    colors = np.array([[COLORS["delta_pos"], COLORS["teacher"]],
                       [COLORS["teacher"],   COLORS["delta_pos"]]])

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    # Manual cell rendering
    for i in range(2):
        for j in range(2):
            count = grid[i, j]
            share = count / n
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1,
                                       facecolor=colors[i, j],
                                       edgecolor="black", lw=1, alpha=0.40))
            label_count = f"{count}"
            label_share = f"{share*100:.1f}%"
            descriptors = [
                ["both correct",          "student right,\nteacher wrong"],
                ["student wrong,\nteacher right", "both wrong"],
            ]
            ax.text(j + 0.5, 1 - i + 0.65, descriptors[i][j],
                    ha="center", va="center", fontsize=9.5, fontstyle="italic")
            ax.text(j + 0.5, 1 - i + 0.35, label_count,
                    ha="center", va="center", fontsize=14, fontweight="bold")
            ax.text(j + 0.5, 1 - i + 0.18, label_share,
                    ha="center", va="center", fontsize=10, color="black", alpha=0.75)

    # Axis labels: student rows, teacher columns
    ax.set_xlim(-0.05, 2.05)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Teacher correct", "Teacher wrong"], fontsize=10)
    ax.set_yticks([1.5, 0.5])
    ax.set_yticklabels(["Student\ncorrect", "Student\nwrong"], fontsize=10)
    ax.tick_params(length=0)
    ax.set_aspect("equal")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Net advantage callout below the grid
    net = student_only - teacher_only
    sign = "+" if net >= 0 else ""
    ax.text(1, -0.30,
            f"Student net advantage: {sign}{net} items "
            f"({student_only} student-only correct − {teacher_only} teacher-only correct)",
            ha="center", fontsize=10, color=COLORS["highlight"], fontweight="bold")

    ax.set_title(f"Pairwise outcomes: 3-epoch student vs 27B teacher (n = {n})",
                 fontsize=12, pad=14)

    plt.tight_layout()
    plt.savefig(OUT)
    plt.close()

    print(f"F5: both_correct={both_correct}, student_only={student_only}, "
          f"teacher_only={teacher_only}, both_wrong={both_wrong}, "
          f"net=+{net}  [matches report: 333/66/64/537 ±2]")


if __name__ == "__main__":
    main()
