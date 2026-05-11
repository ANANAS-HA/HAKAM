"""F4 — Per-question-type accuracy: vanilla / 3-epoch / teacher × 4 qtypes.

Grouped bars per question type (4 groups × 3 bars), ordered by 3-epoch accuracy
descending. The +12 pp Counterfactual win is annotated; the structural floor
on Temporal/Causal is highlighted with a horizontal dashed line.

Source: predictions_test_{vanilla,trained_3ep,teacher}.jsonl.
Target report section: §4.7, replaces Table 4.6.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _metric import get_qtype, is_correct, load_jsonl
from _style import COLORS, FIGSIZE_WIDE, apply_style


ROOT = Path(__file__).parent.parent
PRED_FILES = {
    "vanilla":  ROOT / "predictions_test_vanilla.jsonl",
    "epoch3":   ROOT / "predictions_test_trained_3ep.jsonl",
    "teacher":  ROOT / "predictions_test_teacher.jsonl",
}
OUT = Path(__file__).parent / "F4_per_qtype.png"

# Reference values from the report's Table 4.6.
EXPECTED = {
    "Counterfactual": {"vanilla": 0.467, "epoch3": 0.820, "teacher": 0.700, "n": 150},
    "Descriptive":    {"vanilla": 0.348, "epoch3": 0.619, "teacher": 0.637, "n": 391},
    "Temporal":       {"vanilla": 0.069, "epoch3": 0.108, "teacher": 0.131, "n": 306},
    "Causal":         {"vanilla": 0.000, "epoch3": 0.007, "teacher": 0.020, "n": 153},
}


def main() -> None:
    apply_style()

    by_qtype: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for label, path in PRED_FILES.items():
        for r in load_jsonl(path):
            qt = get_qtype(r)
            ok = is_correct(r["student_text"], r["gold"])
            by_qtype[qt][label].append(ok)

    accuracies: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for qt, by_model in by_qtype.items():
        accuracies[qt] = {m: sum(v) / len(v) for m, v in by_model.items()}
        counts[qt] = max(len(v) for v in by_model.values())

    # Drift check
    for qt, exp in EXPECTED.items():
        got = accuracies.get(qt, {})
        for model in ("vanilla", "epoch3", "teacher"):
            got_v = got.get(model)
            exp_v = exp[model]
            if got_v is None or abs(got_v - exp_v) > 0.005:
                print(f"FAIL: qtype={qt} model={model} expected={exp_v:.3f} got={got_v}")
                sys.exit(1)
        if counts[qt] != exp["n"]:
            print(f"WARN: qtype={qt} n={counts[qt]} expected={exp['n']}")

    # Order by 3-epoch accuracy desc
    qtypes = sorted(accuracies.keys(), key=lambda q: -accuracies[q]["epoch3"])

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = np.arange(len(qtypes))
    width = 0.27

    vanilla_vals = [accuracies[q]["vanilla"] for q in qtypes]
    epoch3_vals  = [accuracies[q]["epoch3"]  for q in qtypes]
    teacher_vals = [accuracies[q]["teacher"] for q in qtypes]

    ax.bar(x - width, vanilla_vals, width, label="Vanilla 0.8B",
           color=COLORS["vanilla"], edgecolor="black", linewidth=0.5)
    ax.bar(x,         epoch3_vals,  width, label="3-epoch trained",
           color=COLORS["epoch3"],  edgecolor="black", linewidth=0.5)
    ax.bar(x + width, teacher_vals, width, label="Teacher 27B-AWQ-Int4",
           color=COLORS["teacher"], edgecolor="black", linewidth=0.5)

    # Annotate accuracy + n on each bar
    for xi, (q, vv, ev, tv) in enumerate(zip(qtypes, vanilla_vals, epoch3_vals, teacher_vals)):
        for off, val in zip([-width, 0, width], [vv, ev, tv]):
            ax.text(xi + off, val + 0.018, f"{val*100:.1f}",
                    ha="center", fontsize=8)
        ax.text(xi, -0.045, f"n={counts[q]}",
                ha="center", fontsize=8, color="gray", transform=ax.get_xaxis_transform())

    # +12.0 pp Counterfactual callout
    cf_idx = qtypes.index("Counterfactual")
    cf_3ep = accuracies["Counterfactual"]["epoch3"]
    cf_tch = accuracies["Counterfactual"]["teacher"]
    delta = (cf_3ep - cf_tch) * 100
    ax.annotate(
        f"+{delta:.1f} pp\nstudent > teacher",
        xy=(cf_idx, cf_3ep), xytext=(cf_idx + 0.5, cf_3ep + 0.08),
        fontsize=9, color=COLORS["delta_pos"], fontweight="bold",
        ha="left",
        arrowprops=dict(arrowstyle="->", color=COLORS["delta_pos"], lw=1.4),
    )

    # Structural-floor horizontal dashed line at 0.15
    ax.axhline(0.15, color=COLORS["delta_neg"], linestyle="--", lw=1.0, alpha=0.7)
    ax.text(len(qtypes) - 1, 0.16, "structural floor (§5.3)",
            ha="right", fontsize=8, color=COLORS["delta_neg"], fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(qtypes, fontsize=9)
    ax.set_ylabel("Accuracy (gold-fuzzy ≥ 0.80)")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([f"{int(t*100)}%" for t in np.arange(0, 1.01, 0.2)])
    ax.set_title("Per-question-type accuracy on 1 000 test questions")
    ax.legend(loc="upper right", ncol=3)

    plt.tight_layout()
    plt.savefig(OUT)
    plt.close()

    print("F4: " + ", ".join(f"{q}(n={counts[q]}):v{accuracies[q]['vanilla']*100:.1f}/3{accuracies[q]['epoch3']*100:.1f}/t{accuracies[q]['teacher']*100:.1f}" for q in qtypes) + "  [matches Table 4.6 ±0.5pp]")


if __name__ == "__main__":
    main()
