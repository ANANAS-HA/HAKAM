"""F3 — Per-sport accuracy: vanilla / 3-epoch / teacher × 5 sport buckets.

Note: predictions report against the upstream `fg/` prefix (FineGym dismount
events are lumped). This matches the report's Table 4.5; do not disaggregate.

Bars are grouped per sport (5 groups × 3 bars), ordered by 3-epoch accuracy
descending. Wins (3-epoch > teacher) marked with a green Δ above the bar.

Source: predictions_test_{vanilla,trained_3ep,teacher}.jsonl (n=200 each
sport bucket). Target report section: §4.6, replaces Table 4.5.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _metric import get_sport, is_correct, load_jsonl
from _style import COLORS, FIGSIZE_WIDE, apply_style


ROOT = Path(__file__).parent.parent
PRED_FILES = {
    "vanilla":  ROOT / "predictions_test_vanilla.jsonl",
    "epoch3":   ROOT / "predictions_test_trained_3ep.jsonl",
    "teacher":  ROOT / "predictions_test_teacher.jsonl",
}
OUT = Path(__file__).parent / "F3_per_sport.png"

# Reference values from the report's Table 4.5 (and verified in earlier
# conversation runs). Used as a drift check.
EXPECTED = {
    "aerobic_gymnastics": {"vanilla": 0.130, "epoch3": 0.380, "teacher": 0.390},
    "basketball":         {"vanilla": 0.395, "epoch3": 0.475, "teacher": 0.415},
    "fg":                 {"vanilla": 0.215, "epoch3": 0.405, "teacher": 0.440},
    "football":           {"vanilla": 0.180, "epoch3": 0.315, "teacher": 0.330},
    "volleyball":         {"vanilla": 0.215, "epoch3": 0.420, "teacher": 0.410},
}


def main() -> None:
    apply_style()

    # Load + aggregate per sport, joining 4 prediction files on qa_id.
    by_sport: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for label, path in PRED_FILES.items():
        for r in load_jsonl(path):
            sport = get_sport(r)
            ok = is_correct(r["student_text"], r["gold"])
            by_sport[sport][label].append(ok)

    accuracies: dict[str, dict[str, float]] = {}
    for sport, by_model in by_sport.items():
        accuracies[sport] = {m: sum(v) / len(v) for m, v in by_model.items()}

    # Drift check
    for sport, exp in EXPECTED.items():
        got = accuracies.get(sport, {})
        for model, exp_v in exp.items():
            got_v = got.get(model)
            if got_v is None or abs(got_v - exp_v) > 0.005:
                print(f"FAIL: sport={sport} model={model} expected={exp_v:.3f} got={got_v}")
                sys.exit(1)

    # Order sports by 3-epoch accuracy (descending)
    sports = sorted(accuracies.keys(), key=lambda s: -accuracies[s]["epoch3"])

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = np.arange(len(sports))
    width = 0.27

    vanilla_vals = [accuracies[s]["vanilla"] for s in sports]
    epoch3_vals  = [accuracies[s]["epoch3"]  for s in sports]
    teacher_vals = [accuracies[s]["teacher"] for s in sports]

    ax.bar(x - width, vanilla_vals, width, label="Vanilla 0.8B",
           color=COLORS["vanilla"], edgecolor="black", linewidth=0.5)
    ax.bar(x,         epoch3_vals,  width, label="3-epoch trained",
           color=COLORS["epoch3"],  edgecolor="black", linewidth=0.5)
    ax.bar(x + width, teacher_vals, width, label="Teacher 27B-AWQ-Int4",
           color=COLORS["teacher"], edgecolor="black", linewidth=0.5)

    # Annotate accuracy on each bar
    for xi, (vv, ev, tv) in enumerate(zip(vanilla_vals, epoch3_vals, teacher_vals)):
        for off, val in zip([-width, 0, width], [vv, ev, tv]):
            ax.text(xi + off, val + 0.012, f"{val*100:.1f}",
                    ha="center", fontsize=8)

    # Mark sports where 3-epoch > teacher with a green Δ
    for xi, s in enumerate(sports):
        diff = accuracies[s]["epoch3"] - accuracies[s]["teacher"]
        if diff > 0:
            ax.text(xi, accuracies[s]["epoch3"] + 0.045,
                    f"Δ +{diff*100:.1f}",
                    ha="center", fontsize=8, color=COLORS["delta_pos"],
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in sports], fontsize=9, rotation=10)
    ax.set_ylabel("Accuracy (gold-fuzzy ≥ 0.80)")
    ax.set_ylim(0, max(vanilla_vals + epoch3_vals + teacher_vals) * 1.25)
    ax.set_yticks(np.arange(0, 0.61, 0.1))
    ax.set_yticklabels([f"{int(t*100)}%" for t in np.arange(0, 0.61, 0.1)])
    ax.set_title("Per-sport accuracy on 1 000 test questions (n = 200 per sport bucket)")
    ax.legend(loc="upper right", ncol=3)

    plt.tight_layout()
    plt.savefig(OUT)
    plt.close()

    print("F3: " + ", ".join(f"{s}=v{accuracies[s]['vanilla']*100:.1f}/3{accuracies[s]['epoch3']*100:.1f}/t{accuracies[s]['teacher']*100:.1f}" for s in sports) + "  [matches Table 4.5 ±0.5pp]")


if __name__ == "__main__":
    main()
