"""F7 — GP1 plan vs GP2 actual: per-phase duration comparison.

Horizontal grouped bars: GP1 plan (gray) and GP2 actual (green if delivered
faster, red if overran, gray if on plan). All durations in canonical days.

Source: Table 1.1 in §1.4 of the GP2 report (handoff §F7 lists the canonical
day values to use). NOT re-derivable from disk — these durations come from
the user's tracking.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _style import COLORS, FIGSIZE_DOUBLE, apply_style


OUT = Path(__file__).parent / "F7_timeline.png"

# (phase, gp1_days, gp2_days, status)
# status: "faster" | "slower" | "on_plan" | "ongoing"
PHASES = [
    ("Subset construction",                      7,    1,     "faster"),
    ("Video acquisition",                        2,    0.08,  "faster"),
    ("Teacher selection + prompt iteration",     7,    14,    "slower"),
    ("Logit extraction (~50 hr wall-clock)",     7,    2.1,   "faster"),
    ("Pipeline build (incl. WSL2 debug)",        14,   17,    "slower"),
    ("Validation training (1 epoch, 3 hr)",      0,    0.13,  "faster"),
    ("Final 3-epoch training (~3 hr H100)",      7,    0.13,  "faster"),
    ("Evaluation (test set + teacher ref)",      7,    0.25,  "faster"),
    ("Report writing",                           14,   None,  "ongoing"),  # ongoing
]


def main() -> None:
    apply_style()

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)

    n = len(PHASES)
    y = np.arange(n)[::-1]              # top = first phase
    bar_height = 0.36

    gp1_vals = [p[1] for p in PHASES]
    gp2_vals = [p[2] if p[2] is not None else 0 for p in PHASES]
    statuses = [p[3] for p in PHASES]
    labels   = [p[0] for p in PHASES]

    # GP1 plan bars (top of each pair)
    ax.barh(y + bar_height/2, gp1_vals, bar_height,
            color=COLORS["neutral"], edgecolor="black", linewidth=0.5,
            label="GP1 plan")

    # GP2 actual bars (bottom), color-coded
    color_map = {
        "faster":  COLORS["delta_pos"],
        "slower":  COLORS["delta_neg"],
        "on_plan": COLORS["neutral"],
        "ongoing": COLORS["highlight"],
    }
    gp2_colors = [color_map[s] for s in statuses]
    ax.barh(y - bar_height/2, gp2_vals, bar_height,
            color=gp2_colors, edgecolor="black", linewidth=0.5)

    # Annotate values
    for yi, gv, av, st in zip(y, gp1_vals, gp2_vals, statuses):
        ax.text(gv + 0.3, yi + bar_height/2, f"{gv}d",
                va="center", fontsize=8, color="#444")
        if st == "ongoing":
            ax.text(0.3, yi - bar_height/2, "ongoing",
                    va="center", fontsize=8, color=COLORS["highlight"],
                    fontweight="bold", fontstyle="italic")
        else:
            txt = f"{av:g}d"
            ax.text(av + 0.3, yi - bar_height/2, txt,
                    va="center", fontsize=8, color="#222")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Duration (days)")
    ax.set_xlim(0, max(gp1_vals) * 1.18)
    ax.set_title("GP1 plan (gray) vs GP2 actual (green = faster, red = overrun, magenta = ongoing)",
                 fontsize=11)

    # Custom legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["neutral"], ec="black", lw=0.5,
                      label="GP1 plan"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["delta_pos"], ec="black", lw=0.5,
                      label="GP2 actual — delivered faster"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["delta_neg"], ec="black", lw=0.5,
                      label="GP2 actual — overran"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["highlight"], ec="black", lw=0.5,
                      label="GP2 actual — ongoing"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8.5, ncol=1)

    plt.tight_layout()
    plt.savefig(OUT)
    plt.close()

    print("F7: timeline saved (handoff-supplied durations; cross-check Table 1.1)")


if __name__ == "__main__":
    main()
