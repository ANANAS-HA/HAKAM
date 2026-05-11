"""F8 — Teacher-to-student parameter compression ratio: Hakam vs prior work.

Horizontal log-scale bars showing teacher/student parameter ratios in the
nearest comparable prior multimodal-VLM distillation work, with this project's
30× ratio highlighted.

Source: Table 2.3 in §2.3.4 of the GP2 report. NOT re-derivable from disk —
the prior-work numbers are encoded from the handoff. Verify against the
report's Table 2.3 before publishing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _style import COLORS, FIGSIZE_SINGLE, apply_style


OUT = Path(__file__).parent / "F8_compression_gap.png"

# (label, ratio, modality, is_this_project)
WORKS = [
    ("LLaVA-KD",                 3.9,  "image-text VLM",    False),
    ("MiniCPM-V (distilled)",    3.5,  "image-text VLM",    False),
    ("TinyLLaVA",                3.5,  "image-text VLM",    False),
    ("Hakam (this project)",     30.0, "video VLM",         True),
]

MODALITY_COLOR = {
    "image-text VLM": "#9DBFD9",
    "video VLM":      COLORS["highlight"],
}


def main() -> None:
    apply_style()

    works_sorted = sorted(WORKS, key=lambda w: w[1])  # ascending so largest at top
    labels = [w[0] for w in works_sorted]
    ratios = [w[1] for w in works_sorted]
    colors = [
        COLORS["highlight"] if w[3] else MODALITY_COLOR[w[2]]
        for w in works_sorted
    ]
    edges = ["black"] * len(works_sorted)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    y = np.arange(len(labels))
    bars = ax.barh(y, ratios, color=colors, edgecolor=edges, linewidth=0.6)

    # Annotate ratio at the right end of each bar
    for yi, r, w in zip(y, ratios, works_sorted):
        is_self = w[3]
        txt = f"{r:.1f}×"
        if is_self:
            ax.text(r * 1.05, yi, txt, va="center", fontsize=10,
                    color=COLORS["highlight"], fontweight="bold")
        else:
            ax.text(r * 1.05, yi, txt, va="center", fontsize=9, color="#333")

    # Callout pointing to Hakam
    self_idx = next(i for i, w in enumerate(works_sorted) if w[3])
    ax.annotate(
        "≈ 8× more aggressive than prior\nmultimodal-VLM distillation",
        xy=(WORKS[-1][1] * 0.95, self_idx),
        xytext=(11, self_idx - 1.7),
        fontsize=9, color=COLORS["highlight"], fontweight="bold",
        ha="left",
        arrowprops=dict(arrowstyle="->", color=COLORS["highlight"], lw=1.4),
    )

    ax.set_xscale("log")
    ax.set_xlim(2, 60)
    ax.set_xticks([3, 5, 10, 20, 30, 50])
    ax.set_xticklabels(["3×", "5×", "10×", "20×", "30×", "50×"])
    ax.set_xlabel("Teacher / student parameter ratio (log scale)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_title("Compression ratio in prior multimodal-VLM distillation vs this project")

    # Modality legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=MODALITY_COLOR["image-text VLM"],
                      ec="black", lw=0.5, label="Image-text VLM"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["highlight"],
                      ec="black", lw=0.5, label="Video VLM (this project)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(OUT)
    plt.close()

    print("F8: compression-gap chart saved (prior-work ratios from handoff/Table 2.3)")


if __name__ == "__main__":
    main()
