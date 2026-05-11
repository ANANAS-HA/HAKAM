"""F6 — End-to-end pipeline architecture diagram.

Vertical flowchart with two hardware-zone backgrounds (rented pod = orange,
local WSL2 = blue). Boxes are persistent artefacts; arrows are scripts or
rsync transfers. Replaces the ASCII diagram in §3.3.5.

Pure matplotlib + FancyBboxPatch — no mermaid-cli or graphviz needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).parent))
from _style import COLORS, apply_style


OUT = Path(__file__).parent / "F6_pipeline.png"

# Layout: vertical pipeline. (y descending = top→bottom).
# Each entry: (id, label, sub-label, x_center, y_top, color_zone)

LOCAL_FILL = "#E8F1F8"
POD_FILL   = "#FCEDD8"
NEUTRAL    = "#F4F4F4"


def main() -> None:
    apply_style()

    fig, ax = plt.subplots(figsize=(7.5, 11.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 26)
    ax.set_aspect("auto")
    ax.axis("off")

    # Zone backgrounds (drawn first so boxes float on top)
    ax.add_patch(Rectangle((0.4, 13.4), 9.2, 5.5, facecolor=POD_FILL,
                           edgecolor="none", alpha=0.55, zorder=0))
    ax.text(0.55, 18.7, "POD ZONE  (rented H100-NVL, ~96 GB VRAM, 172 GB RAM)",
            fontsize=9, color="#7A4A0F", fontstyle="italic", zorder=1)

    ax.add_patch(Rectangle((0.4, 7.4), 9.2, 4.5, facecolor=POD_FILL,
                           edgecolor="none", alpha=0.55, zorder=0))
    ax.text(0.55, 11.7, "POD ZONE  (--pod_mode: upfront vision cache)",
            fontsize=9, color="#7A4A0F", fontstyle="italic", zorder=1)

    ax.add_patch(Rectangle((0.4, 0.3), 9.2, 6.5, facecolor=LOCAL_FILL,
                           edgecolor="none", alpha=0.55, zorder=0))
    ax.text(0.55, 6.6, "LOCAL ZONE  (WSL2 / RTX 3090, 24 GB VRAM)",
            fontsize=9, color="#15446B", fontstyle="italic", zorder=1)

    # ---- box helper ----
    def box(x, y, w, h, text, sub="", fill=NEUTRAL, edge="black",
            mono=False, fontsize=9.5, sub_fontsize=8):
        ax.add_patch(FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            facecolor=fill, edgecolor=edge, linewidth=1.0, zorder=2,
        ))
        if mono:
            ax.text(x, y + (0.10 if sub else 0), text, ha="center", va="center",
                    fontsize=fontsize, family="monospace", zorder=3)
        else:
            ax.text(x, y + (0.10 if sub else 0), text, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", zorder=3)
        if sub:
            ax.text(x, y - 0.18, sub, ha="center", va="center",
                    fontsize=sub_fontsize, color="#444", zorder=3, fontstyle="italic")

    def arrow(x1, y1, x2, y2, label="", dashed=False, label_offset=(0.15, 0)):
        style = "->,head_length=8,head_width=6"
        ls = (0, (4, 3)) if dashed else "solid"
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle=style, linestyle=ls,
                                     color="#222", linewidth=1.1, zorder=4,
                                     mutation_scale=12))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + label_offset[0], my + label_offset[1], label,
                    fontsize=8, color="#444", fontstyle="italic",
                    family="monospace", zorder=5)

    # ---- nodes ----
    box(5, 25.0, 6.2, 0.9, "Sports-QA dataset",
        sub="75 355 questions / 4 771 unique videos, 8 sports", fill=NEUTRAL)

    box(5, 23.0, 6.2, 0.9, "Curated subsets",
        sub="train 1 600 / val 480 / test 480 videos · 14 423 questions", fill=NEUTRAL)

    box(5, 21.0, 6.2, 0.9, "Local video files",
        sub="videos/ ~25 GB, 8 sport subdirectories", fill=LOCAL_FILL)

    box(5, 17.7, 6.2, 1.1, "extract_logits_subset.py",
        sub="cyankiwi/Qwen3.5-27B-AWQ-4bit · ~17 s/sample · ~50 hr",
        fill="white", mono=True)

    box(5, 15.2, 6.2, 1.1, "logits_cache_subset_final/",
        sub="11 680 .pt files · 8 169 valid · 347 MB · top-32 KL cache",
        fill="white", mono=True)

    box(5, 10.8, 6.2, 1.1, "train_student.py --pod_mode",
        sub="Qwen3.5-0.8B + LoRA r=16 · 3 epochs × 11 672 samples · ~3 hr",
        fill="white", mono=True)

    box(5, 8.3, 6.2, 0.9, "student_lora_full3epoch/",
        sub="adapter_model.safetensors · 25 MB · 6.4 M params (0.74%)",
        fill="white", mono=True)

    box(5, 5.4, 6.2, 1.1, "eval_test_predict.py × 4 models",
        sub="vanilla / 1-epoch / 3-epoch / teacher · 1 000 stratified test Q",
        fill="white", mono=True)

    box(5, 2.9, 6.2, 1.1, "eval_compare.py",
        sub="fuzzy Levenshtein @ 0.80 · 4-bucket flip-rate · headline accuracy",
        fill="white", mono=True)

    box(5, 0.9, 6.2, 0.85, "predictions_test_*.jsonl  +  eval_test_flip_report.md",
        sub="final report figures (F2-F5) sourced here",
        fill=LOCAL_FILL, mono=True, fontsize=9, sub_fontsize=7.5)

    # ---- arrows ----
    arrow(5, 24.55, 5, 23.45, "sample_subset.py · sample_test_subset.py")
    arrow(5, 22.55, 5, 21.45, "download_subset_videos.py")
    arrow(5, 20.55, 5, 18.25, "rsync push code + subsets", dashed=True)
    arrow(5, 17.15, 5, 15.75, "writes per-question .pt + manifest")
    arrow(5, 14.65, 5, 11.35, "rsync pull cache to pod again", dashed=True)
    arrow(5, 10.25, 5, 8.75)
    arrow(5, 7.85, 5, 5.95, "rsync pull adapter back to WSL", dashed=True)
    arrow(5, 4.85, 5, 3.45)
    arrow(5, 2.35, 5, 1.32)

    plt.tight_layout()
    plt.savefig(OUT)
    plt.close()

    print("F6: pipeline diagram saved (matplotlib, no external deps)")


if __name__ == "__main__":
    main()
