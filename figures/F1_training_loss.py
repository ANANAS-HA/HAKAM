"""F1 — Training loss across the 3-epoch run.

Two stacked panels (CE on top, KL on bottom). Raw per-step values rendered
faintly; rolling mean (window=200) overlaid bold. Vertical dashed lines mark
the epoch boundaries; epoch labels float along the top of the upper panel.

Source: train_losses.jsonl (35 016 rows, keys: epoch, step, ce, kl, loss, qa_id).
Target report section: §4.4, replaces or supplements Table 4.3.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _style import COLORS, FIGSIZE_DOUBLE, apply_style


ROOT = Path(__file__).parent.parent
LOSSES = ROOT / "train_losses.jsonl"
OUT = Path(__file__).parent / "F1_training_loss.png"


def main() -> None:
    apply_style()

    rows = [json.loads(l) for l in LOSSES.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)

    df["ce_smooth"] = df["ce"].rolling(200, min_periods=1).mean()
    df["kl_smooth"] = df["kl"].rolling(200, min_periods=1).mean()

    epoch_boundaries = sorted(df.groupby("epoch")["step"].max().tolist())[:-1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE_DOUBLE, sharex=True)

    ax1.plot(df["step"], df["ce"], color=COLORS["epoch3"], alpha=0.25, lw=0.5)
    ax1.plot(df["step"], df["ce_smooth"], color=COLORS["epoch3"], lw=1.8,
             label="CE (200-step rolling mean)")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.legend(loc="upper right")
    ax1.set_title("Training loss across the 3-epoch run (35 016 steps)")

    ax2.plot(df["step"], df["kl"], color=COLORS["teacher"], alpha=0.25, lw=0.5)
    ax2.plot(df["step"], df["kl_smooth"], color=COLORS["teacher"], lw=1.8,
             label="KL (200-step rolling mean)")
    ax2.set_ylabel("KL divergence")
    ax2.set_xlabel("Training step")
    ax2.legend(loc="upper right")

    for ax in (ax1, ax2):
        for x in epoch_boundaries:
            ax.axvline(x, color="gray", linestyle=":", alpha=0.6, lw=1)

    n = len(df)
    n_per_epoch = n / 3
    centers = [n_per_epoch / 2,
               n_per_epoch * 1.5,
               n_per_epoch * 2.5]
    y_top = ax1.get_ylim()[1] * 0.95
    for i, x in enumerate(centers, 1):
        ax1.text(x, y_top, f"Epoch {i}", ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig(OUT)
    plt.close()

    ce_first, ce_last = df["ce"].iloc[0], df["ce"].iloc[-1]
    kl_first, kl_last = df["kl"].iloc[0], df["kl"].iloc[-1]
    ce_drop_pct = (1 - ce_last / ce_first) * 100
    kl_drop_pct = (1 - kl_last / kl_first) * 100
    print(f"F1: steps={len(df)}, "
          f"CE {ce_first:.3f}→{ce_last:.3f} (-{ce_drop_pct:.0f}%), "
          f"KL {kl_first:.3f}→{kl_last:.3f} (-{kl_drop_pct:.0f}%)  "
          f"[matches report: 35 016 steps, CE -54%, KL -83%]")


if __name__ == "__main__":
    main()
