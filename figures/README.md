# Report figures — handoff back to the human author

8 publication-quality figures for the GP2 final report, all derived (where possible) from on-disk artefacts and verified to match the report's headline numbers within ±0.5 pp.

Re-run any figure: `.venv/bin/python figures/Fn_*.py`. The script will print a one-line drift check on stdout and either save the PNG or **halt** if the re-derived number differs from the report by more than the tolerance.

## Where each figure goes in the report

| File | Target report section | Replaces / supplements | Source |
|---|---|---|---|
| [F1_training_loss.png](F1_training_loss.png) | §4.4 (or just before Table 4.3) | Table 4.3 | `train_losses.jsonl` |
| [F2_headline_accuracy.png](F2_headline_accuracy.png) | §4.5 | Table 4.4 | 4 × `predictions_test_*.jsonl` |
| [F3_per_sport.png](F3_per_sport.png) | §4.6 | Table 4.5 | vanilla / 3-epoch / teacher predictions |
| [F4_per_qtype.png](F4_per_qtype.png) | §4.7 | Table 4.6 | vanilla / 3-epoch / teacher predictions |
| [F5_pairwise.png](F5_pairwise.png) | §4.8 | Table 4.7 | 3-epoch + teacher predictions joined on qa_id |
| [F6_pipeline.png](F6_pipeline.png) | §3.3.5 | the ASCII diagram | none — encoded layout |
| [F7_timeline.png](F7_timeline.png) | §1.4 | Table 1.1 | hardcoded from handoff §F7 |
| [F8_compression_gap.png](F8_compression_gap.png) | §2.3.4 | Table 2.3 | hardcoded from handoff §F8 |

## Reproducibility / agreement table

Each script prints its drift check on stdout. Copy-pasted from `python figures/Fn_*.py` runs:

| Figure | Re-derived | Report value | Match? |
|---|---|---|---|
| F1 | steps=35 016 · CE 0.967→0.443 (-54%) · KL 1.732→0.287 (-83%) | 35 016 / -54% / -83% | ✓ |
| F2 | vanilla=22.7%, 1ep=36.5%, 3ep=39.9%, teacher=39.7% | 22.7 / 36.5 / 39.9 / 39.7 | ✓ |
| F3 | basketball v39.5/3ep47.5/t41.5 · volleyball v21.5/3ep42.0/t41.0 · fg v21.5/3ep40.5/t44.0 · aerobic_gymnastics v13.0/3ep38.0/t39.0 · football v18.0/3ep31.5/t33.0 | matches Table 4.5 | ✓ |
| F4 | Counterfactual(n=150) v46.7/3ep82.0/t70.0 · Descriptive(n=391) v34.8/3ep61.9/t63.7 · Temporal(n=306) v6.9/3ep10.8/t13.1 · Causal(n=153) v0.0/3ep0.7/t2.0 | matches Table 4.6 | ✓ |
| F5 | both_correct=333, student_only=66, teacher_only=64, both_wrong=537, net=+2 | 333 / 66 / 64 / 537 / +2 | ✓ |
| F6 | (static layout — no derived numbers) | n/a | ✓ |
| F7 | (durations from handoff §F7 — verify against your Table 1.1) | n/a | **user-verify** |
| F8 | (prior-work ratios from handoff §F8 — verify against your Table 2.3; Hakam ratio 30× hardcoded) | n/a | **user-verify** |

## Caveats / things the human author should know

- **F7 and F8 are not data-derived.** The phase durations on F7 and the prior-work ratios on F8 come from the handoff document, not from this repo's artefacts. Cross-check them against the values you used in your report's Table 1.1 (F7) and Table 2.3 (F8) before publication.
- **F8 Hakam ratio is `30×`.** Arithmetic 27 B / 0.8 B = 33.75×, but the handoff specifies 30× to match the report. The bar uses 30× as supplied.
- **F4 `question_type` field is present in all four prediction files** — no fallback join to `subset_test.json` was needed. The script's defensive code path was not exercised.
- **F3 sport bucketing keeps `fg` lumped.** This matches the report's Table 4.5 — the per-sport-bucket count is 5 (aerobic_gymnastics / basketball / fg / football / volleyball), not 8. Disaggregating fg to vault/UB/BB/FX would require joining to `subset_test.json` on qa_id; the report's table stays at 5 buckets, so the figure does too.
- **Field-name notes (codified in `_metric.py`).** Predictions use `student_text` (not `pred`) and `question_type` (not `qtype`). Sport: `predictions_test_trained_3ep.jsonl` and `predictions_test_teacher.jsonl` carry an explicit `sport` field; the older two (`vanilla`, 1-epoch trained) don't, and the metric helper falls back to `video_id.split("/", 1)[0]`. This matches the eval_compare.py fallback.
- **All scripts are self-contained.** Each `Fn_*.py` imports `_style` and (where relevant) `_metric`. Re-running a single figure does not require running others first.
- **Figure 6 is wider than the others (~636 KB).** Pipeline flowchart with a lot of text. Stays under the 2 MB cap and prints fine at 6.5"-content-width.

## Style consistency

All 8 figures share:

- DejaVu Sans, 10 pt body / 12 pt title, no top/right spines, dashed grid at α=0.25.
- Okabe-Ito-derived palette: vanilla `#999999`, 1-epoch `#56B4E9`, 3-epoch `#0072B2`, teacher `#E69F00`, win-green `#009E73`, regress-red `#D55E00`, magenta callout `#CC79A7`.
- 6.5"-wide content (fits Word with 1" margins). 300 DPI export.
- No 3D, no exploded pies, no gradient fills, no rainbow colormaps.

## Insertion checklist for the report author

1. In Word, replace each existing table's numbers/text with the corresponding figure (Insert → Picture → from file).
2. Re-number all figure callouts in the body text. Suggested figure numbers:
   - F1 → Figure 4.1, F2 → Figure 4.2, F3 → Figure 4.3, F4 → Figure 4.4, F5 → Figure 4.5
   - F6 → Figure 3.1
   - F7 → Figure 1.1
   - F8 → Figure 2.1
3. Each script's docstring suggests a caption. Use it verbatim or paraphrase.
4. Set the figure to "in line with text" placement; centre horizontally.
5. Verify figure text is legible at print scale (any thumbnail check should still show the bar values).
6. **Before submitting**: open all 8 PNGs in a row (browser, image viewer, or `feh figures/*.png`) and confirm visual consistency.
