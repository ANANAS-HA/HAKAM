# Logit Extraction — Report

## What I set out to do

The distillation pipeline needs **per-token logits** from the teacher model so that the student can learn more than just the gold label — it can learn the teacher's full distribution over the vocabulary at each step. This report covers the prototype extraction I built at [extract_logits_experiment.py](extract_logits_experiment.py), the design decisions, and what the output looks like.

The prototype uses **Qwen3.5-0.8B as a teacher stand-in** so everything runs in seconds on the 3090. The mechanics are identical to what the real 27B teacher will do on a rented A100/H100; only scale changes.

## Two recipes, one script

Both recipes are implemented so we can see their trade-offs side by side.

**Recipe A — free generation.** Let the teacher generate an answer greedily, record the full-vocab logits at every generated position, then keep only top-K per position. This is what the HANDOFF's `01_extract_logits.py` describes. Rich signal (many tokens per sample), but the teacher's style is whatever the prompt gets it to produce — hence my earlier recommendation to constrain the prompt so the student learns the target format.

**Recipe B — teacher-forced on the gold label.** Feed the model `[prompt + gold_answer]` as a single forward pass, then read the logits at the answer positions. No autoregressive loop, so it's faster. Produces a much shorter sequence (gold labels are 1–3 tokens for this dataset), but the signal is focused precisely on the tokens we want the student to emit.

The earlier discussion about "train unconstrained, eval constrained" maps cleanly to these two recipes: mix both during distillation, with Recipe A providing reasoning/language signal and Recipe B providing format-following signal.

## Storage and correctness choices

- **Top-K, not full vocab.** Qwen3.5's vocabulary is **248,320 tokens**. Storing full logits for every position for 10k samples is hundreds of GB. Top-32 captures 96–100 % of the probability mass in every sample I tested, and drops storage to ~100 MB for the whole dataset. Same trick every real distillation pipeline uses.
- **fp16 values, int32 indices.** Halves the storage vs fp32 with no meaningful loss.
- **Gold token is force-included** for Recipe B. If the teacher disagrees so hard with the gold that the gold token falls out of top-32, KL at that position becomes undefined for the student. The script replaces the lowest-ranked top-K slot with the gold token's logit in that case. Didn't trigger on the 5 samples I tested but is there as a safety net.
- **Greedy-consistency check.** For Recipe A, argmax over the stored top-K logits must exactly reproduce the generated token sequence. All 5 samples passed. This is the canary for alignment bugs — any `greedy_consistent=False` from Recipe A is a real problem.

## Results on 5 training samples (0.8B teacher proxy)

| Metric | Value |
|---|---|
| Top-32 probability mass | 0.96 – 1.00 |
| Recipe A greedy-consistent | 5 / 5 |
| Recipe B gold-in-top-32 (before fix) | 5 / 5 |
| Recipe A time / sample | ~1.7 s |
| Recipe B time / sample | ~2.0 s |
| Saved bytes / sample | ~9 KB (A + B combined) |
| Projected cache for 10k samples | ~100 MB |
| Peak VRAM | 3.4 GB |

The most interesting case was the question "How many types of actions do the players perform?" — gold `"2"`, but the 0.8B teacher greedily predicts `"3"`. Recipe B's stored distribution contains both, with non-zero probability on the gold token. A hard-label cross-entropy loss would throw this disagreement information away; KL over the full top-K distribution preserves it. **This is exactly the "dark knowledge" distillation is meant to transfer.**

## Projected cost on the real 27B teacher

At A100 80 GB prices (~$0.80/hr on Vast.ai on-demand) and a rough ~3–5 s per sample on 27B-bf16, 10k samples ≈ 8–14 h ≈ **$7–12** for a full extraction run. Well inside the project budget.

## What still needs building before the rental run

1. **Video-level caching**: the test set averages 15.7 questions per video. Decoding + re-encoding the video for every question wastes ~93 % of vision-side compute. A production script should decode each video once and loop questions inside.
2. **Per-sample checkpointing**: one `.pt` or `.safetensors` per `qa_id`, plus a manifest JSON, so a crashed run can resume by listing `logits_cache/` and skipping anything already on disk.
3. **Schema-locked on-disk format**: fixed keys so the training-side loader doesn't have to know recipe-specific details.
4. **A separate run for the "constrained" Track B**: matches the "two-prompt" strategy we discussed so the student learns to switch formats based on the system prompt.

The tutorial notebook ([logits_tutorial.ipynb](logits_tutorial.ipynb)) walks through what the saved logits actually contain and how to read / visualize them.
