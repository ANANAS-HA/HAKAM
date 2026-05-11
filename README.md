# Sports-QA Knowledge Distillation: Qwen3.5 27 B → 0.8 B

**Graduation Project 2 submission.** Recipe-A free-generation distillation of a 27 B AWQ-Int4 Qwen3.5 teacher into a 0.8 B LoRA-adapted student for sports video question answering on the Sports-QA dataset.

---

## Headline result

The 3-epoch LoRA student matches the 27 B teacher's accuracy within statistical noise — **at a 30× parameter compression** — on a held-out 1 000-question stratified test sample.

| model | params | gold-fuzzy accuracy ¹ | text-similarity to teacher ² |
|---|---:|---:|---:|
| Vanilla Qwen3.5 0.8 B | 0.8 B | 22.7 % | 0.172 |
| **Trained student (3 epochs LoRA r=16)** | **0.8 B + 6.4 M LoRA** | **39.9 %** | **0.323** |
| Teacher (Qwen3.5 27 B AWQ-Int4) | 27 B | 39.7 % | (reference) |

Student vs teacher Δ = +0.2 pp (well within the ±3 pp 95 % binomial CI at n=1 000). The student even beats the teacher on **Counterfactual** questions (82.0 % vs 70.0 %, n=150).

¹ Sliding-window Levenshtein similarity ≥ 0.80 against the gold label. See [`eval_test_flip_report.md`](eval_test_flip_report.md) for the full per-sport and per-question-type breakdown.
² Mean character-level Levenshtein.ratio between the student's free-form answer and the teacher's free-form answer on the same prompt + video.

See [`figures/F2_headline_accuracy.png`](figures/F2_headline_accuracy.png) for the bar chart and [`figures/F5_pairwise.png`](figures/F5_pairwise.png) for the pairwise item-level outcome.

---

## What's in this repo

### Reports & methodology

| file | content |
|---|---|
| [`GP2_REPORT_SOURCE.md`](GP2_REPORT_SOURCE.md) | 9-section source-of-truth doc for the GP2 final report. Maps onto the official chapter template. |
| [`METHODOLOGY_REPORT.md`](METHODOLOGY_REPORT.md) | Chronological academic methodology write-up: every experimental decision and the evidence that motivated it. |
| [`vanilla_train_test_logits_handling.md`](vanilla_train_test_logits_handling.md) | End-to-end technical reference (15 sections). |
| [`SUBSET_HANDLING.md`](SUBSET_HANDLING.md) | Subset construction (per-sport sampling, yes/no rebalancing, FineGym sub-event recovery). |
| [`LOGITS_REPORT.md`](LOGITS_REPORT.md) | Early logit-extraction experiment record. |
| [`RUNBOOK_RENTAL.md`](RUNBOOK_RENTAL.md) | Step-by-step rental → extraction commands for the H100-NVL pod. |
| [`appendix_materials.md`](appendix_materials.md) | Appendix source: verbatim configs, environment versions, file inventory. |

### Result reports (held-out + test set)

| file | content |
|---|---|
| [`eval_test_flip_report.md`](eval_test_flip_report.md) | **Headline test-set flip-rate, n=1000**: vanilla → 3-epoch trained, sport/qtype breakdown. |
| [`eval_flip_report.md`](eval_flip_report.md) | Held-out flip-rate, n=100 (1-epoch validation run). |
| [`eval_comparison.md`](eval_comparison.md) | Per-row side-by-side from `train_student.py`'s held-out evaluation. |

### Production scripts (in pipeline order)

| script | role |
|---|---|
| [`sample_subset.py`](sample_subset.py) | Build the 200-vid-per-sport train + 60-per-sport val subset. |
| [`sample_test_subset.py`](sample_test_subset.py) | Build the 60-vid-per-sport test subset (recovers FineGym sub-events from Descriptive answers). |
| [`download_subset_videos.py`](download_subset_videos.py) | Parallel HF-Hub video download (8 workers). |
| [`extract_logits_subset.py`](extract_logits_subset.py) | Recipe-A teacher logit extraction. **Pod-only** — requires ~96 GB VRAM. |
| [`train_student.py`](train_student.py) | Full LoRA student training + held-out BEFORE/AFTER eval. `--pod_mode` opt-in for upfront vision cache on high-RAM hosts. |
| [`eval_test_predict.py`](eval_test_predict.py) | Test-set generation (vanilla / trained / teacher). |
| [`eval_compare.py`](eval_compare.py) | Sliding-window Levenshtein flip-rate + per-sport / per-qtype aggregator. |
| [`evaluate_test.py`](evaluate_test.py) | Alternative constrained-prompt evaluation path (kept for completeness; not used in the headline numbers). |
| [`_smoke_27b_int4_v2.py`](_smoke_27b_int4_v2.py) | Locked teacher smoke; primary record of the 4-pass prompt-iteration history. |
| [`_smoke_distill.py`](_smoke_distill.py) | Reference 1-sample distill loop; canonical source of the CE + sparse top-K KL loss formulation. |

### Data artifacts (under [`data/subsets/`](data/subsets/))

- `subset_train.json` — 1 600 videos / 8 937 questions
- `subset_val.json` — 480 videos / 2 743 questions
- `subset_test.json` — 480 videos / 2 793 questions
- `subset_stats.json`, `subset_test_stats.json` — per-sport / per-qtype distribution counts
- `subset_video_manifest.json`, `subset_test_video_manifest.json` — flat lists of unique video stems for the downloader

### Evaluation artifacts (in repo root)

- `train_losses.jsonl` — 35 016-step CE/KL loss curve from the final 3-epoch run
- `eval_before.jsonl`, `eval_after.jsonl` — held-out generations (current files reflect the most recent smoke run; the 100-sample canonical numbers are preserved in `eval_flip_report.md`)
- `predictions_test_vanilla.jsonl` — 1 000 stratified test predictions, pristine 0.8 B
- `predictions_test_trained.jsonl` — same 1 000, 1-epoch trained
- `predictions_test_trained_3ep.jsonl` — same 1 000, **3-epoch final** (the headline)
- `predictions_test_teacher.jsonl` — same 1 000, 27 B-AWQ-Int4 teacher reference

### Figures (under [`figures/`](figures/))

8 publication-quality PNGs (F1–F8) plus the Python scripts that generated them, all reproducible from the on-disk artifacts. See [`figures/README.md`](figures/README.md) for the per-figure agreement table.

---

## External artifacts (NOT in this repo)

| artifact | where | why not here |
|---|---|---|
| Trained LoRA adapter | **<TODO: HF Hub URL — `<ananas0/qwen3.5-0.8b-sportsqa-distill-lora>`>** | 25 MB safetensors; lives where HF Hub workflows expect models |
| Sports-QA upstream dataset (train/val/test JSON + videos) | **<TODO: original Sports-QA repo URL>** | ~110 GB total (videos + metadata); cite the original publication |
| Teacher logit cache (~11 680 .pt files, 347 MB) | **<TODO: HF Hub dataset URL if released; otherwise re-extract per RUNBOOK_RENTAL.md>** | Easier to re-extract on H100 than to host |

After you fill in the HF Hub adapter URL, also update the loading example below.

---

## How to reproduce (~ $80 budget, ~58 wall-clock hours)

```python
# Quick-start: load the trained adapter for inference
from transformers import AutoModelForImageTextToText
from peft import PeftModel
import torch

base = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16,
    device_map="cuda", trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, "ananas0/qwen3.5-0.8b-sportsqa-distill-lora")
model.eval()
```

Full reproduction (in pipeline order — see also [`vanilla_train_test_logits_handling.md`](vanilla_train_test_logits_handling.md) §15):

```bash
# 0. Environment
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pip install peft python-Levenshtein     # not in requirements.txt

# 1. Build subsets
.venv/bin/python sample_subset.py
.venv/bin/python sample_test_subset.py

# 2. Download videos (~25 GB from HF Hub)
.venv/bin/python download_subset_videos.py --repo-id <USER>/<VIDEOS_REPO> --token <HF_TOKEN>

# 3. Extract teacher logits — RENTED GPU REQUIRED (~50 hr H100-NVL, ~$67)
#    See RUNBOOK_RENTAL.md for the pod-side commands.
ssh <pod> 'cd /workspace/<project> && python -u extract_logits_subset.py --split both'

# 4. rsync logit cache back to the local box
rsync -avzs --info=progress2 -e "ssh -p <PORT>" \
    "root@<IP>:/workspace/<project>/logits_cache_subset/" "./logits_cache_subset_final/"

# 5. Train the student (~3 hr on H100 via --pod_mode, ~10 hr on a 24 GB consumer GPU)
.venv/bin/python -u train_student.py --pod_mode --epochs 3 \
    --cache_dir logits_cache_subset_final \
    --holdout_per_sport "Gym=50,volleyball=50,football=50,basketball=50,Balance_Beam=50,Floor_Exercise=50,Uneven_Bar=50,Vault=30" \
    --adapter_dir student_lora_full3epoch

# 6. Test-set evaluation (4 runs, ~6 hr total — teacher run on pod)
.venv/bin/python eval_test_predict.py --stratify_per_sport 200 --out predictions_test_vanilla.jsonl
.venv/bin/python eval_test_predict.py --stratify_per_sport 200 --adapter student_lora_full3epoch --out predictions_test_trained_3ep.jsonl
ssh <pod> '... eval_test_predict.py --model_id cyankiwi/Qwen3.5-27B-AWQ-4bit --stratify_per_sport 200 --out predictions_test_teacher.jsonl'

# 7. Flip-rate comparison + figures
.venv/bin/python eval_compare.py --before predictions_test_vanilla.jsonl --after predictions_test_trained_3ep.jsonl --report eval_test_flip_report.md
.venv/bin/python figures/F1_training_loss.py     # (and F2..F8)
```

Per-stage cost estimate (Vast.ai H100-NVL @ ~$1.33/hr):
- Logit extraction: ~50 hr × $1.33 ≈ $67
- 3-epoch student training: ~3 hr × $1.33 ≈ $4
- Teacher test eval: ~5 hr × $1.33 ≈ $7
- Smoke tests during prompt iteration: ~2 hr × $1.33 ≈ $3
- **Total: ≈ $80** (user-verify against actual Vast.ai billing; values are estimates, not measured)

---

## Engineering notes

The pipeline survived several non-trivial environment issues during development. All are documented in [`vanilla_train_test_logits_handling.md`](vanilla_train_test_logits_handling.md) §8 and [`appendix_materials.md`](appendix_materials.md) §6. Highlights:

- **PEFT ↔ gptqmodel version skew.** `dispatch_awq` imports a class that gptqmodel 7.0.0 renamed. The 0.8 B student isn't AWQ-quantized, so the dispatcher is irrelevant — monkey-patched to no-op at the top of `train_student.py`.
- **WSL2 CPU RAM cap.** Upfront vision feature caching (525 unique videos × ~80 MB ≈ 42 GB) blew past WSL2's default 15 GB limit. Mitigated with a video-major lazy-decoding training loop (RAM peak ≈ 1 video). `--pod_mode` opts back into the original upfront-cache strategy on the rented 172 GB pod.
- **WSL2/CUDA driver race.** Non-deterministic `CUDA driver error: unknown error` on the first training backward. Mitigated with `attn_implementation="eager"`, `CUDA_LAUNCH_BLOCKING=1`, and a real-shape warmup forward+backward. `--pod_mode` skips these on real Linux.
- **`mm_token_type_ids` length mismatch.** Qwen3.5's `get_rope_index` slices that tensor with the attention mask; when manually concatenating gen tokens, the type IDs must be extended too. Fixed at `train_student.py:append_gen_tokens`.

---

## Citation

```bibtex
<TODO: BibTeX entry once the report is filed>
```

## License

<TODO: pick MIT or CC-BY-4.0 once the university confirms publishing rights>
