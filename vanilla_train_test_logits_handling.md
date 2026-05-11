# Vanilla Train, Test & Logits Handling

This document is the technical reference for the knowledge-distillation pipeline built for the graduation project — from dataset construction through teacher logit extraction, student LoRA training, and held-out / test-set evaluation. Every claim is sourced from a script, an artifact, or a saved log in this repo, with markdown links so the reader can jump to the exact code or data.

---

## 1. Project objective

Distill a Qwen3.5-VL family video question-answering model from a heavy 27 B-parameter teacher (quantized to AWQ-Int4 to fit on an H100) into a 0.8 B student that runs comfortably on a consumer 24 GB GPU.

- **Dataset.** Sports-QA: short sports clips with multi-type natural-language questions (Descriptive, Temporal, Causal, Counterfactual). 8 sport buckets across the upstream set: basketball, football, volleyball, aerobic gymnastics, plus four FineGym dismount categories (vault, uneven_bar, balance_beam, floor_exercise) which we collectively reference as `fg/*`.
- **Recipe.** Recipe A — *free-generation distillation*. The teacher produces a free-form analyst-style answer with `model.generate(..., output_logits=True)`. We capture, per generated position, the gen token and the top-K logits. The student is trained to match those distributions on the same prompt + video.
- **Why Recipe A and not Recipe B (teacher-forced).** Teacher-forced distillation requires per-token gold answers; Sports-QA gold labels are 1–3-word classification tags, far shorter than the verbose teacher output we want the student to inherit. Recipe A directly transfers the teacher's *generation behaviour* — its phrasing, its event-naming habits, its temporal cues.
- **Adapter, not full fine-tune.** Student weights are frozen; we add LoRA adapters (rank 16) on the LM's attention and FFN projections. Trains in minutes-to-hours on a 3090 instead of days.

---

## 2. Dataset and subset construction

The original Sports-QA `meta-data/{train,val,test}.json` files contain ~17 k QA pairs across 8 sports, each annotated with `qa_id`, `video`, `type` (qtype), `question`, `answer`, and `ans_cls`. The full splits are too large for the rented-GPU budget, so the project trains on a **balanced subset** while testing against the full upstream test set.

Subset construction lives in [sample_subset.py](sample_subset.py) (train/val) and [sample_test_subset.py](sample_test_subset.py) (test). Output:

| split | videos | questions | per-sport balance |
|---|---:|---:|---|
| train | **1 600** | **8 937** | 200 videos × 8 sports |
| val | **480** | **2 743** | 60 videos × 8 sports |
| test | **480** | **2 793** | 60 videos × 8 sports (FineGym sub-events recovered — see §3.1) |

Detailed counts by sport and qtype are in [data/subsets/subset_stats.json](data/subsets/subset_stats.json):

```json
{
  "train": {
    "videos": 1600, "questions": 8937,
    "per_sport": {"Gym": 200, "volleyball": 200, "football": 200, "basketball": 200,
                  "Uneven_Bar": 200, "Vault": 200, "Floor_Exercise": 200, "Balance_Beam": 200},
    "per_question_type": {"Descriptive": 4614, "Temporal": 2965,
                          "Causal": 1032, "Counterfactual": 326},
    "yes_no": {"yes": 486, "no": 486, "ratio": 0.5, "dropped": 253}
  },
  "val": {
    "videos": 480, "questions": 2743, ...
  }
}
```

Two design decisions are encoded here:

1. **Per-sport quota.** Without balancing, the upstream distribution is skewed toward FineGym; sampling 200 videos per sport guarantees every sport contributes equally to training.
2. **Yes/no rebalancing.** When the answer is `yes` or `no`, we drop excess `yes` rows so the marginal is 50/50 (`dropped: 253`). Prevents the student from learning the "always-yes" prior.

Subset manifests:

- [data/subsets/subset_train.json](data/subsets/subset_train.json) — flattened `{video_id, sport, split, questions: [...]}` rows, ready for extraction.
- [data/subsets/subset_val.json](data/subsets/subset_val.json) — same structure for val.
- [data/subsets/subset_test.json](data/subsets/subset_test.json) — same structure for the curated 480-video test subset.
- [data/subsets/subset_video_manifest.json](data/subsets/subset_video_manifest.json) / [data/subsets/subset_test_video_manifest.json](data/subsets/subset_test_video_manifest.json) — flat lists of unique video stems consumed by the downloader (train+val and test respectively).
- [data/subsets/subset_stats.json](data/subsets/subset_stats.json) and [data/subsets/subset_test_stats.json](data/subsets/subset_test_stats.json) — per-split summary statistics.

### 2.1 The curated test subset (`sample_test_subset.py`)

The upstream `meta-data/test.json` (18 718 questions) collapses the four FineGym dismount events under a single `fg/` directory prefix, breaking the 8-sport parity that train/val have. [sample_test_subset.py](sample_test_subset.py) restores the parity by reading each video's Descriptive answer ("What is the video about?" → "vault" / "uneven bars" / etc.) and mapping it back to the same `Vault` / `Uneven_Bar` / `Balance_Beam` / `Floor_Exercise` labels train/val use. Sample composition (seed=42):

| sport | videos | questions |
|---|---:|---:|
| Gym (aerobic_gymnastics) | 60 | 360 |
| volleyball | 60 | 398 |
| football | 60 | 478 |
| basketball | 60 | 297 |
| Balance_Beam | 60 | 360 |
| Uneven_Bar | 60 | 360 |
| Vault | 60 | 180 |
| Floor_Exercise | 60 | 360 |
| **total** | **480** | **2 793** |

Per-question-type distribution: Descriptive 1 440, Temporal 913, Causal 320, Counterfactual 120. Yes/no answer skew: 136 yes / 229 no (ratio 0.37 — natural distribution preserved; the soft 50/50 balancer used in train/val is **not** applied to test, so honest performance is reported, not a balanced version).

**Caveats inherited from the upstream test split.** The Sports-QA test set is sparse for rare question types on FineGym sub-events:

- **Vault** has Descriptive only — zero Temporal, Causal, Counterfactual.
- **Balance_Beam, Floor_Exercise, Gym, Uneven_Bar** have Descriptive + Temporal only — zero Causal/CF.
- **basketball** has only ~6 Temporal items in the entire upstream split.

This means cells of any *sport × question_type* breakdown will be empty or extremely small in places. Reports must publish the n alongside every cell so a 0% Causal-on-Vault number is read as "no data" rather than "model failure". The curated subset honours this honestly — it does not invent samples to fill missing cells.

---

## 3. Video acquisition

Subset videos are pulled from a HuggingFace Hub dataset repo by [download_subset_videos.py](download_subset_videos.py). The script:

1. Reads `subset_video_manifest.json` to get the desired stems.
2. Lists every file in the HF repo once via `HfApi().list_repo_files`.
3. Indexes repo files by `<sport>/<stem>` keys, with a synthetic `fg/<stem>` alias that maps any `finegym_0X/<stem>` repo path to the manifest's `fg/...` IDs.
4. Downloads matched files in parallel via `hf_hub_download` (8 workers).

Local layout under `videos/`:

```
videos/aerobic_gymnastics/       v_*.avi
videos/volleyball/volleyball/    v_*.avi          # doubled directory; preserved from upstream
videos/football/                 v_*.avi
videos/basketball/               v_*.avi
videos/finegym_01/ ... finegym_05/  v_*.avi       # FineGym is split across 5 buckets
```

Path resolution is centralised in `resolve_video_path(video_key)` ([extract_logits_subset.py:84-97](extract_logits_subset.py#L84-L97), mirrored in [train_student.py:105-118](train_student.py#L105-L118) and [eval_test_predict.py](eval_test_predict.py)). It tolerates the `volleyball/volleyball/` doubling and walks all five `finegym_0X` directories when the sport prefix is `fg`.

The full extracted cache verifies all 18 718 test rows resolve to local files (`resolvable locally: 18718 / 18718 (100.0%)`).

---

## 4. Teacher: model and prompt iteration

### 4.1 Model choice

The teacher is **`cyankiwi/Qwen3.5-27B-AWQ-4bit`** ([extract_logits_subset.py:46](extract_logits_subset.py#L46)). The 27 B AWQ-Int4 build:

- Fits on a single H100 with margin (~14 GB weights, ~8 GB activation budget).
- The original GPTQ-Marlin quant rejected one of Qwen3.5's 48-dim layers (`out_features 48 must be divisible by 64`); AWQ-Int4 has no such constraint.
- Compressed-tensors decompression OOMs on a 3090, which is why teacher extraction runs only on the rental pod.

Smoke tests on smaller siblings ([_smoke_27b.py](_smoke_27b.py), [_smoke_4b.py](_smoke_4b.py), [_smoke_9b.py](_smoke_9b.py), [_smoke_27b_int4_v2.py](_smoke_27b_int4_v2.py)) sanity-checked that `enable_thinking=False` is required at every size — the default flips between sizes and silently produces reasoning-tagged output that confuses downstream parsers.

### 4.2 Prompt iteration

The prompt converged through four passes:

1. **Constrained classifier prompt** ("output only the label"). The 0.8 B and 4 B emit single-word answers, but the 27 B treats it as instruction noise and collapses to `"basketball"` regardless of clip content. Ground-truth coverage for non-trivial questions: ~10 %.
2. **Analyst v1.** Free-form prose. Teacher quality jumped, but verdicts on outcome questions ("did the spike land?") came from reading scoreboards, not from observing the action. Scoreboards lag the action by 1–3 seconds and may not even update within the clip.
3. **Analyst v2.** Added the explicit rule: *evidence must come from the action itself, not from interface elements*. This fixed the scoreboard shortcut.
4. **Analyst v3 (locked).** Added a brevity nudge ("around 300–500 tokens, brevity over rumination"). The v2 wording produced 1 k-token monologues; v3 keeps generations within `max_new_tokens=256` without truncating mid-sentence.

The locked prompt is verbatim at [extract_logits_subset.py:53-79](extract_logits_subset.py#L53-L79) and re-imported by [train_student.py:74-100](train_student.py#L74-L100) and [eval_test_predict.py](eval_test_predict.py). Critical: the **same prompt is used at extraction, training, and test time** — a mismatch would mis-anchor the KL signal.

### 4.3 Sampling configuration

Locked at [extract_logits_subset.py:172-179](extract_logits_subset.py#L172-L179):

```python
out = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.5,
    top_p=0.9,
    top_k=20,
    min_p=0.0,
    repetition_penalty=1.1,
    output_logits=True,
    return_dict_in_generate=True,
)
```

- `temperature=0.5` was empirically the lowest where the teacher still distinguished answers across questions on the same video without going thesaurus-mode.
- `repetition_penalty=1.1` (originally 1.4) was reduced after observing degenerate "synonym chains" at higher penalties.
- `output_logits=True` (not `output_scores`) is critical — `output_scores` returns post-warper logits with `-inf` at filtered positions, which is useless for distillation.

### 4.4 Vision-token capping

Vision messages are constructed at [extract_logits_subset.py:102-110](extract_logits_subset.py#L102-L110):

```python
{"type": "video", "video": str(video_path), "fps": 2.0,
 "max_frames": 32, "max_pixels": 256 * 28 * 28}
```

- `fps=2.0` over 12–15 s clips → ~24–30 frames before capping.
- `max_frames=32` is a hard ceiling; without it, an internal default of `fps=24` produced ~360 frames and 12× the prompt length.
- `max_pixels=256·28·28` (≈ 200 k pixels per frame) keeps per-frame vision-token count bounded.

The result, regardless of source clip length, is roughly 600–900 prompt tokens per question.

---

## 5. Teacher logit extraction

The extraction script is [extract_logits_subset.py](extract_logits_subset.py). It is the most expensive step in the pipeline and runs on a rented GPU pod (see [RUNBOOK_RENTAL.md](RUNBOOK_RENTAL.md)).

### 5.1 What gets saved per question

For each `(video, question)` pair the script saves a `.pt` file at `logits_cache_subset/qa_<qa_id:07d>.pt` ([extract_logits_subset.py:373-393](extract_logits_subset.py#L373-L393)):

| key | dtype | shape | meaning |
|---|---|---|---|
| `gen_tokens` | int64 | `(T,)` | Token IDs the teacher emitted. |
| `topk_values` | float16 | `(T, 32)` | Raw pre-warper logits at the top-32 vocab indices each step. |
| `topk_indices` | int32 | `(T, 32)` | The vocab indices those logits correspond to. |
| `gen_text` | str | – | `processor.batch_decode(gen_tokens, skip_special_tokens=True)`. |
| `qa_id` / `video_id` / `split` / `sport` / `question_type` / `question` / `answer` | – | – | Metadata copied from the subset row. |
| `seq_len` | int | – | `T`, the number of generated tokens. |
| `topk_mass` | float | – | Per-step softmax probability that the top-32 captures (mean across positions). |
| `chosen_in_topk` | float | – | Fraction of positions where the actually-emitted token is inside the top-32. The script forces it to 1.0 by replacing the lowest-probability top-K slot with the gen token if missing ([extract_logits_subset.py:194-199](extract_logits_subset.py#L194-L199)). |
| `wall_time_s` | float | – | Per-question generation cost. |

Storage cost: 25–55 KB per `.pt`. At 5282 samples, the on-disk cache is ~270 MB.

### 5.2 Why `output_logits` not `output_scores`

`output_scores=True` returns the post-warper distribution: top-k filtering and temperature have already been applied, and most positions outside the kept-k are `-inf`. Distilling against `-inf` columns destroys the gradient. `output_logits=True` returns the raw pre-warper distribution — the teacher's natural beliefs over the full vocab — which is what we want the student to mimic.

### 5.3 Per-video grouping

Each video has ~5–6 questions in the subset. The extraction loop groups by `video_id` and decodes the video once per group ([extract_logits_subset.py:317-345](extract_logits_subset.py#L317-L345)):

```python
_, cached_video_inputs = process_vision_info(
    build_messages(video_path, qs[0][0]["question"])
)
for r, _vp in video_bar:
    rec = extract_one(model, processor, video_path, r["question"], cached_video_inputs)
```

This skips ~80 % of redundant decord work.

### 5.4 Resumable manifest

Every successful sample is appended to [logits_cache_subset.manifest.jsonl](logits_cache_subset.manifest.jsonl). Re-running `extract_logits_subset.py` reads `done_ids` from the manifest and skips them ([extract_logits_subset.py:220-234](extract_logits_subset.py#L220-L234)). Errors (decode failure, empty generation, runtime exception) are also recorded with an `"error"` field so they aren't silently retried.

Manifest row example:

```json
{"qa_id": 3248, "video_id": "aerobic_gymnastics/v_CoAiZeQaVCc_c025_00",
 "split": "train", "sport": "Gym", "question_type": "Descriptive",
 "seq_len": 240, "topk_mass": 0.9961, "chosen_in_topk": 1.0,
 "wall_time_s": 25.65, "path": "qa_0003248.pt"}
```

Average sanity metrics across the 5282-sample cache: `topk_mass ≈ 0.997`, `chosen_in_topk = 1.000` (the latter forced — see above).

### 5.5 Live progress UX

Three nested `tqdm` bars are wired up at [extract_logits_subset.py:312-330](extract_logits_subset.py#L312-L330):

- position 0 — **overall**, persistent for the whole run, displays `ok / err / mass / chosen / rate / eta_h`.
- position 1 — **current sport**, resets per sport, hidden when complete.
- position 2 — **current video**, resets per video, hidden when complete.

This makes a 50+ hour run readable from a single SSH session.

### 5.6 Wall time

On the rental H100 + AWQ-Int4: ~17 s per question. With 256 max-new-tokens autoregressive decode at ~15 tok/s on AWQ-Int4 27 B, this is the LM decode loop, not video decoding (~0.4 s/video).

### 5.7 Current cache state

As of writing the local `.pt` cache contains **5 282 samples** across 5 sport buckets; manifest in [logits_cache_subset.manifest.jsonl](logits_cache_subset.manifest.jsonl). Distribution (verified by scanning the `.pt` files directly):

| sport | n |
|---|---:|
| football | 1 886 |
| volleyball | 1 665 |
| Gym (aerobic_gymnastics) | 1 549 |
| basketball | 182 |
| **total** | **5 282** |

Splits in cache: `train` 4 083, `val` 1 199. Question types: Descriptive 2 336, Temporal 1 527, Causal 1 076, Counterfactual 343. Extraction continues incrementally; the FineGym (`fg/*`) bucket has not yet been extracted.

---

## 6. Logit storage and transport

### 6.1 On-disk format

Each sample is a single `torch.save`-d dict (see schema in §5.1). Per-file size depends on `seq_len`; the largest are ~55 KB (256-token generations). `torch.load(path, weights_only=False)` deserialises in ~5 ms on local SSD.

### 6.2 Transport off the rental pod

After extraction, `.pt` files are pulled from the pod with rsync. The project root contains a space ("Granduation Project"), which trips up the standard rsync invocation — the local shell eats the backslash before the remote shell sees the path. Working command pattern:

```bash
rsync -avzs --info=progress2 \
  -e "ssh -p <PORT> -o StrictHostKeyChecking=no" \
  "root@<IP>:/workspace/Granduation Project/logits_cache_subset/" \
  "/home/accy/Granduation Project/logits_cache_subset/"
```

Critical flags:

- `-s` (`--protect-args`) — tells the remote rsync helper to receive paths as-is rather than re-shell-evaluating them. Without this, the space-in-pathname is split.
- `-z` — compresses the per-file transfer; logit `.pt` files are mostly fp16 + int32 and compress moderately.
- `--info=progress2` — single rolling progress line instead of one line per file (useful for thousands of small files).

Re-running the same command is idempotent (rsync only transfers new/changed files) so the loop "extract on pod → rsync to WSL → train more on WSL" works during long-running extractions.

### 6.3 Cross-version recovery

The manifest is the *index* but the `.pt` files are the *source of truth* for training. The training script ([train_student.py:140-165](train_student.py#L140-L165)) was rewritten to scan `.pt` files directly, because the manifest sometimes lags after partial rsyncs:

```python
def load_available_samples() -> list[dict]:
    rows = []
    for pt in tqdm(sorted(CACHE_DIR.glob("*.pt")), desc="scanning .pt cache"):
        d = torch.load(pt, map_location="cpu", weights_only=False)
        ...
        rows.append({"qa_id": int(d["qa_id"]), "video_id": d["video_id"], ...})
    return rows
```

3 253 file scans take ~0.6 s on a fast SSD.

---

## 7. Student training pipeline

[train_student.py](train_student.py) is the full training script. End-to-end it: discovers samples, splits a held-out set, runs a BEFORE-eval, applies LoRA + freezes the visual encoder, runs the training loop, runs an AFTER-eval, and writes a side-by-side comparison report.

### 7.1 Architecture

| component | choice | rationale |
|---|---|---|
| student | `Qwen/Qwen3.5-0.8B` | Same family as the teacher; tokenizers verified identical (parity check at [_smoke_distill.py:54-57](_smoke_distill.py#L54-L57)). |
| dtype | bf16 | 3090 supports bf16 at full speed; fp16 had range issues with KL gradients. |
| LoRA rank | r=16, alpha=32, dropout=0.05 | ~6.4 M trainable params (0.74 % of model). Enough capacity for stylistic transfer without over-fitting on the partial cache. |
| LoRA targets | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | All attention + FFN linears in the LM. Vision encoder is excluded (see 7.3). |
| optimizer | `AdamW(lr=1e-4, wd=0.01)` on LoRA params only | Higher than full-fine-tune lr because LoRA params are zero-initialised and need to move farther. |
| batch size | 1 | Variable sequence lengths across samples make batching a complication that doesn't help at this scale. |
| epochs | 1 (validation), 3 (planned full) | First epoch captures most of the distillation gain; epoch 2–3 close the residual. |

### 7.2 Loss formulation

Verbatim from [_smoke_distill.py:100-104](_smoke_distill.py#L100-L104), reused inside [train_student.py:597-602](train_student.py#L597-L602):

```python
TEMPERATURE = 2.0
KL_WEIGHT = 0.7

s_on_topk = student_logits.gather(-1, teacher_topk_indices)        # (T, 32)
p_t = (teacher_topk_values / TEMPERATURE).softmax(-1)              # softened teacher
log_q_s = (s_on_topk / TEMPERATURE).log_softmax(-1)                # softened student
kl = F.kl_div(log_q_s, p_t, reduction="batchmean") * (TEMPERATURE ** 2)
ce = F.cross_entropy(student_logits, teacher_gen_tokens)
loss = KL_WEIGHT * kl + (1 - KL_WEIGHT) * ce
```

- **CE term** drives the student to assign high probability to the teacher's actual chosen token at each position.
- **KL term** drives the student's *full top-32 distribution shape* to match the teacher's, weighted by Hinton temperature so less-likely candidates carry meaningful gradient.
- **`× T²`** rescales the KL gradient to be comparable to CE at the same temperature.
- **Sparse top-K** rather than full-vocab KL because storing the full 248 k-vocab teacher distribution per token would cost ~2 GB per question; the top-32 captures `topk_mass ≈ 99.7 %` of the probability mass.

### 7.3 Vision encoder is frozen end-to-end

Two-step lock to be defensive ([train_student.py:400-421](train_student.py#L400-L421)):

```python
def freeze_visual(model):
    for name, mod in model.named_modules():
        if any(k in name for k in ("visual", "vision_tower", "vision_model")):
            for p in mod.parameters(recurse=False):
                if p.requires_grad: p.requires_grad = False
    ...

def freeze_lora_under_visual(peft_model):
    for name, p in peft_model.named_parameters():
        if "lora_" in name and any(k in name for k in ("visual", "vision_tower", "vision_model")):
            p.requires_grad = False
```

Freezing visual params *before* PEFT injection prevents gradient flow through the encoder; freezing LoRA-named parameters *after* PEFT injection catches any LoRA adapters that PEFT would have placed on visual `q_proj` / `k_proj` / `v_proj` (if such names happened to exist there). The student never updates its visual representation — only how its language model talks about what it sees.

### 7.4 Teacher-forcing

The student is teacher-forced against the recorded `gen_tokens`. The forward pass uses a manually-concatenated sequence of `[prompt_input_ids, gen_tokens]` ([train_student.py:244-263](train_student.py#L244-L263)):

```python
def append_gen_tokens(inputs, gen_tokens):
    out = dict(inputs)
    gen = gen_tokens.unsqueeze(0).to(device).long()
    out["input_ids"] = torch.cat([inputs["input_ids"], gen], dim=1)
    out["attention_mask"] = torch.cat(
        [inputs["attention_mask"], torch.ones_like(gen)], dim=1
    )
    if "mm_token_type_ids" in out and out["mm_token_type_ids"] is not None:
        tt = out["mm_token_type_ids"]
        pad = torch.zeros(tt.shape[0], gen.shape[1], dtype=tt.dtype, device=tt.device)
        out["mm_token_type_ids"] = torch.cat([tt, pad], dim=1)
    return out
```

Then student logits at `[prompt_len-1 : prompt_len-1 + T]` are exactly the per-step predictions of each gen token, aligned with the teacher's recorded top-K.

The `mm_token_type_ids` extension is non-obvious: Qwen3.5-VL's `compute_3d_position_ids` slices `input_token_type[attention_mask.bool()]` and crashes with an `IndexError` if the two have different lengths. Appending zeros (text type) for the gen tokens fixes it.

### 7.5 Iteration order — video-major

The script iterates **video-major**: at the start of each epoch the unique videos are shuffled; for each video, all of its ~5–6 questions are processed consecutively before moving on.

```python
for epoch in range(args.epochs):
    video_ids = list(by_video.keys()); rng.shuffle(video_ids)
    for vid in video_ids:
        qs = by_video[vid][:]; rng.shuffle(qs)
        _, video_inputs = process_vision_info(build_messages(video_path_of[vid], "x"))
        for r in qs:
            ...                              # forward + loss + backward + step
        del video_inputs
        torch.cuda.empty_cache()
```

This trades sample-level shuffling for memory bounds: only **one decoded video** is in CPU RAM at a time (~80 MB), instead of all ~525 (~42 GB). The reasons we made this trade are documented in §8.3.

Within-video gradient correlation is mild because (a) videos are still globally shuffled per-epoch, (b) AdamW's running averages absorb 5-step micro-correlation, (c) distillation against a fixed teacher distribution is well-conditioned.

### 7.6 Per-step cost

| stage | typical cost |
|---|---|
| video decode (decord, fps=2, max_frames=32) | ~0.4 s |
| processor + first forward pass (prompt 800–900 tokens + 256 gen tokens) | ~0.6 s |
| backward + AdamW step on LoRA params | ~0.4 s |
| **total step** | **~1.0 s** with `CUDA_LAUNCH_BLOCKING=0` |
| total step under CLB=1 (current default) | ~3.7 s |

For 3 153 train samples × 1 epoch the run takes ~3 hr; ×3 epochs ~9 hr.

### 7.7 What gets saved

After training, [train_student.py:626-638](train_student.py#L626-L638) writes:

- [student_lora/](student_lora/) — `adapter_config.json` + `adapter_model.safetensors` (~25 MB).
- [train_losses.jsonl](train_losses.jsonl) — one row per step: `{epoch, step, ce, kl, loss, qa_id}`.

---

## 8. Engineering issues and fixes

This section documents issues that took non-trivial diagnosis. They are all WSL2- or version-skew-driven; none are flaws in the methodology.

### 8.1 PEFT ↔ gptqmodel version skew

**Symptom.** `get_peft_model(...)` raises:
```
ImportError: cannot import name 'AwqGEMMQuantLinear' from
'gptqmodel.nn_modules.qlinear.gemm_awq'. Did you mean: 'AwqGEMMLinear'?
```

**Diagnosis.** PEFT's `dispatch_awq` is called for every target module during LoRA injection. It unconditionally tries to import `AwqGEMMQuantLinear`, but the installed `gptqmodel==7.0.0` renamed that class to `AwqGEMMLinear`.

**Fix.** Monkey-patch `peft.tuners.lora.model.dispatch_awq` to a no-op. The student is plain bf16 (not AWQ), so this dispatcher is irrelevant to us. Code at [train_student.py:44-54](train_student.py#L44-L54), [eval_test_predict.py](eval_test_predict.py), [evaluate_test.py:7-15](evaluate_test.py#L7-L15):

```python
import peft.tuners.lora.model as _peft_lora_model
def _noop_dispatch_awq(target, adapter_name, config, **kwargs):
    return None
_peft_lora_model.dispatch_awq = _noop_dispatch_awq
```

**Residual cost.** None.

### 8.2 `mm_token_type_ids` length mismatch

**Symptom.**
```
IndexError: The shape of the mask [1106] at index 0 does not match the
shape of the indexed tensor [850] at index 0
```
Triggered inside `Qwen3_5Model.compute_3d_position_ids → get_rope_index`.

**Diagnosis.** `processor(...)` returns `mm_token_type_ids` with the same length as `input_ids` *before* we manually append `gen_tokens`. After `append_gen_tokens`, `input_ids` and `attention_mask` are length 1106 but `mm_token_type_ids` is still 850. Internal slicing `input_token_type[attention_mask[batch_idx].bool()]` crashes.

**Fix.** Extend `mm_token_type_ids` with zeros (text type) in `append_gen_tokens` ([train_student.py:255-263](train_student.py#L255-L263)).

**Residual cost.** None.

### 8.3 WSL2 RAM cap kills upfront vision cache

**Symptom.** Bash reports `Killed` (signal 9, OS OOM-killer) at video ~165–174 / 525 during the upfront vision-cache build. Windows Task Manager shows pagefile at 100% active time, ~13 GB resident before kill.

**Diagnosis.** The original training loop pre-decoded all unique videos into CPU RAM:
- 525 unique videos × ~80 MB per decoded `pixel_values_videos` ≈ **~42 GB** of CPU RAM.
- WSL2 default memory cap is 50 % of host RAM. On 31.2 GB host this is ~15 GB, hard ceiling.
- At 174 videos × 80 MB ≈ 13.9 GB the OOM-killer fires.

**Fix.** Switched to **video-major lazy decoding** ([train_student.py:567-630](train_student.py#L567-L630)). Each video is decoded inside the training loop and dropped before the next one starts. Memory peak: one decoded video, ~80 MB. Scales to arbitrarily many unique videos.

**Residual cost.** Each video is decoded once per epoch instead of once per run. With 525 videos × 3 epochs × 0.4 s ≈ 10 min of extra decode time spread across the run; negligible against multi-hour training.

### 8.4 WSL2/CUDA "driver error" on first backward

**Symptom.**
```
RuntimeError: CUDA driver error: unknown error
  File ".../torch/autograd/__init__.py", line 381, in backward
  File ".../torch/autograd/graph.py", line 869, in _engine_run_backward
```
Fires inside the autograd engine on the *first* training backward of a fresh process. Non-deterministic — sometimes the same code succeeds, sometimes fails. nvidia-smi reports a healthy idle GPU.

**Diagnosis.** Not OOM (VRAM peak 10 GB on a 24 GB card), not the model code (the same forward succeeds in a standalone repro). Layered evidence pointed to a **WSL2 + CUDA driver race** specific to the first backward against the LoRA-adapted graph. Specific shapes (volleyball-clip-sized prompts) trigger it more often than smaller (Gym-sized) ones.

**Fixes layered in defensively:**

1. **`gc.collect() + torch.cuda.synchronize() + empty_cache()` between BEFORE-eval and training** ([train_student.py:531-540](train_student.py#L531-L540)). Drops eval-era allocator fragmentation before training starts.
2. **`attn_implementation="eager"`** at model load ([train_student.py:516-519](train_student.py#L516-L519)). Bypasses Qwen3.5-VL's default SDPA attention path, which triggers the race more reliably.
3. **`os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")`** at the top of the script. Forces synchronous CUDA op submission, surfacing latent races as deterministic failures (which then can be retried).
4. **Real-shape warmup forward+backward on a Gym sample before the main loop** ([train_student.py:567-595](train_student.py#L567-L595)). The first backward of the process is then deterministically against a graph shape that has never failed; the first volleyball backward becomes the *second* backward and is past the danger zone.

**Residual cost.** With CLB=1 the per-step time goes from ~1.0 s to ~3.7 s (~3.5× slowdown). For training (~3 153 steps) this is acceptable.

### 8.5 CLB throughput cost in inference

**Symptom.** Inference (forward-only, no backward) was running at ~33 s/q on the test set when it should run at ~6–8 s/q.

**Diagnosis.** [eval_test_predict.py](eval_test_predict.py) inherited `CUDA_LAUNCH_BLOCKING=1` from the training-script template. Inference has no backward, so the race that motivated CLB=1 doesn't apply.

**Fix.** Removed the env default in [eval_test_predict.py](eval_test_predict.py); CLB can still be set externally if a transient error appears. Per-question time dropped from 33 s → 6.7 s (5× speedup).

**Residual cost.** None observed.

---

## 9. Holdout evaluation (BEFORE / AFTER)

The training script carves a held-out set from the available samples *before* training, runs the pristine 0.8 B on it, trains, then re-runs the trained student on the same held-out and writes a side-by-side comparison.

### 9.1 Holdout construction

`stratified_holdout_by_sport()` ([train_student.py:194-228](train_student.py#L194-L228)) takes a per-sport budget and stratifies internally by question type. The validation run used `--holdout_per_sport "Gym=50,volleyball=50"`:

```
holdout = 100 samples
  Gym 50:        25 Descriptive + 25 Temporal       (Gym lacks Causal / Counterfactual)
  volleyball 50: 13 Descriptive + 12 Temporal + 13 Causal + 12 Counterfactual
train = 3153 (the rest of the 3253 .pt files at the time of the run)
```

### 9.2 Generation config matches the teacher's

[train_student.py:run_generation_eval](train_student.py#L271) regenerates the student's own free-form answer on each holdout question using the same `do_sample=True, T=0.5, top_p=0.9, top_k=20, repetition_penalty=1.1` the teacher used. This guarantees the BEFORE / AFTER comparison is apples-to-apples.

### 9.3 Output schema

Per-row jsonl in [eval_before.jsonl](eval_before.jsonl) and [eval_after.jsonl](eval_after.jsonl):

```json
{"qa_id": 28, "video_id": "...", "question": "...", "question_type": "Descriptive",
 "gold": "...", "teacher_gen_text": "...", "student_text": "..."}
```

`teacher_gen_text` carries the recorded teacher generation for context (only populated for held-out training rows; empty string for test-set predictions).

### 9.4 Results — held-out (n = 100)

From [eval_flip_report.md](eval_flip_report.md):

| metric | before | after | Δ |
|---|---:|---:|---:|
| correct rate (gold-fuzzy ≥ 0.80) | 0.240 | 0.400 | **+0.160 pp** |
| avg gold best-window sim | 0.599 | 0.781 | +0.182 |
| avg teacher-text sim | 0.190 | 0.323 | +0.133 |

| bucket | n | share |
|---|---:|---:|
| **wrong → correct (FIX)** | 18 | 18.0 % |
| correct → wrong (REGRESS) | 2 | 2.0 % |
| correct → correct | 22 | 22.0 % |
| wrong → wrong | 58 | 58.0 % |

By question type:

| qtype | n | correct_after |
|---|---:|---:|
| Counterfactual | 12 | **91.7 %** |
| Descriptive | 38 | 73.7 % |
| Temporal | 37 | 2.7 % |
| Causal | 13 | 0.0 % |

By sport:

| sport | n | correct_after |
|---|---:|---:|
| volleyball | 50 | 46.0 % |
| Gym | 50 | 34.0 % |

The 18:2 fix-to-regress ratio and 16-point absolute accuracy gain validated the pipeline.

---

## 10. Test-set evaluation

The held-out comes from the same training pool; for true unseen-data validation we evaluate against the curated [data/subsets/subset_test.json](data/subsets/subset_test.json) (480 videos / 2 793 questions, 8-sport parity with train+val, FineGym sub-events recovered — see §2.1).

### 10.1 The script — [eval_test_predict.py](eval_test_predict.py)

Designed to be run twice — once for vanilla 0.8 B, once with the LoRA adapter — and the two output jsonls fed into [eval_compare.py](eval_compare.py). Schema is identical to the held-out eval (with an extra `sport` field carrying the curated 8-sport taxonomy), so no new comparison code is needed.

Key knobs:

```
# canonical test eval against the curated 480-video subset (recommended):
python eval_test_predict.py \
    --subset_file data/subsets/subset_test.json \
    --out predictions_test_vanilla.jsonl

python eval_test_predict.py \
    --subset_file data/subsets/subset_test.json \
    --adapter student_lora \
    --out predictions_test_trained.jsonl

# alternative: ad-hoc stratified sample directly from upstream meta-data/test.json
python eval_test_predict.py --stratify_per_sport 200 --out X.jsonl   # 1000 q

# debug
python eval_test_predict.py --limit 50 --out smoke.jsonl
```

When `--subset_file` is set, the per-question rows carry the curated `sport` field (e.g. `Vault`, `Uneven_Bar`) instead of the raw video-id prefix `fg`, so per-sport eval breakdowns from [eval_compare.py](eval_compare.py) are correctly grouped under the 8-sport taxonomy.

### 10.2 Subset sizing

The full 18 718-question upstream test split takes ~35 hr per model on the 3090 (~70 hr for both), which is out of budget locally. Two options sized to fit:

- **Curated 8-sport subset** (`subset_test.json`, 2 793 Q) — reproducible, mirrors train/val taxonomy, ~5 hr per model. Recommended for final reported numbers.
- **Ad-hoc 5-bucket stratification** (`--stratify_per_sport 200` → 1 000 Q) — quick smoke (~1.9 hr per model), used for the historical 1000-Q numbers below.

### 10.3 Results — historical 1 000-Q ad-hoc run

The numbers below come from the *initial* ad-hoc evaluation that used `--stratify_per_sport 200 --split test.json` directly against the upstream `meta-data/test.json` (5 raw buckets, with `fg/*` lumped together). They predate `subset_test.json` and are kept here as the validated headline numbers used to demonstrate distillation effect. A future run against the curated 480-video subset will produce comparable but more taxonomy-faithful numbers.

From [eval_test_flip_report.md](eval_test_flip_report.md):

| metric | vanilla | trained | Δ |
|---|---:|---:|---:|
| **correct rate (gold-fuzzy ≥ 0.80)** | **22.7 %** | **36.5 %** | **+13.8 pp (+61 % rel.)** |
| avg gold best-window sim | 0.546 | 0.712 | +0.166 |

| bucket | n | share |
|---|---:|---:|
| **wrong → correct (FIX)** | **174** | 17.4 % |
| correct → wrong (REGRESS) | 36 | 3.6 % |
| correct → correct | 191 | 19.1 % |
| wrong → wrong | 599 | 59.9 % |

By sport (200 questions each):

| sport | n | fix | regress | correct_after | Δ |
|---|---:|---:|---:|---:|---:|
| volleyball | 200 | 43 | 3 | 41.5 % | +20.0 pp |
| aerobic_gymnastics | 200 | 44 | 5 | 32.5 % | +19.5 pp |
| **fg** (vault / UB / BB / FX) | 200 | 40 | 6 | 38.5 % | +17.0 pp |
| football | 200 | 33 | 3 | 33.0 % | +15.0 pp |
| basketball | 200 | 14 | 19 | 37.0 % | **−2.5 pp** |

By question type:

| qtype | n | vanilla | trained |
|---|---:|---:|---:|
| Counterfactual | 150 | 46.7 % | **72.0 %** |
| Descriptive | 391 | 34.8 % | **57.8 %** |
| Temporal | 306 | 6.9 % | 9.8 % |
| Causal | 153 | 0.0 % | 0.7 % |

### 10.4 Cross-validation against the held-out

| metric | held-out (n = 100) | test (n = 1 000) |
|---|---:|---:|
| accuracy before | 24.0 % | 22.7 % |
| accuracy after | 40.0 % | 36.5 % |
| Δ | +16.0 pp | **+13.8 pp** |
| FIX rate | 18 % | 17.4 % |
| REGRESS rate | 2 % | 3.6 % |

The held-out gains hold up almost identically on the larger, truly-unseen test set. The slight drop (16 → 13.8 pp) is the expected cost of in-distribution-holdout vs true-test rigor.

### 10.5 Notable findings

1. **Generalization to fg/* despite zero training samples on it.** The student never saw FineGym dismounts during distillation (the cache had no `fg/*` extracts at training time), yet test accuracy on `fg/*` jumped +17 pp. That's evidence the distilled *style* — recognising "this is a gymnastics moment", emitting timestamped phrases, naming events — transfers to unseen gymnastics events even without sport-specific training data.
2. **Basketball slightly regresses (−2.5 pp).** Vanilla had unusually high baseline (39.5 %) because basketball gold labels include verbose phrases like "basketball 2-point shot" that the base 0.8 B already paraphrases adequately. Training on Gym + volleyball + football styles slightly drifts basketball-specific patterns.
3. **Temporal / Causal stuck near baseline.** Their gold answers are FineGym-taxonomy strings ("aerobic gymnastics straight jump", "explosive push up") which the **27 B teacher itself does not emit**. Distillation can't transfer knowledge the teacher doesn't have.

---

## 11. Comparison metric: fuzzy best-window Levenshtein

The first version of the eval used strict substring matching: `gold.lower() in student_text.lower()`. That undercounted obvious wins like the student writing `"aerobics gymnastics"` instead of `"aerobic gymnastics"` (extra `s`). To fix this, [eval_compare.py](eval_compare.py) uses a **sliding-window Levenshtein**:

```python
def best_window_sim(text: str, gold: str) -> float:
    text, gold = text.lower(), gold.lower()
    if len(text) < len(gold):
        return Levenshtein.ratio(text, gold)
    g = len(gold); best = 0.0
    for i in range(len(text) - g + 1):
        s = Levenshtein.ratio(text[i:i+g], gold)
        if s > best:
            best = s
        if best >= 0.999:
            break
    return best
```

Correctness criterion: `best_window_sim ≥ 0.80`. The threshold catches typos and 1-character drifts; it rejects "competition routine" as a match for "aerobic gymnastics" (sim ≈ 0.30).

Each (before, after) pair is bucketed into one of four states:

```python
def classify(sim_b, sim_a, threshold):
    correct_b, correct_a = sim_b >= threshold, sim_a >= threshold
    if not correct_b and correct_a:   return "fix"          # wrong → correct
    if correct_b and not correct_a:   return "regress"      # correct → wrong
    if correct_b and correct_a:       return "persist_ok"   # correct → correct
    return "persist_bad"                                    # wrong → wrong
```

The script outputs aggregate counts, per-sport and per-qtype breakdowns, and a markdown listing of every flip-to-correct (sorted by similarity gain) and every regression. Used identically for both the held-out and test-set evals.

---

## 12. Loss curves

[train_losses.jsonl](train_losses.jsonl) records every step of the run: `{epoch, step, ce, kl, loss, qa_id}`. 3 153 rows for the 1-epoch validation run.

| window | mean CE | mean KL |
|---|---:|---:|
| step 0 (single sample) | 2.471 | 2.512 |
| first 50 steps | 1.679 | 1.753 |
| last 50 steps | **1.133** | **0.938** |

Total loss is `0.7·KL + 0.3·CE`. Both terms decrease monotonically (with normal stochastic noise) over the run; gradient clipping at norm 1.0 and AdamW prevent any divergence. No NaN, no exploding gradients. The KL term drops faster than CE in absolute terms — the student is learning the teacher's distribution shape (KL) faster than it is committing to the exact next-token argmax (CE), which is the expected behaviour for distillation.

---

## 13. What more training would do (predictions)

Given the above results, here are calibrated predictions for the *next* run — full extraction (~12 k samples once `fg/*` is also extracted) + 3 epochs:

| qtype | current (1 epoch / 3 k) | predicted (3 epochs / 12 k) | rationale |
|---|---:|---:|---|
| Descriptive | 57.8 % | 65–70 % | Approaching teacher ceiling; more diverse training prose helps. |
| Counterfactual | 72.0 % | 75–80 % | Already strong; gains are diminishing. |
| Temporal | 9.8 % | 10–14 % | Stuck — teacher's blind spot for FineGym taxonomy. |
| Causal | 0.7 % | 1–3 % | Same. |
| **overall** | **36.5 %** | **44–50 %** | Driven by `fg/*` and basketball going from under-trained to in-training-distribution. |

**Diminishing returns past epoch 2.** At LoRA r=16 the adapter saturates; rank-32 might keep gaining for longer at the cost of more VRAM during training.

**Highest-leverage change** (not pure scale): mix a **CE-against-gold** term alongside the existing teacher-KL loss. The student would learn both teacher style AND the FineGym labels directly. Expected effect: Temporal / Causal accuracy from 0–10 % to 30–40 %, which moves the overall metric more than another full epoch will. This change is ~10 lines in `train_student.py`.

---

## 14. File inventory

Relative paths to every artifact referenced in this document.

### Scripts

| path | purpose |
|---|---|
| [sample_subset.py](sample_subset.py) | Build the balanced 200-vid-per-sport train + 60-per-sport val subsets. |
| [sample_test_subset.py](sample_test_subset.py) | Build the curated 60-vid-per-sport test subset (8-sport parity via FineGym recovery). |
| [download_subset_videos.py](download_subset_videos.py) | Pull subset videos from HF-Hub. |
| [extract_logits_subset.py](extract_logits_subset.py) | Recipe-A teacher logit extraction (rented GPU). |
| [train_student.py](train_student.py) | Full training script + held-out BEFORE/AFTER eval. |
| [eval_test_predict.py](eval_test_predict.py) | Test-set generation script (vanilla + trained variants). |
| [eval_compare.py](eval_compare.py) | Fuzzy-similarity flip-rate analysis. |
| [evaluate_test.py](evaluate_test.py) | Constrained-classification-prompt eval (alternative regime, kept for completeness). |
| [_smoke_27b_int4_v2.py](_smoke_27b_int4_v2.py) | Locked teacher smoke + prompt iteration. |
| [_smoke_27b.py](_smoke_27b.py), [_smoke_4b.py](_smoke_4b.py), [_smoke_9b.py](_smoke_9b.py) | Per-size smoke tests during teacher selection. |
| [_smoke_distill.py](_smoke_distill.py) | Reference 1-sample distill loop, source of the loss formulation. |

### Configs / metadata

| path | content |
|---|---|
| [meta-data/train.json](meta-data/train.json), [meta-data/val.json](meta-data/val.json), [meta-data/test.json](meta-data/test.json) | Sports-QA originals (train ~ 17 k Q, val, test). |
| [data/subsets/subset_train.json](data/subsets/subset_train.json) | Balanced training subset (1 600 vids / 8 937 Q). |
| [data/subsets/subset_val.json](data/subsets/subset_val.json) | Balanced val subset (480 vids / 2 743 Q). |
| [data/subsets/subset_test.json](data/subsets/subset_test.json) | Curated test subset (480 vids / 2 793 Q, 8-sport parity). |
| [data/subsets/subset_stats.json](data/subsets/subset_stats.json) | Train+val subset distribution counts. |
| [data/subsets/subset_test_stats.json](data/subsets/subset_test_stats.json) | Test subset distribution counts. |
| [data/subsets/subset_video_manifest.json](data/subsets/subset_video_manifest.json) | Train+val video stems list, consumed by the downloader. |
| [data/subsets/subset_test_video_manifest.json](data/subsets/subset_test_video_manifest.json) | Test video stems list, consumed by the downloader. |

### Data caches

| path | content (current) |
|---|---|
| `logits_cache_subset/` | 5 282 `.pt` files, one per extracted sample (~270 MB total). |
| [logits_cache_subset.manifest.jsonl](logits_cache_subset.manifest.jsonl) | One row per extracted sample, append-only, used for resume + sanity stats. |
| `videos/` | Local subset videos in sport-prefixed directories (`aerobic_gymnastics/`, `volleyball/volleyball/`, `football/`, `basketball/`, `finegym_01..05/`). |

### Trained artifacts

| path | content |
|---|---|
| [student_lora/adapter_config.json](student_lora/adapter_config.json) | LoRA config (r=16, alpha=32, target_modules). |
| [student_lora/adapter_model.safetensors](student_lora/adapter_model.safetensors) | Trained LoRA weights (~25 MB). |
| [train_losses.jsonl](train_losses.jsonl) | Per-step CE / KL / loss / qa_id over 3 153 steps. |

### Eval artifacts

| path | content |
|---|---|
| [eval_before.jsonl](eval_before.jsonl) | 100-sample held-out, vanilla 0.8 B answers. |
| [eval_after.jsonl](eval_after.jsonl) | 100-sample held-out, trained student answers. |
| [eval_comparison.md](eval_comparison.md) | Per-row side-by-side written by `train_student.py`. |
| [eval_flip_report.md](eval_flip_report.md) | Held-out flip-rate report (fuzzy similarity). |
| [predictions_test_vanilla.jsonl](predictions_test_vanilla.jsonl) | 1 000-sample test set, vanilla 0.8 B answers. |
| [predictions_test_trained.jsonl](predictions_test_trained.jsonl) | 1 000-sample test set, trained student answers. |
| [eval_test_flip_report.md](eval_test_flip_report.md) | Test-set flip-rate report. |

### Operational

| path | content |
|---|---|
| [RUNBOOK_RENTAL.md](RUNBOOK_RENTAL.md) | Step-by-step rental → extraction → rsync runbook. |
| [METHODOLOGY_REPORT.md](METHODOLOGY_REPORT.md) | Longer prose write-up (academic audience). |

---

## 15. How to reproduce

1. **Build the subsets.**
   ```bash
   python sample_subset.py          # train + val
   python sample_test_subset.py     # test (8-sport parity)
   ```

2. **Pull subset videos.**
   ```bash
   # train + val
   python download_subset_videos.py --repo-id KD0GP0IMSIU/KnowledgeDistillationVLMsQA \
       --token <HF_TOKEN>

   # test
   python download_subset_videos.py --repo-id KD0GP0IMSIU/KnowledgeDistillationVLMsQA \
       --manifest data/subsets/subset_test_video_manifest.json \
       --splits test --token <HF_TOKEN>
   ```

3. **Rent a GPU pod** (H100 or H100-NVL recommended) and follow [RUNBOOK_RENTAL.md](RUNBOOK_RENTAL.md) to push code + pull weights + extract logits:
   ```bash
   # on the pod
   python -u extract_logits_subset.py --split both
   ```

4. **Pull logits back to local WSL** (note `-s` for the space-in-pathname):
   ```bash
   rsync -avzs --info=progress2 \
     -e "ssh -p <PORT> -o StrictHostKeyChecking=no" \
     "root@<IP>:/workspace/Granduation Project/logits_cache_subset/" \
     "/home/accy/Granduation Project/logits_cache_subset/"
   rsync -avzs --info=progress2 \
     -e "ssh -p <PORT> -o StrictHostKeyChecking=no" \
     "root@<IP>:/workspace/Granduation Project/logits_cache_subset.manifest.jsonl" \
     "/home/accy/Granduation Project/logits_cache_subset.manifest.jsonl"
   ```

5. **Train the student (~3 hr at 1 epoch on 3090).**
   ```bash
   python -u train_student.py --holdout_per_sport "Gym=50,volleyball=50" --epochs 1
   ```
   Outputs: `student_lora/`, `eval_before.jsonl`, `eval_after.jsonl`, `eval_comparison.md`, `train_losses.jsonl`.

6. **Generate test-set predictions for both vanilla and trained.** Two regimes:

   ```bash
   # Canonical: curated 480-vid / 2 793-Q subset, 8-sport parity (~5 hr per model).
   python -u eval_test_predict.py \
       --subset_file data/subsets/subset_test.json \
       --out predictions_test_vanilla.jsonl

   python -u eval_test_predict.py \
       --subset_file data/subsets/subset_test.json \
       --adapter student_lora \
       --out predictions_test_trained.jsonl

   # Quick smoke (~1.9 hr per model, 5 raw buckets, fg lumped):
   python -u eval_test_predict.py --stratify_per_sport 200 \
       --out predictions_test_vanilla.jsonl
   python -u eval_test_predict.py --stratify_per_sport 200 \
       --adapter student_lora \
       --out predictions_test_trained.jsonl
   ```

7. **Run the flip-rate comparison.**
   ```bash
   python eval_compare.py \
       --before predictions_test_vanilla.jsonl \
       --after  predictions_test_trained.jsonl \
       --report eval_test_flip_report.md
   ```

The pipeline is fully resumable. Steps 3, 5, 6 all read existing output files on restart and skip already-completed work.
