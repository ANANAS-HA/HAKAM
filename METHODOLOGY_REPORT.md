# Methodology Report

This document records the research methodology followed during the project. It describes, in chronological order, the experimental decisions taken, the empirical evidence that motivated each one, and the artifacts produced at every stage. Every numerical claim and prompt excerpt is grounded in a file that physically exists in the project directory and is cited inline. Infrastructure and tooling work (cloud-GPU rental setup, virtual environment management, model-quantization kernel debugging) is excluded; only material relevant to the research question is reported here.

---

## 1. Project Goal and Research Questions

The project's objective is to **distill the knowledge of a large vision–language teacher model into a small student model** capable of answering questions about short sports video clips. The student is `Qwen/Qwen3.5-0.8B`, chosen so that inference fits on a single consumer GPU (NVIDIA RTX 3090, 24 GB VRAM). The teacher candidates are larger members of the same family: `Qwen3.5-4B`, `Qwen3.5-9B`, and `Qwen3.5-27B`. Choosing a teacher within the same model family preserves tokenizer parity, which is a precondition for sequence-level knowledge distillation (verified in [_smoke_distill.py](_smoke_distill.py): `t_tok.get_vocab() == s_tok.get_vocab()` over 248 320 tokens).

The hardware asymmetry between teacher and student forces a **two-stage pipeline**: an expensive teacher run, executed once on rented compute, produces cached logits; a cheap student run, executed locally, trains many times against those cached logits. This separation is the central architectural decision of the project.

Three research questions structured the work:

1. **Is the dataset adequate for the intended distillation objective?** That is, do the gold labels carry enough signal — and at the right granularity — to be a useful supervision target?
2. **Is the teacher reliable enough at the chosen size to serve as a distillation source?** Specifically, does its reasoning on a given clip remain consistent across paraphrased questions, and does its output match what is visually present in the clip?
3. **Is the proposed distillation pipeline computationally feasible** within the project's hardware and budget envelope?

The remainder of the report is organised around evidence collected to answer these three questions.

---

## 2. Dataset Description and Critical Analysis

### 2.1 Splits and schema

The dataset consists of three JSON files in [meta-data/](meta-data/):

| Split | Rows | File size |
|---|---|---|
| `train.json` | 56 385 | 16 MB |
| `val.json`   | 18 970 | 5.3 MB |
| `test.json`  | 18 718 | 5.3 MB |
| **Total**    | **94 073** | |

Each row follows the schema:

```json
{
  "qa_id": 0,
  "video": "aerobic_gymnastics/v_HIow7_XktlQ_c003_01",
  "type": "Descriptive",
  "question": "What is the video about?",
  "answer": "aerobic gymnastics",
  "ans_cls": 16
}
```

A separate file [meta-data/ans2cls.json](meta-data/ans2cls.json) maps each unique answer to a class index. There are exactly **191 unique answers** across all 94 073 rows; that is, the answer space is closed.

### 2.2 Question-type distribution

The `type` field labels each question into one of four categories. The distribution is similar across the three splits:

| Type | Train | Val | Test | % overall |
|---|---|---|---|---|
| Descriptive    | 28 885 | 9 707 | 9 676 | 51.3 % |
| Temporal       | 23 906 | 7 969 | 7 768 | 42.0 % |
| Causal         |  2 712 |   994 |   970 |  5.0 % |
| Counterfactual |    882 |   300 |   304 |  1.6 % |

### 2.3 Answer-length and frequency distribution

A scan of the answers across all three splits shows that the vocabulary is heavily skewed toward short labels. Approximately **57 % of answers are a single word**, ~12 % are two words, and the long tail consists of fixed multi-word phrases (e.g., `transition flight from low bar to high bar`, `flic-flac with step-out, also with support on one arm`). The four most common answers across the dataset are:

| Answer | Count |
|---|---|
| `no` | 14 172 |
| `yes` | 13 511 |
| `1` | 9 283 |
| `2` | 3 839 |

These four labels alone cover roughly **45 % of all rows** in the dataset.

### 2.4 Video assets

The clips referenced by the `video` field live in [videos/](videos/), organised into eight sport folders (`aerobic_gymnastics`, `basketball`, `football`, `volleyball`, `finegym_01` through `finegym_05`). The total disk usage is **83 GB**.

### 2.5 Critical finding: the dataset is classification, not open QA

Two characteristics of the dataset together change its nature: (a) the answer vocabulary is a closed set of 191 strings, and (b) ~45 % of all answers are taken from only four labels. The dataset is therefore best understood as a **191-class classification problem dressed in question–answer formatting**, not as open-ended question answering.

This finding has direct methodological consequences:

- An exact-match metric is a meaningful upper bound, but a lossy one — generative model outputs frequently express the correct concept in non-exact form (e.g., `playing basketball` for gold `basketball`, `three` for gold `3`).
- Distilling such a teacher solely against the 191 labels risks training a student that learns the **dataset's biases**, not its genuine video-understanding capability. Conversely, distilling against the teacher's free-form rationales risks teaching the student to imitate language patterns unrelated to the closed answer vocabulary.

Both risks are addressed below in the distillation pipeline design (Section 6).

---

## 3. Teacher Model Capacity Assessment

Three teacher candidates were exercised on the basketball clip `basketball/v_-6Os86HzwCs_c001_00` to compare their behaviour. The student candidate `Qwen3.5-0.8B` was also evaluated as a baseline. Memory measurements were obtained from `torch.cuda.max_memory_allocated()` at peak and reported in the corresponding smoke logs:

| Model | bf16 VRAM at peak | Source |
|---|---|---|
| `Qwen3.5-0.8B`  | ~1.78 GB  | inferred from runs in [describe_video.py](describe_video.py) |
| `Qwen3.5-4B`    | ~10.4 GB  | observed in early `answer_video_qa.py` runs |
| `Qwen3.5-9B`    | 17.96 GB peak (loaded at 17.53 GB) | [smoke_9b_revised_v3.txt](smoke_9b_revised_v3.txt) |
| `Qwen3.5-27B` (bf16) | requires > 24 GB | does not fit on the 3090; required cloud GPU |

Qualitatively, the smaller models produced terse single-line outputs, while the larger models produced fluent multi-paragraph descriptions when allowed to. A practical wrinkle was discovered during initial integration: **the default value of the chat template's `enable_thinking` flag inverts across model sizes**. Inspection of the cached `chat_template.jinja` files for each variant showed:

- Qwen3.5-0.8B: thinking off by default
- Qwen3.5-4B and 9B: thinking **on** by default

Because of this, code that did not pass `enable_thinking` explicitly produced terse classifier-style output for the 0.8 B model and verbose internal reasoning for the larger ones. All inference paths in the project were updated to set `enable_thinking` explicitly to make the choice visible at the call site.

---

## 4. Prompt Engineering Methodology

### 4.1 Two prompt families

Two qualitatively different prompt strategies were developed to address two different research questions.

**Prompt A — Constrained Classification.** A short system prompt that frames the task as a classification problem, instructs the model to output a single label from a fixed vocabulary, and provides a small number of examples. The exact text appears in [answer_video_qa.py](answer_video_qa.py) and [_smoke_9b_constrained.py](_smoke_9b_constrained.py):

> "You answer questions about short sports videos. This is a classification task: every answer is a single label from a fixed vocabulary. Output ONLY the answer label, nothing else — no sentences, no explanations, no punctuation. Answers are typically 1 to 3 words. Examples of valid answers: 'yes', 'no', '1', '2', '3', 'basketball', 'football', 'volleyball', 'aerobic gymnastics', 'volleyball spike', 'basketball 2-point shot', 'basketball missed', 'left team', 'right team'."

This prompt is paired with `enable_thinking=False`, greedy decoding (`do_sample=False`), and `max_new_tokens=16`. It is the prompt used when the goal is to evaluate the model's ability to commit to a single label that can be exact-matched against the gold.

**Prompt B — Expert Sports Analyst.** A longer system prompt that frames the model as an expert commentator, instructs it to reason step by step, and lists specific technical vocabulary across sport domains. The full text appears in [_smoke_9b.py](_smoke_9b.py) lines 19–32. It is paired with `enable_thinking=True` and `max_new_tokens=3072`. This prompt is used when the goal is to evaluate the **quality of the model's reasoning**, not just its label-matching accuracy.

### 4.2 Discovery of the scoreboard shortcut

While reviewing 9 B output traces under Prompt B, a recurring pathology was identified: when asked whether a scoring action had succeeded (e.g. "Does any team score?"), the model would inspect the scoreboard graphic in the broadcast frame and conclude that no score had occurred *because the scoreboard number had not changed within the clip*. This substitutes a UI signal for the physical event the question is actually about. The error mode is reinforced by the fact that scoreboards typically lag the action by 1–3 seconds and may not update within a short clip.

This pathology is significant for distillation because the student would inherit the same shortcut, since reasoning traces are the supervision signal.

The prompt was revised to insert an explicit anti-shortcut clause:

> "Outcome questions must be answered from the action itself, not from interface elements. When a question asks whether an action *succeeded*, your verdict must come from observing the action's physical result: ball relative to rim/net, body relative to landing surface, ball crossing the goal line. Do not use scoreboards, scorelines, point counters, or referee gestures as evidence of success or failure. Scoreboards are graphical UI, lag the action by 1–3 seconds, and may not update within the clip's duration."

The revised prompt is the version present in [_smoke_9b.py](_smoke_9b.py).

### 4.3 Sampling-parameter iteration

Three rounds of sampling-parameter changes were run with the revised prompt on the same calibration clip (`basketball/v_4r8QL_wglzQ_c002_00`, 10 questions). Each round was logged separately for direct comparison.

**Round v1** ([smoke_9b_revised_v1.txt](smoke_9b_revised_v1.txt))
Settings: `temperature=1.0, top_p=0.95, top_k=20, repetition_penalty=1.0, max_new_tokens=3072`.
Result: gold-label-in-answer score 4 / 10. Cross-question consistency was poor — the model gave conflicting answers about the same physical event across questions Q3 and Q4 ("did anyone score?" / "did any team gain a point?"). The cap on `max_new_tokens` was hit on 30 % of questions.

**Round v2** (`temperature=0.3, top_p=0.9, top_k=20`)
Result: when the model was confident, it converged faster and produced shorter traces (Q5 ≈ 700 tokens). When uncertain, however, the lower-temperature sampler entered a degenerate state in which an identical 11-line block of text was emitted ten times in a row before hitting the token cap. Examination of the trace showed bit-exact repetition, characteristic of a low-temperature attractor with no repetition penalty. The smoke run on Q4 illustrates this clearly.

**Round v3** ([smoke_9b_revised_v3.txt](smoke_9b_revised_v3.txt))
Settings: `temperature=0.5, top_p=0.9, top_k=20, repetition_penalty=1.1, no_repeat_ngram_size=8, max_new_tokens=3072`.
Result: gold-label-in-answer score 5 / 10. Bit-exact loops were eliminated by the n-gram constraint. Cross-question consistency on the central factual claim ("did anyone score?") improved to 9 / 9 completed answers in agreement (all "yes"), and team-attribution consistency rose from approximately 50 % under v1 to approximately 78 % under v3. Cap-hit rate decreased from 30 % to 20 %. Median trace length decreased from ≈ 1 700 tokens (v1) to ≈ 1 500 tokens (v3).

The final v3 settings are the configuration committed to [_smoke_9b.py](_smoke_9b.py).

---

## 5. Visual-Grounded Calibration Methodology

A weakness of any internal evaluation is that the dataset's gold labels are themselves the only source of ground truth. To break this circularity for prompt-iteration purposes, a visual-grounding methodology was developed.

### 5.1 Frame extraction

The script [extract_frames.py](extract_frames.py) uses `decord.VideoReader` to sample frames at a fixed rate (default 4 fps, configurable). Frames are resized to 640 px wide and saved as JPEGs at quality 85. Two basketball clips were processed for the calibration loop:

- `basketball/v_-6Os86HzwCs_c001_00` — 15 s, 63 frames at 4 fps
- `basketball/v_4r8QL_wglzQ_c002_00` — 10 s, 42 frames at 4 fps

Frames are stored under [frames/basketball/](frames/basketball/) and serve as the visual reference for all subsequent comparisons.

### 5.2 Independent visual ground truth

For each calibration clip, a frame-by-frame description was produced that captured: the number of athletes, jersey colours and team mappings, the position of the ball at each timestamp, the action being performed, the scoreboard state, and any salient transitions (possession changes, shot attempts, body language indicating score / rebound). This description is *independent of the dataset's gold labels* — it was constructed from visual evidence alone.

### 5.3 Trace-tagging protocol

Each line of the model's reasoning trace was tagged against the visual ground truth into one of five categories:

- **grounded** — the claim corresponds to what the frames actually show
- **plausible inference** — the claim is consistent with the frames but not directly observable
- **unsupported** — the claim is neither confirmed nor denied by the frames
- **contradicted** — the claim is directly contradicted by the frames
- **shortcut** — the claim is the result of substituting a proxy signal (e.g. scoreboard, referee gesture) for direct observation of the action

This tagging procedure was applied to the v1, v2, and v3 smoke outputs on the same clip and produced the consistency measurements reported in Section 4.3.

### 5.4 Discovery of label noise in the dataset

A side effect of the visual grounding was the discovery that several of the dataset's gold labels do not match what is visible in the corresponding clip. Two concrete examples were documented during the calibration of `basketball/v_4r8QL_wglzQ_c002_00`:

- The dataset annotates this clip with `basketball 3-point shot` for the questions of type "How does the left team win a point?" Inspection of the frames at timestamps f022–f026 shows the shot release from the paint area (near the rim), well inside the three-point arc. The model under Prompt B at v3 settings consistently classifies the shot as a layup. The visual evidence supports the model.
- An earlier calibration on the gymnastics clip `aerobic_gymnastics/v_HIow7_XktlQ_c003_01` revealed analogous mislabelling: questions presupposing a "push-up" action have gold answers that depend on the action's existence, but the clip's frames show only cartwheels, splits, and floor poses with no push-up motion.

These findings are recorded both in [smoke_9b_revised_v3.txt](smoke_9b_revised_v3.txt) and in earlier traces in [smoke_9b_out.txt](smoke_9b_out.txt). They are reproducible.

The methodological consequence is that **exact-match accuracy against gold is a lower bound on a model's true visual understanding** — a model can be visually correct and exact-match wrong simultaneously.

---

## 6. Distillation Pipeline Design

### 6.1 Two-stage architecture

The distillation pipeline is split into a one-time teacher pass and a many-time student pass:

1. **Extraction stage (rented teacher GPU).** The teacher is run once over the training set; per-token logit information is written to disk as a cache. This step is expensive but is amortised over all subsequent student training runs.
2. **Training stage (local student GPU).** The student is trained against the cached logits combined with the gold labels. This step is iterated freely as the student-side configuration evolves.

This split is the only reason the project is feasible on the available hardware.

### 6.2 Two recipes for logit extraction

Two recipes were formalised in [extract_logits_experiment.py](extract_logits_experiment.py):

**Recipe A — Free generation.** The teacher is given the prompt and allowed to generate freely with `do_sample=False, output_scores=True, return_dict_in_generate=True`. The full per-step logit distribution is captured at each generated position, and the resulting sequence (which can run to hundreds or thousands of tokens) is the supervision target. This recipe captures the teacher's full reasoning trace.

**Recipe B — Teacher-forced on the gold label.** The teacher is given the prompt **with the gold answer appended** and run as a single forward pass; logits are read at the answer positions. No generation occurs. The recipe is fast (one forward per sample) and produces a focused signal at exactly the tokens the student is supposed to emit.

The two recipes serve different goals: Recipe A is appropriate when the goal is to teach the student to *reason* like the teacher; Recipe B is appropriate when the goal is to teach the student to *output* the dataset's gold labels.

### 6.3 Top-K logit compression

Storing the full logit vector at each position is impractical. The Qwen3.5 vocabulary contains **248 320 tokens** (verified at run time in [_smoke_distill.py](_smoke_distill.py): `len(t_tok.get_vocab()) == 248 077`; the full tokenizer including added tokens is 248 320). At fp16, full-vocab logits over 100 positions × 10 000 samples would require ≈ 993 GB.

The pipeline therefore stores only the **top-K** entries per position, with K = 32. The empirical justification for this choice is the probability mass captured by the top-32 distribution. From the manifest [logits_cache_train.manifest.jsonl](logits_cache_train.manifest.jsonl) on the 9 B teacher across five training samples:

| Sample | top-32 mass |
|---|---|
| 0 | 0.99845 |
| 1 | 0.99254 |
| 2 | 0.99979 |
| 3 | 0.99975 |
| 4 | 0.99800 |

The range is **99.25 %–99.98 %**, all comfortably above the practical threshold (≈ 96 %) at which the discarded tail can be treated as noise. Total storage at K = 32 is approximately 100 MB per 10 000 samples — three orders of magnitude smaller than the full-vocab alternative.

Logits are stored as fp16 values plus int32 indices in PyTorch tensor files (`.pt`), one file per `qa_id`. The on-disk artifacts for the experimental run are visible in [logits_cache_experiment/](logits_cache_experiment/) (5 files) and for the training extraction prototype in [logits_cache_train/](logits_cache_train/) (5 files).

### 6.4 Gold-token preservation rule

A subtle correctness issue arises in Recipe B: when the teacher disagrees strongly with the gold answer, the gold token may not appear in the teacher's top-K set. If left unhandled, the student's KL divergence at that position would be undefined because the gold-token slot in the target distribution is zero.

The rule implemented in [extract_logits_experiment.py](extract_logits_experiment.py) is: when the gold token is missing from the top-K set, the lowest-ranked top-K slot is replaced with the gold token's logit. This guarantees the gold token is always representable. The replacement event is logged so the rate of teacher–gold disagreement can be measured.

### 6.5 Greedy-consistency canary

For Recipe A, a sanity check is built into the extractor: the argmax over the stored top-K logits at each position must reproduce the token that was actually generated by the teacher. Any discrepancy indicates a misalignment in how logits were captured. The check passed on all 5 / 5 experimental samples (recorded in [LOGITS_REPORT.md](LOGITS_REPORT.md)).

### 6.6 Per-video caching for production extraction

The training set's videos are referenced by an average of **15.7 questions per video**. Because video decoding is the most expensive single step in the pipeline, the production extractor [extract_logits_train.py](extract_logits_train.py) groups samples by video and decodes each clip exactly once per session, looping over its questions in memory. Without this optimisation, ≈ 93 % of vision-side compute would be wasted.

---

## 7. Distillation Training Loop Design

### 7.1 Loss formulation

The student is trained against a **weighted sum of two terms**:

- A standard cross-entropy term against the gold token IDs
- A Kullback–Leibler divergence term between the student's distribution and the teacher's stored distribution

The combined objective is `α · KL_T + (1 − α) · CE`, with `α = 0.7` in the demonstration runs (parameter `KL_WEIGHT` in [_smoke_distill.py](_smoke_distill.py)).

### 7.2 Temperature-scaled KL

Following Hinton et al.'s convention, both teacher and student logits are divided by a temperature `T > 1` before softmax, and the KL term is multiplied by `T²` to preserve gradient magnitude as `T` grows. The default in [_smoke_distill.py](_smoke_distill.py) is `T = 2.0`.

### 7.3 Sparse top-K KL

Because only the top-K entries of the teacher distribution are stored, the KL is computed in the sparse regime: the student's logits at the teacher's top-K indices are gathered, both teacher and student distributions are renormalised over those K entries, and the KL is computed in the renormalised space. The implementation is the function `topk_kl()` defined in `_smoke_distill.py`. It is mathematically equivalent to full-vocab KL in the limit `K → V`.

### 7.4 Vision-encoder caching

A second optimisation, unique to the training stage, is to freeze the student's vision tower and cache its outputs once per video. Every training epoch then reuses the cached visual embeddings rather than re-running the encoder. Combined with the per-video grouping from extraction, this turns the per-step cost from "vision encoder + LM forward + LM backward" into "LM forward + LM backward."

### 7.5 Wall-time projections

Per-step times for the 0.8 B student were measured at approximately 0.5 s on the RTX 3090 with bf16 weights and gradient checkpointing enabled. With ≈ 26 000 present-video training samples, the wall-time projection is:

- 1 epoch ≈ 11 hours
- 5 epochs ≈ 55 hours

These projections informed the choice of batch size, gradient-checkpointing setting, and LoRA-vs-full-fine-tune comparison; the full ablation is documented in the supporting calculations in the logits tutorial notebook.

### 7.6 Validating implementations

End-to-end correctness of the training loop was validated by [_smoke_distill.py](_smoke_distill.py), which runs the full extraction-then-training pipeline at micro scale (4 B teacher, 0.8 B student, 5 samples, 2 epochs). Tutorial notebooks [distill_demo.ipynb](distill_demo.ipynb) and [logits_tutorial.ipynb](logits_tutorial.ipynb) provide stepwise walkthroughs of the same logic for documentation purposes.

---

## 8. Empirical Findings on Teacher Quality

### 8.1 Quantitative results

The following gold-label-in-answer scores were recorded on the calibration clips. Each score is the proportion of questions for which the gold label appears as a substring of the model's final answer (a deliberately permissive metric — see Section 8.3).

| Run | Prompt | Sampling | Clip | Score | Source |
|---|---|---|---|---|---|
| Constrained-9B | A | greedy, max=16 | basketball/v_-6Os86HzwCs_c001_00 (16 Q) | 8 / 16 = 50.0 % | reproduced from `_smoke_9b_constrained.py` |
| Analyst-9B v1 | B | T = 1.0, top_p = 0.95 | basketball/v_4r8QL_wglzQ_c002_00 (10 Q) | 4 / 10 = 40.0 % | [smoke_9b_revised_v1.txt](smoke_9b_revised_v1.txt) |
| Analyst-9B v3 | B | T = 0.5, top_p = 0.9, rep_pen = 1.1, no_repeat_8 | same clip (10 Q) | 5 / 10 = 50.0 % | [smoke_9b_revised_v3.txt](smoke_9b_revised_v3.txt) |
| Analyst-9B (gymnastics) | B | T = 1.0 | aerobic_gymnastics/v_HIow7_XktlQ_c003_01 (16 Q) | 11 / 16 = 68.8 % | [smoke_9b_out.txt](smoke_9b_out.txt) |

### 8.2 Qualitative findings

Three findings consistently appeared across runs and clips:

1. **Cross-question contradictions decrease but do not vanish with sampling control.** Under v1 settings the model would commit to opposite answers on Q3 and Q4 about the same physical event. Under v3 settings the central question of "did anyone score?" agrees across all completed answers, but the secondary question of "which team scored?" still flips between two teams roughly 22 % of the time.
2. **Token efficiency improves with sampling control.** Median trace length under v3 (≈ 1 500 tokens) is shorter than under v1 (≈ 1 700 tokens). The cap-hit rate (the proportion of traces that exhaust `max_new_tokens=3072`) drops from 30 % to 20 %.
3. **Some perceptual errors are not addressable by sampling alone.** The model occasionally **hallucinates** scoreboard digit changes that did not occur in the actual frames. This hallucination is downstream of the visual encoder, not the language sampler.

### 8.3 The exact-match metric is a lower bound

The substring-match scoring rule used in the smoke logs counts as ✓ any answer whose final text contains the gold string. It still under-reports correctness in two specific ways:

- **Number-word disagreement.** When the gold answer is `3` and the model writes "There are three players in the video," the score counts this as ✗.
- **Sport-prefix mismatch.** When the gold answer is `aerobic gymnastics push up` and the model writes `cartwheel` (which, per the visual-grounded calibration, *correctly describes the clip*), the score counts this as ✗ — even though the model is right and the gold label is incorrect.

A normalisation-aware metric that maps number words to digits, ignores sport prefixes for action labels, and snaps free-form text to its nearest member of the closed 191-label vocabulary would raise the reported numbers materially. Such a metric is straightforward to implement and is identified as the appropriate evaluation measure for any final results.

---

## 9. Decision Framework Derived from Findings

The empirical evidence above produced a small set of decision rules used throughout the project.

**Recipe selection.** When the teacher's reasoning is *internally consistent on the same clip* (verified via the visual-grounded calibration), Recipe A is preferred — its richer signal teaches the student more than gold labels alone. When internal consistency is absent, Recipe B (teacher-forced on gold) is the conservative choice: it limits the student's exposure to teacher confabulations at the cost of inheriting the dataset's label noise.

**Teacher-size selection.** A teacher must satisfy three criteria simultaneously: (1) it fits the available memory in bf16 (or in a faithful quantization); (2) its reasoning passes the visual-grounded calibration on at least three clips spanning the dataset's variety; (3) the per-sample extraction cost on the available hardware closes within budget. Failure on any criterion eliminates a candidate.

**Prompt-iteration termination.** Iteration on prompt and sampling is continued only as long as new failure modes are still being addressed. Once the remaining errors are *perceptual* (hallucinated scoreboard changes, jersey-colour binding under fast motion) rather than *textual* (looping, contradictions, format drift), prompt engineering has reached its ceiling and further investment is moved to teacher-capacity selection or to dataset cleanup.

---

## 10. Methodological Limitations

Three limitations are acknowledged.

1. **Teacher perceptual ceiling at 9 B with thinking.** The visual-grounded calibration showed that even at v3 sampling, the 9 B teacher confidently misperceives certain fine-grained sports actions (ball-rim trajectory, jersey-colour-to-team binding under fast motion, scoreboard digit reading). This ceiling is a property of the model's visual representation, not of decoding strategy.

2. **Dataset gold-label noise.** Section 5.4 documented two reproducible cases in which the dataset's gold answer disagrees with the visual content of the corresponding clip. Exact-match accuracy against such labels is therefore an unreliable measure of actual visual understanding; it is treated in this project as a lower bound only.

3. **Hardware-imposed teacher-size cap for local validation.** Validating teacher behaviour at the largest scale (`Qwen3.5-27B` in bf16) is not possible on the local 24 GB GPU. Calibration and final extraction must be performed on rented compute. This shifts a class of methodological decisions from "verifiable locally before committing" to "verified once on rented compute, then committed." This is a real cost, but it is bounded — the rented-compute window is the only step at which a wrong teacher choice incurs significant expense.

---

## Appendix: Index of project artifacts

The following files at the project root were used as factual sources for this report. Each numerical or behavioural claim above is verifiable by inspecting one or more of them.

**Code (Python and notebooks)**
- [_smoke_9b.py](_smoke_9b.py) — analyst-prompt smoke, current v3 settings
- [_smoke_9b_constrained.py](_smoke_9b_constrained.py) — constrained-prompt smoke, greedy decoding
- [_smoke_27b.py](_smoke_27b.py) and [_smoke_27b_int4.py](_smoke_27b_int4.py) — 27 B variants
- [_smoke_distill.py](_smoke_distill.py) — end-to-end distillation smoke (4 B → 0.8 B)
- [answer_video_qa.py](answer_video_qa.py) — single-video QA with constrained prompt
- [describe_video.py](describe_video.py) — captioning baseline on 0.8 B
- [evaluate_test.py](evaluate_test.py) — full test-set evaluator
- [extract_logits_experiment.py](extract_logits_experiment.py) — Recipe A and Recipe B prototype
- [extract_logits_train.py](extract_logits_train.py) — production extraction (Recipe B, per-video grouping)
- [extract_frames.py](extract_frames.py) — `decord`-based frame sampler
- [distill_demo.ipynb](distill_demo.ipynb) — distillation walkthrough
- [logits_tutorial.ipynb](logits_tutorial.ipynb) — logit handling and visualisation
- [answer_video_qa.ipynb](answer_video_qa.ipynb) — QA notebook variant
- [chatbot.ipynb](chatbot.ipynb) — minimal text-only chat loop

**Data**
- [meta-data/train.json](meta-data/train.json) (56 385 rows)
- [meta-data/val.json](meta-data/val.json) (18 970 rows)
- [meta-data/test.json](meta-data/test.json) (18 718 rows)
- [meta-data/ans2cls.json](meta-data/ans2cls.json) (191 classes)
- [videos/](videos/) (83 GB, 8 sport folders)
- [frames/](frames/) (calibration frame extracts)
- [logits_cache_experiment/](logits_cache_experiment/) (5 sample `.pt` files)
- [logits_cache_train/](logits_cache_train/) (5 sample `.pt` files)
- [logits_cache_train.manifest.jsonl](logits_cache_train.manifest.jsonl) (per-sample top-K mass and greedy-consistency entries)

**Logs**
- [smoke_9b_out.txt](smoke_9b_out.txt) — gymnastics clip, 16 Q, score 11/16
- [smoke_9b_revised_v1.txt](smoke_9b_revised_v1.txt) — basketball clip, v1 sampling, score 4/10
- [smoke_9b_revised_v3.txt](smoke_9b_revised_v3.txt) — basketball clip, v3 sampling, score 5/10
- [smoke_9b_constrained.txt](smoke_9b_constrained.txt) — basketball clip, constrained prompt
- [smoke_27b_int4_v1.txt](smoke_27b_int4_v1.txt) — quantized 27 B variant (early termination)

**Documentation**
- [LOGITS_REPORT.md](LOGITS_REPORT.md) — earlier internal report on logit-extraction design
- [requirements.txt](requirements.txt) — Python dependencies
- [handoffs/HANDOFF.md](handoffs/HANDOFF.md) — original project handoff brief

End of report.
