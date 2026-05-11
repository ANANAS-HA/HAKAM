# Appendix Materials — Hakam GP2 Final Report

Verbatim source content gathered for web Claude. No prose interpretation; all sections present the raw artifacts referenced in the GP2 report.

---

## SECTION 1: Locked analyst-v3 system prompt

`SYSTEM_PROMPT` constant from `extract_logits_subset.py:53–79`:

```text
You are an elite sports video analyst with deep expertise across basketball, football, volleyball, and gymnastics (including aerobic gymnastics, vault, uneven bars, balance beam, and floor exercise). You watch short game or performance clips and parse them the way a coach or color commentator would: identifying the sport, naming specific techniques and skills (e.g. 2-point shot, spike, push-up, split, round-off, flic-flac, giant circle), counting discrete events and the number of athletes involved, tracking temporal order (what comes before / after what), and reasoning about cause and effect (why an action succeeded or failed, what a counterfactual outcome would have been).

When given a question, think it through step by step — describe what you see in the clip, locate the moment the question refers to, and then give a precise, expert answer. Be decisive: sports are concrete, so give committed answers rather than hedged ones. If the question is a yes / no, still justify briefly. If it asks 'how many', count carefully. If it asks for the name of an action, use the technical term.

**Outcome questions must be answered from the action itself, not from interface elements.** When a question asks whether an action *succeeded* (did the shot go in, did the team score, did the spike land, did the dismount stick), your verdict must come from observing the action's physical result: ball relative to rim/net, body relative to landing surface, ball crossing the goal line. **Do not use scoreboards, scorelines, point counters, or referee gestures as evidence of success or failure.** Scoreboards are graphical UI, lag the action by 1-3 seconds, and may not update within the clip's duration. If the play is cut off before the result is visually resolved, say so explicitly and answer from what was visible — do not infer success or failure from a static score.

**Be concise.** Aim for around 300-500 tokens total. Identify the moment, state the evidence, commit to a verdict. Do not re-examine the same evidence twice or hedge between interpretations. Brevity over rumination.
```

Sampling configuration block from `extract_logits_subset.py:167–180`:

```python
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
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

Where `MAX_NEW_TOKENS = 256` (defined as a module-level constant earlier in the file).

---

## SECTION 2: Software / environment versions

Output of the version-probe commands run on 2026-05-09 (the local WSL2 host):

```text
=== python --version ===
Python 3.12.3
=== torch ===
torch: 2.11.0+cu130
cuda: 13.0
=== transformers ===
transformers: 5.5.3
=== peft ===
peft: 0.19.1
=== Levenshtein ===
Levenshtein: 0.27.3
=== decord ===
decord: 0.6.0
=== qwen_vl_utils ===
qwen_vl_utils: present
=== autoawq ===
ModuleNotFoundError: No module named 'autoawq'
=== gptqmodel ===
gptqmodel: 7.0.0
=== triton ===
triton: 3.6.0
=== nvidia-smi ===
name, driver_version, memory.total [MiB]
NVIDIA GeForce RTX 3090, 591.86, 24576 MiB
=== uname ===
Linux DESKTOP-KPJDUFD 6.6.87.2-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Thu Jun  5 18:30:46 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux
=== os-release ===
PRETTY_NAME="Ubuntu 24.04.3 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.3 LTS (Noble Numbat)"
VERSION_CODENAME=noble
```

Notes:
- `autoawq` is **not installed locally** — that's correct. The 27 B AWQ-Int4 teacher only ran on the rented H100-NVL pod, where `autoawq` was installed alongside the pod's pre-existing PyTorch image. The local 3090 cannot fit the AWQ teacher and never needed `autoawq`.
- `gptqmodel` is installed but unused for the actual training; its `dispatch_awq` is monkey-patched to a no-op (see Section 6 for the workaround code).

`requirements.txt` contents:

```text
# Pinned where the project needs a specific minimum; loose otherwise so we
# don't fight Colab's preinstalled torch / CUDA stack.

# Core HF stack — must support Qwen3.5's qwen3_5 model_type
transformers>=5.5,<6
accelerate>=1.0

# Project-specific (NOT preinstalled on Colab)
qwen-vl-utils>=0.0.14
decord>=0.6
av>=10                # video-decoder fallback used by qwen_vl_utils

# torchvision is imported transitively by qwen_vl_utils.vision_process.
# Colab ships a torchvision matched to its torch; pip will skip if already satisfied.
torchvision

# Notebook extras — only needed if the mate runs the tutorial / chatbot notebooks
matplotlib>=3.5
ipywidgets>=8
```

`pyproject.toml` and `environment.yml`: not present. Only `requirements.txt` exists.

`peft` and `python-Levenshtein` are not in `requirements.txt` — they were installed manually with `.venv/bin/pip install -q peft python-Levenshtein` during the training-pipeline build phase. The repo state at the time of this report has them installed at the versions printed above.

---

## SECTION 3: Subset statistics

Verbatim contents of `data/subsets/subset_stats.json`:

```json
{
  "train": {
    "videos": 1600,
    "questions": 8937,
    "per_sport": {
      "Gym": 200,
      "volleyball": 200,
      "football": 200,
      "basketball": 200,
      "Uneven_Bar": 200,
      "Vault": 200,
      "Floor_Exercise": 200,
      "Balance_Beam": 200
    },
    "per_question_type": {
      "Descriptive": 4614,
      "Temporal": 2965,
      "Causal": 1032,
      "Counterfactual": 326
    },
    "yes_no": {
      "yes": 486,
      "no": 486,
      "ratio": 0.5,
      "dropped": 253
    }
  },
  "val": {
    "videos": 480,
    "questions": 2743,
    "per_sport": {
      "Gym": 60,
      "volleyball": 60,
      "football": 60,
      "basketball": 60,
      "Uneven_Bar": 60,
      "Floor_Exercise": 60,
      "Balance_Beam": 60,
      "Vault": 60
    },
    "per_question_type": {
      "Descriptive": 1390,
      "Temporal": 909,
      "Causal": 339,
      "Counterfactual": 105
    },
    "yes_no": {
      "yes": 153,
      "no": 153,
      "ratio": 0.5,
      "dropped": 68
    }
  }
}
```

Verbatim contents of `data/subsets/subset_test_stats.json`:

```json
{
  "test": {
    "videos": 480,
    "questions": 2793,
    "per_sport_videos": {
      "Gym": 60,
      "volleyball": 60,
      "football": 60,
      "basketball": 60,
      "Balance_Beam": 60,
      "Uneven_Bar": 60,
      "Vault": 60,
      "Floor_Exercise": 60
    },
    "per_sport_questions": {
      "Gym": 360,
      "volleyball": 398,
      "football": 478,
      "basketball": 297,
      "Balance_Beam": 360,
      "Uneven_Bar": 360,
      "Vault": 180,
      "Floor_Exercise": 360
    },
    "per_question_type": {
      "Descriptive": 1440,
      "Temporal": 913,
      "Causal": 320,
      "Counterfactual": 120
    },
    "yes_no": {
      "yes": 136,
      "no": 229,
      "ratio": 0.3726
    }
  }
}
```

---

## SECTION 4: Training loss summaries

Filepath: `train_losses.jsonl` (4 311 465 bytes; 35 016 rows).

First 3 lines:

```json
{"epoch": 1, "step": 0, "ce": 0.9674575924873352, "kl": 1.732259750366211, "loss": 1.5028190612792969, "qa_id": 14908}
{"epoch": 1, "step": 1, "ce": 2.11539888381958, "kl": 2.3985676765441895, "loss": 2.31361722946167, "qa_id": 6950}
{"epoch": 1, "step": 2, "ce": 1.5775257349014282, "kl": 1.908103108406067, "loss": 1.8089299201965332, "qa_id": 57962}
```

Middle 3 lines (around step 17 507, ~halfway):

```json
{"epoch": 2, "step": 17507, "ce": 0.6704775094985962, "kl": 0.5602970123291016, "loss": 0.5933511257171631, "qa_id": 70042}
{"epoch": 2, "step": 17508, "ce": 0.5064325928688049, "kl": 0.3530332148075104, "loss": 0.39905303716659546, "qa_id": 46179}
{"epoch": 2, "step": 17509, "ce": 0.8577216267585754, "kl": 0.6807525753974915, "loss": 0.7338433265686035, "qa_id": 2272}
```

Last 3 lines:

```json
{"epoch": 3, "step": 35013, "ce": 0.9441699981689453, "kl": 0.729698896408081, "loss": 0.7940402030944824, "qa_id": 25534}
{"epoch": 3, "step": 35014, "ce": 0.9073855876922607, "kl": 0.6237658262252808, "loss": 0.7088517546653748, "qa_id": 3114}
{"epoch": 3, "step": 35015, "ce": 0.44276171922683716, "kl": 0.28672316670417786, "loss": 0.33353471755981445, "qa_id": 76983}
```

Computed summary (Python script per the handoff spec):

```text
=== train_losses.jsonl ===
steps: 35016
keys: ['epoch', 'step', 'ce', 'kl', 'loss', 'qa_id']
CE  first=0.9675  last=0.4428  min=0.3092  max=2.6998
KL  first=1.7323  last=0.2867  min=0.1843  max=2.4356
```

Loss key names match the script's expected names (`ce`, `kl`, `loss`); no adaptation needed. The composite `loss` column at each step equals `0.7 · kl + 0.3 · ce` (verifiable on any single row).

---

## SECTION 5: Hardware / cost table — actual measurement

**Vast.ai billing history**: `Pending verification — need user to check Vast.ai billing dashboard`.

The user must paste exact rental hours, session count, and dates from the Vast.ai billing page before publication.

Adapter directory listing:

```text
=== student_lora_full3epoch/ ===
total 25008
drwxrwxr-x  2 accy accy     4096 May  8 19:42 .
drwxr-xr-x 17 accy accy     4096 May  9 19:27 ..
-rw-rw-r--  1 accy accy     5188 May  8 19:42 README.md
-rw-rw-r--  1 accy accy     1096 May  8 19:42 adapter_config.json
-rw-rw-r--  1 accy accy 25587104 May  8 19:42 adapter_model.safetensors

25M	student_lora_full3epoch/
```

Logits cache:

```text
=== logits_cache_subset_final/ ===
347M	logits_cache_subset_final/

qa_0000016.pt
qa_0000017.pt
qa_0000028.pt
qa_0000030.pt
qa_0000031.pt
qa_0000032.pt
qa_0000049.pt
qa_0000050.pt
qa_0000051.pt
qa_0000061.pt
qa_0000062.pt
qa_0000064.pt
qa_0000095.pt
qa_0000096.pt
qa_0000101.pt
qa_0000109.pt
qa_0000111.pt
qa_0000112.pt
qa_0000113.pt
qa_0000114.pt

file count:
11680
```

Note: the cache directory contains 11 680 .pt files; ~3 511 of them are 0-byte EOFError-on-load placeholders (extraction errors / partial rsync). The training loop's `load_available_samples()` silently skips empty files, so the effective usable cache is ~8 169 questions.

---

## SECTION 6: WSL2 / CUDA driver-bug mitigation code

**Mitigation 1: gc / sync / empty_cache** — between BEFORE-eval and training, in `train_student.py:546–557`:

```python
        print(f"[eval BEFORE] loaded {len(eval_before)} cached entries from {args.eval_before}")

    # 4b. Flush CUDA allocator state before switching to training. The BEFORE
    # eval allocates / frees lots of KV-cache buffers of varying shapes, which
    # leaves the allocator fragmented. The first training backward sometimes
    # surfaces this as "CUDA driver error: unknown error" when a fresh
    # workspace allocation collides with stranded blocks. A sync + collect +
    # empty_cache before training prevents that.
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"[transition] post-cleanup VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
```

**Mitigation 2: eager attention** — in `train_student.py:521–533`:

```python
    processor = AutoProcessor.from_pretrained(STUDENT_ID, trust_remote_code=True)
    _model_kwargs = dict(
        dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
    )
    # eager attention dodges a WSL2 SDPA backward race; on real Linux pods
    # we let HF pick SDPA for ~2x faster forward.
    if not args.pod_mode:
        _model_kwargs["attn_implementation"] = "eager"
    base_model = AutoModelForImageTextToText.from_pretrained(
        STUDENT_ID, **_model_kwargs
    )
    print(f"[model] loaded. VRAM={torch.cuda.memory_allocated()/1024**3:.2f} GB"
          f" (attn={'sdpa' if args.pod_mode else 'eager'})")
```

**Mitigation 3: CUDA_LAUNCH_BLOCKING** — at the top of the file in `train_student.py:30–38`:

```python
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# WSL2 + Qwen3.5-VL bf16 + LoRA backward hits a driver race that surfaces as
# "CUDA driver error: unknown error" on the first training backward. Forcing
# synchronous CUDA op submission makes this go away. ~5-10% slowdown is
# acceptable to keep the pipeline running. The default below keeps WSL2
# users covered; pod runs (--pod_mode) skip it.
_POD_MODE = "--pod_mode" in sys.argv
if not _POD_MODE:
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
```

**Mitigation 4: real-shape warmup forward+backward** — in `train_student.py:593–628`:

```python
    peft_model.train()

    # 7b. Real-shape warmup forward+backward to dodge a WSL2/CUDA driver race
    # where the very first backward of a fresh process fails with "CUDA driver
    # error: unknown error". The race doesn't fire on real Linux, so --pod_mode
    # skips this. The race is shape-sensitive: Gym samples reliably succeed,
    # volleyball samples sometimes don't on certain WSL2 states. Pick a Gym
    # sample for the warmup so the real first-shuffled training step is the
    # *second* backward of the process — past the danger zone.
    if not args.pod_mode:
        print("[train] warmup forward+backward on a Gym sample ...")
        warmup_row = next((r for r in train_rows if r["sport"] == "Gym"), None)
        if warmup_row is None:
            print("  no Gym sample — falling back to first train_row")
            warmup_row = train_rows[0]
        warmup_sample = torch.load(warmup_row["pt_path"], map_location="cpu",
                                    weights_only=False)
        warmup_inputs, warmup_plen = build_prompt_inputs(
            processor, warmup_row["video_path"], warmup_sample["question"])
        warmup_full = append_gen_tokens(warmup_inputs, warmup_sample["gen_tokens"])
        warmup_out = peft_model(**warmup_full)
        warmup_slog = warmup_out.logits[
            0, warmup_plen - 1: warmup_plen - 1 + int(warmup_sample["seq_len"])
        ].float()
        warmup_gen_ids = warmup_sample["gen_tokens"].to(warmup_slog.device)
        warmup_loss = F.cross_entropy(warmup_slog, warmup_gen_ids)
        optimizer.zero_grad(set_to_none=True)
        warmup_loss.backward()
        optimizer.zero_grad(set_to_none=True)  # discard the gradients
        torch.cuda.synchronize()
        del warmup_out, warmup_slog, warmup_inputs, warmup_full, warmup_sample
        torch.cuda.empty_cache()
        print(f"[train] warmup ok (qa_id={warmup_row['qa_id']}, "
              f"vram={torch.cuda.memory_allocated()/1024**3:.2f} GB)")
    else:
        print("[train] --pod_mode: skipping WSL2 warmup forward+backward")
```

**`--pod_mode` flag handling**, in `train_student.py:462–478` (CLI registration + cache-dir override):

```python
    ap.add_argument("--pod_mode", action="store_true",
                    help="Real-Linux high-RAM training: skip WSL2 fixes "
                         "(CUDA_LAUNCH_BLOCKING / eager attention / warmup) "
                         "and use upfront vision feature cache + sample-major "
                         "shuffle instead of video-major lazy decode.")
    ap.add_argument("--cache_dir", default=None,
                    help="Override the default logits_cache_subset/ directory "
                         "(useful when the cache lives at logits_cache_subset_final/ "
                         "or any other path).")
    args = ap.parse_args()
    # let --cache_dir override the module-level CACHE_DIR
    global CACHE_DIR
    if args.cache_dir:
        CACHE_DIR = (Path(args.cache_dir)
                     if Path(args.cache_dir).is_absolute()
                     else (ROOT / args.cache_dir))
        print(f"[data] cache_dir override: {CACHE_DIR}")
```

---

## SECTION 7: GP1-vs-GP2 differences (script-detectable items)

### Teacher model loaded

```text
extract_logits_subset.py:
46:MODEL_ID = "cyankiwi/Qwen3.5-27B-AWQ-4bit"
246:                         "Use Qwen/Qwen3.5-4B for fast local pipeline validation.")
297:    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
298:    model = AutoModelForImageTextToText.from_pretrained(
```

### Student model loaded

```text
train_student.py:
1:"""Distill Qwen3.5-27B-AWQ teacher logits into Qwen3.5-0.8B student via LoRA.
31:# WSL2 + Qwen3.5-VL bf16 + LoRA backward hits a driver race that surfaces as
65:STUDENT_ID = "Qwen/Qwen3.5-0.8B"
257:    Qwen3.5-VL also requires `mm_token_type_ids` (text=0, image=1, video=2) to
521:    processor = AutoProcessor.from_pretrained(STUDENT_ID, trust_remote_code=True)
529:    base_model = AutoModelForImageTextToText.from_pretrained(
```

### Sampling configuration locked

```text
extract_logits_subset.py:172–178:
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            top_k=20,
            min_p=0.0,
            repetition_penalty=1.1,
            output_logits=True,
```

### LoRA configuration

```text
train_student.py:563–571:
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
```

CLI defaults at `train_student.py:447–452`:

```text
--lora_r          16
--lora_alpha      32
--lora_dropout    0.05
```

### Loss formulation

`train_student.py` constants (lines 71–73):

```text
TEMPERATURE = 2.0
KL_WEIGHT = 0.7
MAX_NEW_TOKENS = 256
```

`train_student.py` loss computation (lines 696–701):

```python
                s_on_topk = slog.gather(-1, tk_idx)
                p_t = (tk_vals / TEMPERATURE).softmax(-1)
                log_q_s = (s_on_topk / TEMPERATURE).log_softmax(-1)
                kl = F.kl_div(log_q_s, p_t, reduction="batchmean") * (TEMPERATURE ** 2)
                ce = F.cross_entropy(slog, gen_ids)
                loss = KL_WEIGHT * kl + (1 - KL_WEIGHT) * ce
```

Reference implementation in `_smoke_distill.py:10` and `_smoke_distill.py:98–104` (matches the production code verbatim):

```python
FPS, TOPK, TEMPERATURE, KL_WEIGHT = 2.0, 32, 2.0, 0.7
...
    s_on_topk = slog.gather(-1, t_idx)
    p_t = (t_vals / TEMPERATURE).softmax(-1)
    logq_s = (s_on_topk / TEMPERATURE).log_softmax(-1)
    kl = F.kl_div(logq_s, p_t, reduction='batchmean') * (TEMPERATURE ** 2)
    ce = F.cross_entropy(slog, goldid_s)
    loss = KL_WEIGHT * kl + (1 - KL_WEIGHT) * ce
```

### Evaluation metric

`eval_compare.py:36–54` (sliding-window Levenshtein):

```python
def best_window_sim(text: str, gold: str) -> float:
    """Slide a window of len(gold) across text, return max Levenshtein.ratio
    over any window. Catches the gold label appearing anywhere inside the
    student's verbose paragraph, with mild typo tolerance."""
    text = (text or "").lower()
    gold = (gold or "").lower()
    if not text or not gold:
        return 0.0
    if len(text) < len(gold):
        return Levenshtein.ratio(text, gold)
    g = len(gold)
    best = 0.0
    for i in range(len(text) - g + 1):
        s = Levenshtein.ratio(text[i:i + g], gold)
        if s > best:
            best = s
        if best >= 0.999:
            break
    return best
```

`eval_compare.py:57–65` (full-string Levenshtein, used for teacher-text similarity):

```python
def lev_sim(a: str, b: str) -> float:
    """1 − normalized Levenshtein over the full strings. Mirrors the helper
    in train_student.py — used for teacher-text similarity, where both
    strings are paragraphs."""
    a, b = a or "", b or ""
    n = max(len(a), len(b))
    if n == 0:
        return 1.0
    return 1.0 - Levenshtein.distance(a, b) / n
```

`eval_compare.py:72–81` (4-bucket classifier; default threshold 0.80 is the CLI default):

```python
def classify(sim_b: float, sim_a: float, threshold: float) -> str:
    correct_b = sim_b >= threshold
    correct_a = sim_a >= threshold
    if not correct_b and correct_a:
        return "fix"            # wrong → correct
    if correct_b and not correct_a:
        return "regress"        # correct → wrong
    if correct_b and correct_a:
        return "persist_ok"     # correct → correct
    return "persist_bad"        # wrong → wrong
```

---

## SECTION 8: Pipeline script inventory

```text
=== ls -la *.py ===
-rw-r--r-- 1 accy accy  7148 May  5 14:55 _smoke_27b.py
-rw-r--r-- 1 accy accy  6061 May  5 17:58 _smoke_27b_int4.py
-rw-r--r-- 1 accy accy 10532 May  5 18:03 _smoke_27b_int4_v2.py
-rw-r--r-- 1 accy accy  8736 May  5 17:39 _smoke_4b.py
-rw-r--r-- 1 accy accy  7143 May  5 12:53 _smoke_9b.py
-rw-r--r-- 1 accy accy  4947 Apr 23 14:55 _smoke_9b_constrained.py
-rw-r--r-- 1 accy accy  5376 Apr 22 17:54 _smoke_distill.py
-rw-r--r-- 1 accy accy  3896 Apr 23 14:57 answer_video_qa.py
-rw-r--r-- 1 accy accy  2232 Apr 20 16:33 describe_video.py
-rw-r--r-- 1 accy accy  8031 May  7 03:54 download_subset_videos - Copy.py
-rw-r--r-- 1 accy accy  8031 May  7 03:54 download_subset_videos.py
-rw-r--r-- 1 accy accy 11780 May  7 14:47 eval_compare.py
-rw-r--r-- 1 accy accy 17176 May  8 20:04 eval_test_predict.py
-rw-r--r-- 1 accy accy  7067 May  7 02:11 evaluate_test.py
-rw-r--r-- 1 accy accy  1305 May  5 12:17 extract_frames.py
-rw-r--r-- 1 accy accy  9844 Apr 22 16:58 extract_logits_experiment.py
-rw-r--r-- 1 accy accy 18447 May  5 21:12 extract_logits_subset.py
-rw-r--r-- 1 accy accy  9547 Apr 23 11:40 extract_logits_train.py
-rw-r--r-- 1 accy accy 13429 May  5 19:06 sample_subset.py
-rw-r--r-- 1 accy accy  4836 May  7 14:24 sample_test_subset.py
-rw-r--r-- 1 accy accy 37823 May  8 07:09 train_student.py

=== wc -l *.py ===
   143 _smoke_27b.py
   136 _smoke_27b_int4.py
   224 _smoke_27b_int4_v2.py
   173 _smoke_4b.py
   143 _smoke_9b.py
   121 _smoke_9b_constrained.py
   113 _smoke_distill.py
   107 answer_video_qa.py
    74 describe_video.py
   215 download_subset_videos - Copy.py
   215 download_subset_videos.py
   287 eval_compare.py
   389 eval_test_predict.py
   162 evaluate_test.py
    46 extract_frames.py
   222 extract_logits_experiment.py
   441 extract_logits_subset.py
   233 extract_logits_train.py
   397 sample_subset.py
   136 sample_test_subset.py
   837 train_student.py
  4814 total
```

Total Python in the repo: **21 scripts, 4 814 lines**. Production-critical components (the 7 files cited in the report) account for ~2 350 lines of that; the rest are smoke tests, early experiments, and one-off utilities.

---

## SECTION 9: Test prediction file head/tail samples

Each of the four test-prediction jsonls has 1 000 rows (one per stratified test question) with the same shared schema.

### predictions_test_vanilla.jsonl

```text
lines: 1000
```

First 2 lines:

```json
{"qa_id": 28822, "video_id": "aerobic_gymnastics/v_NzCihhjR_NE_c102_00", "question": "What action do the players perform before bent leg jump?", "question_type": "Temporal", "gold": "aerobic gymnastics scissors leap", "teacher_gen_text": "", "student_text": "The players perform a split jump before executing a bent-leg jump."}
{"qa_id": 28812, "video_id": "aerobic_gymnastics/v_NzCihhjR_NE_c102_00", "question": "How many players are performing gymnastics?", "question_type": "Descriptive", "gold": "2", "teacher_gen_text": "", "student_text": "Two players are performing gymnastics."}
```

Last 2 lines:

```json
{"qa_id": 30780, "video_id": "volleyball/v_Fsbu9m-6xQ4_c011_00", "question": "Why is the serve of the right team failed?", "question_type": "Causal", "gold": "volleyball out of bounds", "teacher_gen_text": "", "student_text": "The serve of the right team failed because the ball went into the net."}
{"qa_id": 29460, "video_id": "volleyball/v_g34vS_2KvYY_c010_00", "question": "Does the right team do spike successfully?", "question_type": "Descriptive", "gold": "no", "teacher_gen_text": "", "student_text": "No, the right team does not spike successfully. The spike lands on the left side of the court, away from their intended area."}
```

### predictions_test_trained.jsonl (1-epoch student)

```text
lines: 1000
```

First 2 lines:

```json
{"qa_id": 28822, "video_id": "aerobic_gymnastics/v_NzCihhjR_NE_c102_00", "question": "What action do the players perform before bent leg jump?", "question_type": "Temporal", "gold": "aerobic gymnastics scissors leap", "teacher_gen_text": "", "student_text": "In the clip, at approximately 0:00–0:01, both athletes are seen running forward toward each other on the floor. Just before executing the bent-leg jump (which occurs around 0:01), they perform a **running run** — specifically, a coordinated sprint where one athlete leads and the other follows, maintaining momentum and positioning for the jump.\n\nThe key action immediately preceding the bent-leg jump is this **running run**, which sets up their leaping motion and allows them to generate lift for the jump.\n\nAnswer: Running run."}
{"qa_id": 28812, "video_id": "aerobic_gymnastics/v_NzCihhjR_NE_c102_00", "question": "How many players are performing gymnastics?", "question_type": "Descriptive", "gold": "2", "teacher_gen_text": "", "student_text": "Two athletes are performing on the floor during this aerobic gymnastics routine. They execute synchronized movements including jumps and turns, which are characteristic of group aerobic routines. No additional performers enter or leave the frame during these frames.\n\nAnswer: 2"}
```

Last 2 lines:

```json
{"qa_id": 30780, "video_id": "volleyball/v_Fsbu9m-6xQ4_c011_00", "question": "Why is the serve of the right team failed?", "question_type": "Causal", "gold": "volleyball out of bounds", "teacher_gen_text": "", "student_text": "The clip shows a men’s beach volleyball match at London 2012 (GBR vs ALG). At roughly 0:01–0:02, the player on the right side (ALG) attempts a serve. The receiving team (GBR) has three players positioned near the net; one is already jumping at the net while another dives low to the court. As the serve crosses the net, the diving player makes contact with the ball just before it can reach the floor—this is a “dig” that deflects the ball upward but keeps it in play. Because the ball never lands cleanly inside GBR’s court, the serve fails due to an effective defensive block/dig. Verdict: the serve failed because the receiving team successfully blocked the ball mid-air with a dive."}
{"qa_id": 29460, "video_id": "volleyball/v_g34vS_2KvYY_c010_00", "question": "Does the right team do spike successfully?", "question_type": "Descriptive", "gold": "no", "teacher_gen_text": "", "student_text": "No, the right team does not spike successfully.\n\nAt approximately 0:04–0:06 into the clip, the USA player on the right side (wearing red jersey #18) attempts a powerful spike from the front row. However, the Russian blockers (#7 and #9) time their jump perfectly and form a solid block directly in front of the spiker. The ball makes contact with the block and deflects sharply downward onto the U.S. side of the court, out of bounds near the end line. Since the ball landed on the opponent’s side without being touched by any U.S. defender after the spike, the attack was unsuccessful. The spike failed due to effective blocking by Russia."}
```

### predictions_test_trained_3ep.jsonl (3-epoch student — final)

```text
lines: 1000
```

First 2 lines:

```json
{"qa_id": 28822, "video_id": "aerobic_gymnastics/v_NzCihhjR_NE_c102_00", "sport": "aerobic_gymnastics", "question": "What action do the players perform before bent leg jump?", "question_type": "Temporal", "gold": "aerobic gymnastics scissors leap", "teacher_gen_text": "", "student_text": "In the clip, at approximately 0:00–0:01, two male aerobic gymnasts begin their routine on the mat. Just before they execute a bent-leg jump (where both knees are pulled toward the chest mid-air), they perform a synchronized **leap** — specifically, a **split leap**, where one leg extends forward and the other backward while airborne, arms reaching outward for balance and expression. This leap serves as a dynamic entry into the subsequent acrobatic element. The sequence is clear: first the leap, then the bent-leg jump that follows immediately after. No scoreboard or external cues are needed; the movement itself confirms the preceding action.  \nAnswer: They perform a **split leap**."}
{"qa_id": 28812, "video_id": "aerobic_gymnastics/v_NzCihhjR_NE_c102_00", "sport": "aerobic_gymnastics", "question": "How many players are performing gymnastics?", "question_type": "Descriptive", "gold": "2", "teacher_gen_text": "", "student_text": "There are two athletes performing on the floor during this clip. They execute synchronized movements typical of aerobics gymnastics, including jumps, turns, and dynamic transitions. Both remain active throughout the sequence shown.\n\nAnswer: 2"}
```

Last 2 lines:

```json
{"qa_id": 30780, "video_id": "volleyball/v_Fsbu9m-6xQ4_c011_00", "sport": "volleyball", "question": "Why is the serve of the right team failed?", "question_type": "Causal", "gold": "volleyball out of bounds", "teacher_gen_text": "", "student_text": "The clip shows a men’s indoor volleyball match at London 2012 (GBR vs ALG). At roughly 0.0–0.2 s, the right-side team (in white) sets up an attack; their setter delivers the ball near the net, and their outside hitter jumps to spike. The left side (red) has two blockers at the net and one defender back on the court.\n\nWhat happens: As the spiker contacts the ball, his contact point is slightly too far inside the block. The ball hits the top of the blockers’ hands instead of clearing the net. Instead of going out of bounds or into the net, the ball deflects backward toward the attacking team’s own side. It lands inbounds on the red team’s court—visible as the ball bouncing once on the orange playing area before rolling away.\n\nCause: Poor timing/positioning relative to the blockers led to a blocked attempt that didn’t clear the net. Counterfactually, if the attacker had aimed more sharply downward or used a tip/roll shot to bypass the block, the ball likely would have landed untouched in the opponent’s court.\n\nVerdict: The serve (in this case, the attack) failed because the ball was successfully"}
{"qa_id": 29460, "video_id": "volleyball/v_g34vS_2KvYY_c010_00", "sport": "volleyball", "question": "Does the right team do spike successfully?", "question_type": "Descriptive", "gold": "no", "teacher_gen_text": "", "student_text": "The clip shows a women’s indoor volleyball match at London 2012 between Russia (RUS) and USA. The camera angle is wide, showing both sides of the net. At approximately 0.4–0.5 seconds into the clip, the player on the right side of the court (USA, in red/blue uniforms) executes an attack: she approaches from the back row, jumps near the antenna, and swings her arm to hit the ball sharply downward toward the Russian side.\n\nAt the moment of contact, the ball is clearly traveling fast and low, aimed well inside the boundary lines of the opponent’s court. A Russian defender attempts to block but fails to stop the ball; another Russian player dives late and cannot reach it either. The ball lands untouched on the Russian side of the court — specifically, just past the front row defenders’ reach, near the sideline. There is no visual indication that the ball touched the net or went out of bounds after contact.\n\nTherefore, based solely on the physical trajectory and final position of the ball relative to the court boundaries — which is fully visible and unambiguous — the spike resulted in a successful point for the attacking team.\n\nYes, the right team spiked successfully."}
```

### predictions_test_teacher.jsonl (27B-AWQ-Int4 teacher reference)

```text
lines: 1000
```

First 2 lines:

```json
{"qa_id": 28822, "video_id": "aerobic_gymnastics/v_NzCihhjR_NE_c102_00", "sport": "aerobic_gymnastics", "question": "What action do the players perform before bent leg jump?", "question_type": "Temporal", "gold": "aerobic gymnastics scissors leap", "teacher_gen_text": "", "student_text": "The clip shows two female rhythmic gymnasts performing on a carpeted court. The sequence begins at 0:00 with both athletes standing apart, arms raised. At approximately 0:01, they simultaneously execute a leap where their legs are extended straight out in front of them, parallel to the ground, while their bodies remain upright. This is a classic \"split leap\" or \"grand jeté.\" Immediately following this leap, at roughly 0:02, they land and transition into a squatting position with one leg bent and the other extended back, which corresponds to the \"bent leg jump\" mentioned in your query. Therefore, the specific acrobatic skill performed directly prior to the bent leg movement is the split leap.\n\nAnswer: Split leap."}
{"qa_id": 28812, "video_id": "aerobic_gymnastics/v_NzCihhjR_NE_c102_00", "sport": "aerobic_gymnastics", "question": "How many players are performing gymnastics?", "question_type": "Descriptive", "gold": "2", "teacher_gen_text": "", "student_text": "There are two athletes performing on the floor. Both are actively engaged in synchronized movements typical of aerobic gymnastics routines. One is in front wearing a dark leotard with red and white accents; the other is slightly behind, wearing a white top with maroon shorts. They move together through jumps, turns, and arm extensions. No third performer enters the frame during this segment.\n\nAnswer: 2"}
```

Last 2 lines:

```json
{"qa_id": 30780, "video_id": "volleyball/v_Fsbu9m-6xQ4_c011_00", "sport": "volleyball", "question": "Why is the serve of the right team failed?", "question_type": "Causal", "gold": "volleyball out of bounds", "teacher_gen_text": "", "student_text": "The clip shows a volleyball match at the London 2012 Olympics. The team on the right (in white) is serving. At approximately 0.1–0.2 seconds into the clip, we can observe the server making contact with the ball. However, instead of clearing the net and entering the opponent’s court, the ball travels upward but fails to cross over the net — it appears to hit the top of the net or fall back onto the server’s own side immediately after contact.\n\nThis is a classic case of a “net” error during service: the ball touches the net and does not pass over into the opposing court. According to FIVB rules, if a served ball contacts the net and does not enter the opponent’s playing area, the serve is considered unsuccessful, and the receiving team gains possession.\n\nThere is no indication that the ball went out of bounds or was blocked mid-air; the failure occurs purely due to insufficient power or trajectory control upon contact — the ball simply doesn’t clear the net.\n\nVerdict: The serve failed because the ball contacted the net and did not cross into the opponent’s court. This is a direct violation of successful serve requirements under official volleyball rules. No scoreboard or external cue is needed — the"}
{"qa_id": 29460, "video_id": "volleyball/v_g34vS_2KvYY_c010_00", "sport": "volleyball", "question": "Does the right team do spike successfully?", "question_type": "Descriptive", "gold": "no", "teacher_gen_text": "", "student_text": "Yes.\n\nAt roughly 0:04–0:06, the player on the right side (red jersey) executes a jump approach, contacts the ball high near the net, and drives it sharply downward into the left court’s front zone. The ball clearly crosses the plane of the net and lands inside the opponent’s court without being touched by any defender. There is no block contact that redirects the ball back; the trajectory and impact are consistent with a successful attack. Therefore, the spike succeeds."}
```

Schema notes:

- All four files share the keys `qa_id`, `video_id`, `question`, `question_type`, `gold`, `teacher_gen_text`, `student_text`.
- The newer two files (`predictions_test_trained_3ep.jsonl`, `predictions_test_teacher.jsonl`) additionally carry an explicit `sport` field — added after `eval_test_predict.py` was extended with `--subset_file` support so per-sport breakdowns work correctly even when the curated 8-sport taxonomy is in play. The earlier two files (`predictions_test_vanilla.jsonl`, `predictions_test_trained.jsonl`) lack this field; `eval_compare.py` falls back to the video-id prefix for them.
- `teacher_gen_text` is empty for all four files because no teacher logit cache exists for upstream test-set rows (extraction was done only on the train+val subset). This is correct.
- `student_text` is the model's actual answer regardless of which model produced it — vanilla student, LoRA-adapted student, or 27B teacher. The schema is uniform so `eval_compare.py` can pair any two files.

---

## SECTION 10: Anything you noticed that I didn't ask about

Several artifacts and observations that may be relevant for the appendix:

### 10.1 Duplicate / leftover prediction files

The project root contains several `.jsonl` files that are **not** the canonical four used in the report:

```text
predictions_test_0p8b copy.jsonl       — Windows-style duplicate from an early attempt (Apr 20)
predictions_test_0p8b.jsonl            — early constrained-prompt run (Apr 22)
predictions_test_0p8b_lora.jsonl       — early LoRA smoke (May 7, 28 KB)
predictions_test_subset_trained.jsonl  — abortive 2,793-Q curated test run (May 7, 12 KB)
predictions_test_subset_vanilla.jsonl  — same, vanilla (May 7, 13 KB)
```

None of these are used in the headline report. They are leftovers from earlier exploratory attempts (the constrained-prompt path that was later dropped, and the 2 793-Q curated test run that was killed in favour of going back to the 1 000-Q stratified sample). The user can safely delete or archive these to keep the project root clean for publication.

### 10.2 `download_subset_videos - Copy.py`

There is a Windows-style duplicate of `download_subset_videos.py` named `download_subset_videos - Copy.py`. Same byte size (8 031), so functionally identical. Should be deleted before publication — having "Copy" in a filename in a thesis appendix looks unintentional.

### 10.3 `eval_before.jsonl` / `eval_after.jsonl` are now stale (8 entries each)

The held-out flip-rate report cited in the GP2 source doc (§5.8 of `GP2_REPORT_SOURCE.md` — *18 fix / 2 regress on n=100*) was generated by the 1-epoch validation run. That run wrote 100-row `eval_before.jsonl` / `eval_after.jsonl` files and the file `eval_flip_report.md`. **A subsequent smoke test with `--n_holdout 8` overwrote** `eval_before.jsonl` and `eval_after.jsonl` to 8 rows each. The original 100-row report (`eval_flip_report.md`) is still preserved on disk, so the cited n=100 numbers remain reproducible from that markdown file — but the source jsonls themselves have been clobbered. If web Claude needs to rerun the analysis, the n=100 numbers must be quoted from `eval_flip_report.md` directly (the markdown report).

### 10.4 Two student adapters on disk

```text
student_lora/                  — 1-epoch validation adapter (May 6, 25 MB)
student_lora_full3epoch/       — 3-epoch final adapter (May 8, 25 MB)
```

Both are still present. The headline numbers in the report are from `student_lora_full3epoch/`. The 1-epoch adapter (`student_lora/`) was used for the held-out flip-rate validation (§5.8 in the source doc) and is preserved for reproducing those numbers if needed.

### 10.5 The `requirements.txt` file is incomplete

`peft`, `python-Levenshtein`, and a few other packages used in the training and evaluation pipelines are not listed in `requirements.txt`. The file was authored early in the project (with a Colab-aimed deployment intent that didn't end up being relevant) and was never updated as new deps came in. Anyone trying to reproduce the project from `requirements.txt` alone would have to install peft and python-Levenshtein manually. The user may want to regenerate `requirements.txt` with `pip freeze` before publication.

### 10.6 Empty .pt files in the cache

3 511 of the 11 680 `.pt` files in `logits_cache_subset_final/` are 0-byte EOFError-on-load placeholders. The training loop's `load_available_samples()` silently skips them (the `try / except Exception: continue` clause around `torch.load`). This is documented in the GP2 source doc §5.2 but worth restating: the cache has 11 680 entries on disk but only **8 169 valid** when loaded. Anyone re-counting the cache for the report should distinguish "files on disk" from "valid samples".

The pod's working copy of the cache appears to have been clean — the 3-epoch training run processed 35 016 / 3 ≈ 11 672 valid samples per epoch, much closer to 11 680 than to 8 169. The 0-byte files locally are most likely a side effect of partial / interrupted rsync transfers, not actual extraction failures.

### 10.7 `_smoke_*` files document the prompt-iteration history

The smoke-test scripts (`_smoke_27b.py`, `_smoke_27b_int4.py`, `_smoke_27b_int4_v2.py`, `_smoke_4b.py`, `_smoke_9b.py`, `_smoke_9b_constrained.py`, `_smoke_distill.py`) collectively encode the four-pass prompt iteration the GP2 report references in §3.4 of the source doc. If the report needs a "what each pass tried" appendix, those files have the prompt-string history.

### 10.8 Two existing markdown reports are reused as feeders

The existing files `METHODOLOGY_REPORT.md` (391 lines, academic prose) and `vanilla_train_test_logits_handling.md` (884 lines, technical reference) are referenced extensively as feeders in `GP2_REPORT_SOURCE.md`. Web Claude should not re-derive from raw artifacts if those feeders already cover the needed material.

### 10.9 The pod was at IP `79.117.54.182:50274` for the most recent rentals

The user's training and teacher-eval rentals during the final reporting period used `ssh -p 50274 root@79.117.54.182`. The Vast.ai dashboard should show the rental records under that host. Earlier rentals (extraction phase) used different IPs/ports — the user should consolidate the billing records when populating Section 5.
