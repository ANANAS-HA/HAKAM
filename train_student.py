"""Distill Qwen3.5-27B-AWQ teacher logits into Qwen3.5-0.8B student via LoRA.

Pipeline:
  1. Discover local .pt cache files in `logits_cache_subset/` cross-checked
     against `logits_cache_subset.manifest.jsonl`.
  2. Stratified-by-question_type holdout of N samples for before/after eval.
  3. Eval BEFORE: pristine 0.8B + same teacher prompt + same sampling config.
     Save `eval_before.jsonl`.
  4. Train: LoRA on attention + FFN, vision encoder frozen.  Loss = CE + sparse
     top-K KL with Hinton temperature, identical to `_smoke_distill.py:100-104`.
  5. Eval AFTER: same model + LoRA on same holdout. Save `eval_after.jsonl`.
  6. Save adapter to `student_lora/` and a side-by-side `eval_comparison.md`.

Usage:
    python train_student.py                         # full validation run
    python train_student.py --max_samples 20 --epochs 1   # smoke
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# WSL2 + Qwen3.5-VL bf16 + LoRA backward hits a driver race that surfaces as
# "CUDA driver error: unknown error" on the first training backward. Forcing
# synchronous CUDA op submission makes this go away. ~5-10% slowdown is
# acceptable to keep the pipeline running. The default below keeps WSL2
# users covered; pod runs (--pod_mode) skip it.
_POD_MODE = "--pod_mode" in sys.argv
if not _POD_MODE:
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, PeftModel, get_peft_model

# Workaround for peft<->gptqmodel version skew: peft's dispatch_awq runs for
# every target module and tries `from gptqmodel.nn_modules.qlinear.gemm_awq
# import AwqGEMMQuantLinear`, but the installed gptqmodel renamed that class
# to `AwqGEMMLinear`. The 0.8B student is plain bf16 (not AWQ-quantized) so
# this dispatcher is irrelevant — replace it with a no-op.
import peft.tuners.lora.model as _peft_lora_model

def _noop_dispatch_awq(target, adapter_name, config, **kwargs):
    return None

_peft_lora_model.dispatch_awq = _noop_dispatch_awq

import Levenshtein


# ---------- constants — must match the teacher's run ----------

ROOT = Path(__file__).parent
STUDENT_ID = "Qwen/Qwen3.5-0.8B"
CACHE_DIR = ROOT / "logits_cache_subset"
MANIFEST_PATH = ROOT / "logits_cache_subset.manifest.jsonl"

FPS = 2.0
TOPK = 32
TEMPERATURE = 2.0
KL_WEIGHT = 0.7
MAX_NEW_TOKENS = 256

# Verbatim from extract_logits_subset.py; the student must see the same prompt
# the 27B teacher saw, otherwise the KL signal is mis-anchored.
SYSTEM_PROMPT = (
    "You are an elite sports video analyst with deep expertise across basketball, football, "
    "volleyball, and gymnastics (including aerobic gymnastics, vault, uneven bars, balance beam, "
    "and floor exercise). You watch short game or performance clips and parse them the way a "
    "coach or color commentator would: identifying the sport, naming specific techniques and "
    "skills (e.g. 2-point shot, spike, push-up, split, round-off, flic-flac, giant circle), "
    "counting discrete events and the number of athletes involved, tracking temporal order "
    "(what comes before / after what), and reasoning about cause and effect (why an action "
    "succeeded or failed, what a counterfactual outcome would have been).\n\n"
    "When given a question, think it through step by step — describe what you see in the clip, "
    "locate the moment the question refers to, and then give a precise, expert answer. Be "
    "decisive: sports are concrete, so give committed answers rather than hedged ones. "
    "If the question is a yes / no, still justify briefly. If it asks 'how many', count carefully. "
    "If it asks for the name of an action, use the technical term.\n\n"
    "**Outcome questions must be answered from the action itself, not from interface elements.** "
    "When a question asks whether an action *succeeded* (did the shot go in, did the team score, "
    "did the spike land, did the dismount stick), your verdict must come from observing the action's "
    "physical result: ball relative to rim/net, body relative to landing surface, ball crossing the "
    "goal line. **Do not use scoreboards, scorelines, point counters, or referee gestures as evidence "
    "of success or failure.** Scoreboards are graphical UI, lag the action by 1-3 seconds, and may not "
    "update within the clip's duration. If the play is cut off before the result is visually resolved, "
    "say so explicitly and answer from what was visible — do not infer success or failure from a static "
    "score.\n\n"
    "**Be concise.** Aim for around 300-500 tokens total. Identify the moment, state the evidence, "
    "commit to a verdict. Do not re-examine the same evidence twice or hedge between interpretations. "
    "Brevity over rumination."
)


# ---------- video path resolution (mirrors extract_logits_subset.py) ----------

def resolve_video_path(video_key: str) -> Path | None:
    sport, stem = video_key.split("/", 1)
    videos = ROOT / "videos"
    exts = (".avi", ".mp4", ".mkv", ".webm")
    cands = []
    if sport == "fg":
        for d in (videos / f"finegym_0{i}" for i in range(1, 6)):
            cands += [d / f"{stem}{e}" for e in exts]
    elif sport == "volleyball":
        cands += [videos / "volleyball" / "volleyball" / f"{stem}{e}" for e in exts]
        cands += [videos / "volleyball" / f"{stem}{e}" for e in exts]
    else:
        cands += [videos / sport / f"{stem}{e}" for e in exts]
    return next((p for p in cands if p.exists()), None)


def build_messages(video_path: Path, question: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video", "video": str(video_path), "fps": FPS,
             "max_frames": 32, "max_pixels": 256 * 28 * 28},
            {"type": "text", "text": question},
        ]},
    ]


def apply_chat_template(processor, msgs):
    return processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


# ---------- data discovery ----------

def load_available_samples() -> list[dict]:
    """Scan logits_cache_subset/*.pt — each .pt carries its own metadata, so
    we don't depend on the manifest jsonl (which lags real cache contents
    when files arrive via rsync)."""
    if not CACHE_DIR.exists():
        raise FileNotFoundError(f"Cache dir not found: {CACHE_DIR}")
    rows = []
    paths = sorted(CACHE_DIR.glob("*.pt"))
    for pt in tqdm(paths, desc="scanning .pt cache", unit="f"):
        try:
            d = torch.load(pt, map_location="cpu", weights_only=False)
        except Exception:
            continue
        vp = resolve_video_path(d["video_id"])
        if vp is None:
            continue
        rows.append({
            "qa_id":         int(d["qa_id"]),
            "video_id":      d["video_id"],
            "video_path":    vp,
            "split":         d.get("split", "train"),
            "sport":         d.get("sport", "?"),
            "question_type": d.get("question_type", "?"),
            "pt_path":       pt,
        })
    return rows


def stratified_holdout(rows: list[dict], n: int, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Take ~equal counts per question_type, shuffling deterministically."""
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(r)
    types = sorted(by_type.keys())
    per_type = max(1, n // len(types))
    holdout: list[dict] = []
    for t in types:
        bucket = by_type[t][:]
        rng.shuffle(bucket)
        holdout.extend(bucket[:per_type])
    rng.shuffle(holdout)
    holdout = holdout[:n]
    holdout_ids = {r["qa_id"] for r in holdout}
    train = [r for r in rows if r["qa_id"] not in holdout_ids]
    return train, holdout


def stratified_holdout_by_sport(rows: list[dict], per_sport: dict[str, int],
                                 seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Per-sport holdout, internally stratified by question_type within each
    sport. Lets us evaluate seen vs. unseen sports separately."""
    rng = random.Random(seed)
    by_sport: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sport[r["sport"]].append(r)
    holdout: list[dict] = []
    for sport, n in per_sport.items():
        cands = by_sport.get(sport, [])
        if not cands:
            print(f"  WARN: requested {n} from '{sport}' but 0 available")
            continue
        if len(cands) < n:
            print(f"  WARN: only {len(cands)} '{sport}' samples available "
                  f"(asked for {n})")
            n = len(cands)
        by_qt: dict[str, list[dict]] = defaultdict(list)
        for r in cands:
            by_qt[r["question_type"]].append(r)
        qts = sorted(by_qt.keys())
        per_qt = max(1, n // len(qts))
        picked: list[dict] = []
        for qt in qts:
            bucket = by_qt[qt][:]
            rng.shuffle(bucket)
            picked.extend(bucket[:per_qt])
        # if rounding shorted us, pad from the leftover pool
        if len(picked) < n:
            extra = [r for r in cands if r["qa_id"] not in {p["qa_id"] for p in picked}]
            rng.shuffle(extra)
            picked.extend(extra[:n - len(picked)])
        rng.shuffle(picked)
        holdout.extend(picked[:n])
    holdout_ids = {r["qa_id"] for r in holdout}
    train = [r for r in rows if r["qa_id"] not in holdout_ids]
    return train, holdout


# ---------- input construction ----------

def build_prompt_inputs(processor, video_path: Path, question: str,
                        cached_video_inputs=None):
    """Return (inputs_on_cuda, prompt_len). Decodes video unless given cache."""
    msgs = build_messages(video_path, question)
    text = apply_chat_template(processor, msgs)
    if cached_video_inputs is None:
        image_inputs, video_inputs = process_vision_info(msgs)
    else:
        image_inputs, video_inputs = None, cached_video_inputs
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to("cuda")
    prompt_len = inputs["input_ids"].shape[1]
    return inputs, prompt_len


def append_gen_tokens(inputs: dict, gen_tokens: torch.Tensor) -> dict:
    """Concat teacher gen_tokens to prompt input_ids + extend attention_mask.

    Used for teacher-forcing: the student's logits at positions
    [prompt_len-1 .. prompt_len-1+T] are the per-step predictions for each
    gen_token, which is exactly what the recorded teacher topK indices align to.

    Qwen3.5-VL also requires `mm_token_type_ids` (text=0, image=1, video=2) to
    be the same length as input_ids — extend it with zeros (the appended gen
    tokens are all text).
    """
    out = dict(inputs)
    device = inputs["input_ids"].device
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


# ---------- generation eval ----------

def run_generation_eval(model, processor, holdout: list[dict],
                        sample_pts: dict[int, dict], desc: str) -> list[dict]:
    """Run model.generate() under the teacher's sampling config on holdout set."""
    model.eval()
    out = []
    for r in tqdm(holdout, desc=desc, unit="q"):
        sample = sample_pts[r["qa_id"]]
        inputs, prompt_len = build_prompt_inputs(
            processor, r["video_path"], sample["question"],
        )
        try:
            with torch.inference_mode():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.9,
                    top_k=20,
                    min_p=0.0,
                    repetition_penalty=1.1,
                )
            new_tokens = gen[0, prompt_len:].cpu()
            text = processor.batch_decode(
                [new_tokens], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        except Exception as e:
            text = f"<gen_error: {type(e).__name__}: {e}>"
        out.append({
            "qa_id":            r["qa_id"],
            "video_id":         r["video_id"],
            "question":         sample["question"],
            "question_type":    sample["question_type"],
            "gold":             sample["answer"],
            "teacher_gen_text": sample["gen_text"],
            "student_text":    text,
        })
        # free intermediate GPU buffers between samples
        del inputs
        torch.cuda.empty_cache()
    return out


# ---------- comparison metrics ----------

def label_in(text: str, gold: str) -> bool:
    return gold.strip().lower() in (text or "").strip().lower()


def lev_sim(a: str, b: str) -> float:
    """1 − normalized Levenshtein. Falls in [0, 1], higher = more similar."""
    a, b = a or "", b or ""
    if not a and not b:
        return 1.0
    n = max(len(a), len(b))
    if n == 0:
        return 1.0
    return 1.0 - Levenshtein.distance(a, b) / n


def write_comparison_md(before: list[dict], after: list[dict],
                        path: Path) -> dict:
    by_id_after = {r["qa_id"]: r for r in after}
    rows = []
    n = 0
    sum_lab_b, sum_lab_a = 0, 0
    sum_tsim_b, sum_tsim_a = 0.0, 0.0
    sum_gsim_b, sum_gsim_a = 0.0, 0.0
    for b in before:
        a = by_id_after.get(b["qa_id"])
        if a is None:
            continue
        n += 1
        lab_b = int(label_in(b["student_text"], b["gold"]))
        lab_a = int(label_in(a["student_text"], a["gold"]))
        tsim_b = lev_sim(b["student_text"], b["teacher_gen_text"])
        tsim_a = lev_sim(a["student_text"], a["teacher_gen_text"])
        gsim_b = lev_sim(b["student_text"], b["gold"])
        gsim_a = lev_sim(a["student_text"], a["gold"])
        sum_lab_b += lab_b; sum_lab_a += lab_a
        sum_tsim_b += tsim_b; sum_tsim_a += tsim_a
        sum_gsim_b += gsim_b; sum_gsim_a += gsim_a
        rows.append({
            **b,
            "student_before": b["student_text"],
            "student_after":  a["student_text"],
            "label_before": lab_b, "label_after": lab_a,
            "tsim_before":  tsim_b, "tsim_after": tsim_a,
            "gsim_before":  gsim_b, "gsim_after": gsim_a,
            "delta_tsim":   tsim_a - tsim_b,
        })

    rows.sort(key=lambda r: r["delta_tsim"], reverse=True)
    summary = {
        "n":                n,
        "label_in_before":  sum_lab_b / max(1, n),
        "label_in_after":   sum_lab_a / max(1, n),
        "teacher_sim_before": sum_tsim_b / max(1, n),
        "teacher_sim_after":  sum_tsim_a / max(1, n),
        "gold_sim_before":  sum_gsim_b / max(1, n),
        "gold_sim_after":   sum_gsim_a / max(1, n),
    }

    with path.open("w") as f:
        f.write("# Student before/after distillation — held-out comparison\n\n")
        f.write(f"Held-out size: **{n}**\n\n")
        f.write("## Aggregate metrics\n\n")
        f.write("| metric | before | after | Δ |\n|---|---|---|---|\n")
        f.write(f"| label-in-answer rate | {summary['label_in_before']:.3f} "
                f"| {summary['label_in_after']:.3f} "
                f"| {summary['label_in_after']-summary['label_in_before']:+.3f} |\n")
        f.write(f"| teacher-text similarity | {summary['teacher_sim_before']:.3f} "
                f"| {summary['teacher_sim_after']:.3f} "
                f"| {summary['teacher_sim_after']-summary['teacher_sim_before']:+.3f} |\n")
        f.write(f"| gold-label similarity | {summary['gold_sim_before']:.3f} "
                f"| {summary['gold_sim_after']:.3f} "
                f"| {summary['gold_sim_after']-summary['gold_sim_before']:+.3f} |\n\n")
        f.write("## Per-sample (sorted by Δ teacher-similarity)\n\n")
        for r in rows:
            f.write(f"### qa_id={r['qa_id']}  ({r['question_type']})\n")
            f.write(f"- **Q:** {r['question']}\n")
            f.write(f"- **Gold:** `{r['gold']}`\n")
            f.write(f"- **Δ tsim:** {r['delta_tsim']:+.3f}  "
                    f"label match: {r['label_before']}→{r['label_after']}\n\n")
            f.write(f"  **Teacher (27B) gen:**\n  > {r['teacher_gen_text'][:600]}\n\n")
            f.write(f"  **Student BEFORE:**\n  > {r['student_before'][:600]}\n\n")
            f.write(f"  **Student AFTER:**\n  > {r['student_after'][:600]}\n\n---\n\n")
    return summary


# ---------- training ----------

def freeze_visual(model):
    """Set requires_grad=False on the vision tower so no gradient flows.

    Qwen2.5/3.5-VL exposes the visual encoder as `model.visual` (sometimes
    `model.vision_tower`). Walk both possible attribute names defensively.
    """
    n_frozen = 0
    for name, mod in model.named_modules():
        if any(k in name for k in ("visual", "vision_tower", "vision_model")):
            for p in mod.parameters(recurse=False):
                if p.requires_grad:
                    p.requires_grad = False
                    n_frozen += 1
    return n_frozen


def freeze_lora_under_visual(peft_model):
    """After PEFT injection, kill any LoRA params that landed inside the vision
    tower — keeps the visual representation frozen end-to-end."""
    for name, p in peft_model.named_parameters():
        if "lora_" in name and any(k in name for k in ("visual", "vision_tower", "vision_model")):
            p.requires_grad = False


def write_jsonl(rows: list[dict], path: Path):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items()
                                if k != "video_path"}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--n_holdout", type=int, default=8,
                    help="Total holdout, qtype-stratified (ignored if --holdout_per_sport set).")
    ap.add_argument("--holdout_per_sport", type=str, default=None,
                    help='Per-sport holdout, e.g. "Gym=50,volleyball=50". '
                         "Internally stratified by question_type within each sport.")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Cap on training samples (debug). Holdout stays as configured.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adapter_dir", default="student_lora")
    ap.add_argument("--eval_before", default="eval_before.jsonl")
    ap.add_argument("--eval_after", default="eval_after.jsonl")
    ap.add_argument("--report", default="eval_comparison.md")
    ap.add_argument("--skip_before", action="store_true",
                    help="Skip the BEFORE eval (e.g. resuming after eval already ran).")
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

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 1. Discover data
    rows = load_available_samples()
    print(f"[data] {len(rows)} samples available locally")
    if not rows:
        raise SystemExit("No usable samples — check logits_cache_subset/.")
    sports_total = defaultdict(int)
    for r in rows:
        sports_total[r["sport"]] += 1
    print(f"[data] by sport: {dict(sports_total)}")

    if args.holdout_per_sport:
        per_sport: dict[str, int] = {}
        for tok in args.holdout_per_sport.split(","):
            k, _, v = tok.partition("=")
            per_sport[k.strip()] = int(v)
        print(f"[data] per-sport holdout request: {per_sport}")
        train_rows, holdout_rows = stratified_holdout_by_sport(
            rows, per_sport, args.seed)
    else:
        train_rows, holdout_rows = stratified_holdout(rows, args.n_holdout, args.seed)

    print(f"[data] holdout={len(holdout_rows)} train={len(train_rows)}")
    sp_h = defaultdict(int); qt_h = defaultdict(int)
    for r in holdout_rows:
        sp_h[r["sport"]] += 1; qt_h[r["question_type"]] += 1
    print(f"[data] holdout by sport: {dict(sp_h)}")
    print(f"[data] holdout by qtype: {dict(qt_h)}")
    if args.max_samples:
        train_rows = train_rows[:args.max_samples]
        print(f"[data] capping train to {len(train_rows)} samples (debug)")

    # 2. Pre-load .pt files for holdout (cheap; small) for use during eval
    holdout_pts = {r["qa_id"]: torch.load(r["pt_path"], map_location="cpu",
                                          weights_only=False)
                   for r in holdout_rows}

    # 3. Load student processor + base model in bf16
    print(f"[model] loading {STUDENT_ID} ...")
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

    # 4. Eval BEFORE — pristine 0.8B under teacher's prompt + sampling config
    if not args.skip_before:
        print(f"\n[eval BEFORE] generating on {len(holdout_rows)} held-out samples ...")
        eval_before = run_generation_eval(
            base_model, processor, holdout_rows, holdout_pts, "BEFORE",
        )
        write_jsonl(eval_before, ROOT / args.eval_before)
        print(f"[eval BEFORE] saved -> {args.eval_before}")
    else:
        eval_before = [json.loads(l) for l in (ROOT / args.eval_before).read_text().splitlines() if l.strip()]
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

    # 5. Apply LoRA + freeze visual
    print("\n[train] applying LoRA + freezing visual encoder ...")
    n_frozen_visual = freeze_visual(base_model)
    print(f"[train] froze {n_frozen_visual} visual modules pre-PEFT")
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_cfg)
    freeze_lora_under_visual(peft_model)
    peft_model.config.use_cache = False
    peft_model.print_trainable_parameters()

    # 6. Optimizer on LoRA params only
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    # 7. Group questions per video — we'll decode each video on-demand inside
    # the training loop and drop it before moving on, so RAM peak is one
    # video instead of all of them. (525 videos × ~50 MB each blew past
    # WSL's RAM cap when pre-cached.)
    by_video: dict[str, list[dict]] = defaultdict(list)
    video_path_of: dict[str, Path] = {}
    for r in train_rows:
        by_video[r["video_id"]].append(r)
        video_path_of[r["video_id"]] = r["video_path"]
    print(f"\n[train] {len(train_rows)} questions across {len(by_video)} "
          f"unique videos (avg {len(train_rows)/max(1,len(by_video)):.1f} q/video)")

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

    # 8. Train loop. Two paths:
    #   - default (WSL2-safe): video-major lazy decoding, RAM peak ~one video.
    #   - --pod_mode (real Linux, ~172 GB RAM): upfront vision feature cache
    #     of every unique training video, then sample-major shuffled iteration.
    print(f"[train] {args.epochs} epochs over {len(train_rows)} samples "
          f"(lr={args.lr}, T={TEMPERATURE}, α={KL_WEIGHT}, "
          f"mode={'upfront-cache' if args.pod_mode else 'video-major-lazy'})")
    rng = random.Random(args.seed)
    step = 0
    losses = []

    if args.pod_mode:
        # Pre-decode every unique training video once into CPU RAM.
        print(f"[train] building upfront vision cache for {len(by_video)} "
              f"unique videos ...")
        vision_cache: dict[str, object] = {}
        for vid in tqdm(list(by_video.keys()), desc="vision-cache", unit="vid"):
            try:
                _, vision_cache[vid] = process_vision_info(
                    build_messages(video_path_of[vid], "x")
                )
            except Exception as e:
                print(f"  [WARN] decord failed for {vid}: {e}")
        print(f"[train] cache built ({len(vision_cache)} videos in CPU RAM)")

        for epoch in range(args.epochs):
            order = train_rows[:]
            rng.shuffle(order)
            ce_run, kl_run, n_run = 0.0, 0.0, 0
            bar = tqdm(order, desc=f"epoch {epoch+1}/{args.epochs}", unit="q")
            for r in bar:
                vid = r["video_id"]
                if vid not in vision_cache:
                    continue
                sample = torch.load(r["pt_path"], map_location="cpu",
                                    weights_only=False)
                gen_tokens = sample["gen_tokens"]
                topk_values = sample["topk_values"].float()
                topk_indices = sample["topk_indices"].long()
                T = int(sample["seq_len"])
                if T == 0:
                    continue

                try:
                    inputs, prompt_len = build_prompt_inputs(
                        processor, r["video_path"], sample["question"],
                        cached_video_inputs=vision_cache[vid],
                    )
                except Exception as e:
                    print(f"  [WARN] prompt build failed qa_id={r['qa_id']}: {e}")
                    continue

                inputs_full = append_gen_tokens(inputs, gen_tokens)
                try:
                    logits = peft_model(**inputs_full).logits
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  [OOM] qa_id={r['qa_id']} T={T} "
                          f"prompt_len={prompt_len} — skipping")
                    continue

                slog = logits[0, prompt_len - 1: prompt_len - 1 + T].float()
                gen_ids = gen_tokens.to(slog.device)
                tk_idx = topk_indices.to(slog.device)
                tk_vals = topk_values.to(slog.device)

                s_on_topk = slog.gather(-1, tk_idx)
                p_t = (tk_vals / TEMPERATURE).softmax(-1)
                log_q_s = (s_on_topk / TEMPERATURE).log_softmax(-1)
                kl = F.kl_div(log_q_s, p_t, reduction="batchmean") * (TEMPERATURE ** 2)
                ce = F.cross_entropy(slog, gen_ids)
                loss = KL_WEIGHT * kl + (1 - KL_WEIGHT) * ce

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()

                ce_run += ce.item(); kl_run += kl.item(); n_run += 1
                losses.append({"epoch": epoch + 1, "step": step,
                               "ce": ce.item(), "kl": kl.item(),
                               "loss": loss.item(), "qa_id": r["qa_id"]})
                step += 1
                if n_run % 16 == 0:
                    bar.set_postfix(ce=f"{ce_run/n_run:.3f}",
                                    kl=f"{kl_run/n_run:.3f}")
                del logits, slog, inputs, inputs_full
            print(f"[train] epoch {epoch+1}: avg CE={ce_run/max(1,n_run):.3f} "
                  f"avg KL={kl_run/max(1,n_run):.3f}  ({n_run} steps)")
        del vision_cache
        gc.collect()
    else:
        for epoch in range(args.epochs):
            video_ids = list(by_video.keys())
            rng.shuffle(video_ids)
            ce_run, kl_run, n_run = 0.0, 0.0, 0
            bar = tqdm(total=len(train_rows),
                       desc=f"epoch {epoch+1}/{args.epochs}", unit="q")
            for vid in video_ids:
                qs = by_video[vid][:]
                rng.shuffle(qs)
                try:
                    _, video_inputs = process_vision_info(
                        build_messages(video_path_of[vid], "x")
                    )
                except Exception as e:
                    print(f"  [WARN] decord failed for {vid}: {e}")
                    bar.update(len(qs))
                    continue

                for r in qs:
                    sample = torch.load(r["pt_path"], map_location="cpu",
                                        weights_only=False)
                    gen_tokens = sample["gen_tokens"]
                    topk_values = sample["topk_values"].float()
                    topk_indices = sample["topk_indices"].long()
                    T = int(sample["seq_len"])
                    if T == 0:
                        bar.update(1); continue

                    try:
                        inputs, prompt_len = build_prompt_inputs(
                            processor, r["video_path"], sample["question"],
                            cached_video_inputs=video_inputs,
                        )
                    except Exception as e:
                        print(f"  [WARN] prompt build failed qa_id={r['qa_id']}: {e}")
                        bar.update(1); continue

                    inputs_full = append_gen_tokens(inputs, gen_tokens)
                    try:
                        logits = peft_model(**inputs_full).logits
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        print(f"  [OOM] qa_id={r['qa_id']} T={T} "
                              f"prompt_len={prompt_len} — skipping")
                        bar.update(1); continue

                    slog = logits[0, prompt_len - 1: prompt_len - 1 + T].float()
                    gen_ids = gen_tokens.to(slog.device)
                    tk_idx = topk_indices.to(slog.device)
                    tk_vals = topk_values.to(slog.device)

                    s_on_topk = slog.gather(-1, tk_idx)
                    p_t = (tk_vals / TEMPERATURE).softmax(-1)
                    log_q_s = (s_on_topk / TEMPERATURE).log_softmax(-1)
                    kl = F.kl_div(log_q_s, p_t, reduction="batchmean") * (TEMPERATURE ** 2)
                    ce = F.cross_entropy(slog, gen_ids)
                    loss = KL_WEIGHT * kl + (1 - KL_WEIGHT) * ce

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()

                    ce_run += ce.item(); kl_run += kl.item(); n_run += 1
                    losses.append({"epoch": epoch + 1, "step": step,
                                   "ce": ce.item(), "kl": kl.item(),
                                   "loss": loss.item(), "qa_id": r["qa_id"]})
                    step += 1
                    if n_run % 16 == 0:
                        bar.set_postfix(ce=f"{ce_run/n_run:.3f}",
                                        kl=f"{kl_run/n_run:.3f}")
                    del logits, slog, inputs, inputs_full
                    bar.update(1)

                # release this video's decoded tensors before the next one;
                # also flush the GPU allocator so per-video shape changes don't
                # accumulate fragmentation.
                del video_inputs
                torch.cuda.empty_cache()

            bar.close()
            print(f"[train] epoch {epoch+1}: avg CE={ce_run/max(1,n_run):.3f} "
                  f"avg KL={kl_run/max(1,n_run):.3f}  ({n_run} steps)")

    # 9. Save adapter + loss curve
    adapter_path = ROOT / args.adapter_dir
    peft_model.save_pretrained(adapter_path)
    print(f"[train] saved LoRA adapter -> {adapter_path}")
    (ROOT / "train_losses.jsonl").write_text(
        "\n".join(json.dumps(l) for l in losses) + "\n"
    )

    # 10. Eval AFTER — use the live peft_model (already has the trained adapter).
    # Switch back to use_cache=True for faster generation.
    peft_model.config.use_cache = True
    print(f"\n[eval AFTER] generating on {len(holdout_rows)} held-out samples ...")
    eval_after = run_generation_eval(
        peft_model, processor, holdout_rows, holdout_pts, "AFTER",
    )
    write_jsonl(eval_after, ROOT / args.eval_after)
    print(f"[eval AFTER] saved -> {args.eval_after}")

    # 11. Comparison report
    summary = write_comparison_md(eval_before, eval_after, ROOT / args.report)
    print(f"\n[report] -> {args.report}")
    print(f"[report] label-in-answer:  {summary['label_in_before']:.3f}"
          f" -> {summary['label_in_after']:.3f}")
    print(f"[report] teacher-text sim: {summary['teacher_sim_before']:.3f}"
          f" -> {summary['teacher_sim_after']:.3f}")
    print(f"[report] gold-label sim:   {summary['gold_sim_before']:.3f}"
          f" -> {summary['gold_sim_after']:.3f}")
    print(f"\npeak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")


if __name__ == "__main__":
    main()
