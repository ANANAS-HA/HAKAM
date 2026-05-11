"""Generate predictions on `meta-data/test.json` with the analyst SYSTEM_PROMPT
the 27B teacher used during extraction. Designed to be run twice — once for
the vanilla 0.8B and once with `--adapter student_lora` for the LoRA-trained
student — and the two output jsonls fed to `eval_compare.py` for the
flip-rate / similarity comparison.

Output schema mirrors `eval_before.jsonl` / `eval_after.jsonl` exactly so
`eval_compare.py` can consume it with no changes:
    {qa_id, video_id, question, question_type, gold, teacher_gen_text,
     student_text}
`teacher_gen_text` is set to "" (we never ran the 27B on the test split).

Usage:
    # vanilla 0.8B
    python eval_test_predict.py --out predictions_test_vanilla.jsonl

    # trained student
    python eval_test_predict.py --adapter student_lora \
        --out predictions_test_trained.jsonl

    # smoke
    python eval_test_predict.py --limit 20 --out smoke.jsonl

    # stratified subsample (recommended first run)
    python eval_test_predict.py --stratify_per_sport 200 \
        --out predictions_test_vanilla_1k.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# This script is forward-only (generate inside torch.inference_mode), so the
# WSL2 backward race we hit during training does not apply. Don't set
# CUDA_LAUNCH_BLOCKING here — it costs ~3x throughput. If you do see any
# transient CUDA error you can override at the shell:  CUDA_LAUNCH_BLOCKING=1 ...

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel

# peft<->gptqmodel version skew workaround — see evaluate_test.py / train_student.py
import peft.tuners.lora.model as _peft_lora_model
def _noop_dispatch_awq(target, adapter_name, config, **kwargs): return None
_peft_lora_model.dispatch_awq = _noop_dispatch_awq


ROOT = Path(__file__).parent
STUDENT_ID = "Qwen/Qwen3.5-0.8B"
META_DIR = ROOT / "meta-data"

FPS = 2.0
MAX_NEW_TOKENS = 256

# Verbatim from train_student.py / extract_logits_subset.py — the same prompt
# the 27B teacher saw, so the student's distillation has a chance to transfer.
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


def load_subset_file(path: Path) -> list[dict]:
    """Flatten a curated subset JSON (e.g. data/subsets/subset_test.json) into
    the per-question row format the rest of this script expects.

    Subset schema (one element per video):
        {video_id, sport, split, questions: [
            {question_id, question_type, question_text, answer}, ...
        ]}

    Output rows match the flat upstream Sports-QA jsonl shape used elsewhere:
        {qa_id, video, type, question, answer, sport}
    `sport` here comes from the subset's curated taxonomy (incl. recovered
    FineGym sub-events), not the raw video-id prefix — needed for accurate
    per-sport eval breakdowns when running through subset_test.json.
    """
    data = json.loads(path.read_text())
    out = []
    for v in data:
        for q in v.get("questions", []):
            out.append({
                "qa_id":    int(q["question_id"]),
                "video":    v["video_id"],
                "type":     q.get("question_type", "?"),
                "question": q["question_text"],
                "answer":   q["answer"],
                "sport":    v.get("sport", "?"),
            })
    return out


def stratify_per_sport(rows: list[dict], n_per_sport: int, seed: int) -> list[dict]:
    """Take n questions per sport, sub-stratified by question type within
    each sport. Sports with fewer than n total are taken in full."""
    rng = random.Random(seed)
    by_sport: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        head = r["video"].split("/", 1)[0]
        by_sport[head].append(r)
    out = []
    for sport, items in sorted(by_sport.items()):
        if len(items) <= n_per_sport:
            out.extend(items)
            continue
        by_qt: dict[str, list[dict]] = defaultdict(list)
        for r in items:
            by_qt[r["type"]].append(r)
        per_qt = max(1, n_per_sport // len(by_qt))
        picked = []
        for qt, bucket in by_qt.items():
            bb = bucket[:]
            rng.shuffle(bb)
            picked.extend(bb[:per_qt])
        if len(picked) < n_per_sport:
            extra = [r for r in items if r["qa_id"] not in {p["qa_id"] for p in picked}]
            rng.shuffle(extra)
            picked.extend(extra[:n_per_sport - len(picked)])
        rng.shuffle(picked)
        out.extend(picked[:n_per_sport])
    return out


def load_done_qa_ids(path: Path) -> set[int]:
    """Return qa_ids already in the output file so we can resume mid-run."""
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            try:
                done.add(int(json.loads(line)["qa_id"]))
            except Exception:
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test.json",
                    help="Upstream Sports-QA file under meta-data/. Default test.json. "
                         "Ignored if --subset_file is given.")
    ap.add_argument("--subset_file", default=None,
                    help="Curated subset (e.g. data/subsets/subset_test.json) — "
                         "video-stratified format produced by sample_test_subset.py. "
                         "Overrides --split when set.")
    ap.add_argument("--out", required=True,
                    help="Append-only jsonl. Resumes if file exists.")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter dir under project root (e.g. 'student_lora'). "
                         "Omit to evaluate vanilla 0.8B.")
    ap.add_argument("--model_id", default=STUDENT_ID,
                    help="HF model id to load. Default = the 0.8B student "
                         "('Qwen/Qwen3.5-0.8B'). Use 'cyankiwi/Qwen3.5-27B-AWQ-4bit' "
                         "to evaluate the teacher (pod only — won't fit on a 24GB card).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Truncate to first N rows after stratification (debug).")
    ap.add_argument("--stratify_per_sport", type=int, default=None,
                    help="Sub-stratified sample of N per sport. Omit for full set.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    out_path = ROOT / args.out
    done_ids = load_done_qa_ids(out_path)
    print(f"[resume] {len(done_ids)} predictions already in {out_path.name}")

    if args.subset_file:
        subset_path = ROOT / args.subset_file if not Path(args.subset_file).is_absolute() else Path(args.subset_file)
        rows = load_subset_file(subset_path)
        print(f"[data] {subset_path.name}: {len(rows)} questions "
              f"(curated subset, sport metadata from FineGym recovery)")
    else:
        rows = json.loads((META_DIR / args.split).read_text())
        print(f"[data] {args.split}: {len(rows)} questions")

    if args.stratify_per_sport:
        rows = stratify_per_sport(rows, args.stratify_per_sport, args.seed)
        print(f"[data] stratified to {len(rows)} ({args.stratify_per_sport}/sport)")
    if args.limit:
        rows = rows[:args.limit]
        print(f"[data] limited to first {len(rows)}")

    # filter resolvable videos + skip already-done
    by_video: dict[str, list[dict]] = defaultdict(list)
    skipped_done = 0
    skipped_missing = 0
    video_path_of: dict[str, Path] = {}
    for r in rows:
        if r["qa_id"] in done_ids:
            skipped_done += 1
            continue
        vp = resolve_video_path(r["video"])
        if vp is None:
            skipped_missing += 1
            continue
        by_video[r["video"]].append(r)
        video_path_of[r["video"]] = vp
    total_pending = sum(len(v) for v in by_video.values())
    print(f"[data] pending: {total_pending} questions across {len(by_video)} unique videos "
          f"(skipped: {skipped_done} done, {skipped_missing} video-missing)")
    if total_pending == 0:
        print("Nothing to do.")
        return

    # ---- model ----
    print(f"[model] loading {args.model_id} ...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    # eager attention is a WSL2-driver-race dodge that costs throughput on
    # native Linux. Skip it when loading the heavy AWQ teacher (which only
    # ever runs on the pod) — let HF pick SDPA for ~2x faster forward.
    _model_kwargs = dict(
        dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
    )
    if "AWQ" not in args.model_id and "awq" not in args.model_id:
        _model_kwargs["attn_implementation"] = "eager"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, **_model_kwargs
    )
    if args.adapter:
        adapter_path = ROOT / args.adapter
        print(f"[model] applying LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    print(f"[model] loaded. VRAM={torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # ---- generate ----
    answered = errors = 0
    sum_seq_len = 0
    t_start = time.time()
    bar = tqdm(total=total_pending, desc="generating", unit="q")
    with out_path.open("a") as fout:
        for video_key, qs in by_video.items():
            video_path = video_path_of[video_key]
            try:
                _, video_inputs = process_vision_info(
                    build_messages(video_path, qs[0]["question"])
                )
            except Exception as e:
                err = f"decode: {type(e).__name__}: {e}"
                for r in qs:
                    fout.write(json.dumps({
                        "qa_id":            r["qa_id"],
                        "video_id":         video_key,
                        "sport":            r.get("sport", video_key.split("/", 1)[0]),
                        "question":         r["question"],
                        "question_type":    r["type"],
                        "gold":             r["answer"],
                        "teacher_gen_text": "",
                        "student_text":    f"<gen_error: {err}>",
                    }) + "\n")
                    errors += 1
                    bar.update(1)
                fout.flush()
                continue

            for r in qs:
                msgs = build_messages(video_path, r["question"])
                text = processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
                try:
                    inputs = processor(
                        text=[text], images=None, videos=video_inputs,
                        padding=True, return_tensors="pt",
                    ).to("cuda")
                    prompt_len = inputs["input_ids"].shape[1]
                    with torch.inference_mode():
                        gen = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=True,
                            temperature=0.5,
                            top_p=0.9,
                            top_k=20,
                            min_p=0.0,
                            repetition_penalty=1.1,
                        )
                    new_tokens = gen[0, prompt_len:].cpu()
                    student_text = processor.batch_decode(
                        [new_tokens], skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0].strip()
                    sum_seq_len += int(new_tokens.shape[0])
                except Exception as e:
                    student_text = f"<gen_error: {type(e).__name__}: {e}>"
                    errors += 1

                fout.write(json.dumps({
                    "qa_id":            r["qa_id"],
                    "video_id":         video_key,
                    "sport":            r.get("sport", video_key.split("/", 1)[0]),
                    "question":         r["question"],
                    "question_type":    r["type"],
                    "gold":             r["answer"],
                    "teacher_gen_text": "",
                    "student_text":     student_text,
                }) + "\n")
                answered += 1
                bar.update(1)
                # free per-step buffers
                del inputs

            fout.flush()
            del video_inputs
            torch.cuda.empty_cache()

            elapsed = time.time() - t_start
            rate = answered / elapsed if elapsed else 0
            avg_len = sum_seq_len / max(1, answered - errors)
            bar.set_postfix(rate=f"{rate:.2f}q/s",
                            avg_tok=f"{avg_len:.0f}",
                            errs=errors)
    bar.close()

    print(f"\nDone. answered={answered} errors={errors} "
          f"avg_gen_tokens={sum_seq_len/max(1, answered-errors):.0f}")
    print(f"peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"\nNext step (once both vanilla and trained jsonls exist):")
    print(f"  python eval_compare.py "
          f"--before predictions_test_vanilla.jsonl "
          f"--after predictions_test_trained.jsonl "
          f"--report eval_test_flip_report.md")


if __name__ == "__main__":
    main()
