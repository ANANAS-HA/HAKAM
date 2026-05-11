"""
Recipe A logit extraction on the curated 2080-video subset.

Reads:
  data/subsets/subset_train.json
  data/subsets/subset_val.json

For each (video, question) pair:
  1. Build prompt with the analyst persona (matches _smoke_27b_int4_v2.py exactly).
  2. model.generate(...) with v2 sampling, output_logits=True.
  3. Stack out.logits → top-32 per generated position.
  4. Save .pt with gen_tokens, top-K logits, full metadata.

Outputs:
  logits_cache_subset/qa_<qa_id:07d>.pt
  logits_cache_subset.manifest.jsonl  (one line per sample, resumable)

Per-video grouping caches the decoded video tensor across the ~5–6 questions
per video, eliminating ~80% of redundant vision-side compute.

Resume: every successful sample is logged to the manifest.jsonl, so re-running
the script picks up where it left off.

Use:
  python -u extract_logits_subset.py --split both
  python -u extract_logits_subset.py --split train --limit 50    # debug
"""
import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

# Help with fragmentation when memory is tight.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info


# ---------- constants copied verbatim from _smoke_27b_int4_v2.py ----------

MODEL_ID = "cyankiwi/Qwen3.5-27B-AWQ-4bit"
ROOT = Path(__file__).parent
SUBSET_DIR = ROOT / "data" / "subsets"
FPS = 2.0
TOPK = 32
MAX_NEW_TOKENS = 256

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


# ---------- video path resolution (copied from existing extractors) ----------

def resolve_video_path(video_key):
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


# ---------- prompt construction matches _smoke_27b_int4_v2.py:build_inputs ----------

def build_messages(video_path, question):
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


# ---------- subset schema flattening ----------

def flatten_subset(rows, split_label):
    """Convert the subset's nested format into per-question flat dicts.

    Input row: {video_id, sport, split, questions: [{question_id, question_type, question_text, answer}]}
    Output:    {qa_id, video_id, split, sport, question_id, question_type, question, answer}
    """
    flat = []
    for r in rows:
        vid = r["video_id"]
        sport = r.get("sport", "?")
        for q in r.get("questions", []):
            flat.append({
                "qa_id": int(q["question_id"]),
                "video_id": vid,
                "split": split_label,
                "sport": sport,
                "question_id": int(q["question_id"]),
                "question_type": q.get("question_type", "?"),
                "question": q["question_text"],
                "answer": q["answer"],
            })
    return flat


# ---------- the actual extraction ----------

def extract_one(model, processor, video_path, question, cached_video_inputs):
    """Recipe A: generate with v2 settings, capture per-position raw top-K logits.

    Uses output_logits=True (NOT output_scores) so the captured distribution is
    the teacher's natural pre-warper distribution, which is what we want to
    distill into the student. output_scores returns post-top_k filtered logits
    where most positions are -inf, which is useless for the student's KL term.
    """
    msgs = build_messages(video_path, question)
    prompt_text = apply_chat_template(processor, msgs)

    if cached_video_inputs is None:
        image_inputs, video_inputs = process_vision_info(msgs)
    else:
        image_inputs, video_inputs = None, cached_video_inputs

    inputs = processor(text=[prompt_text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to("cuda")
    prompt_len = inputs["input_ids"].shape[1]

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
    dt = time.perf_counter() - t0

    # out.sequences shape (1, prompt_len + T)
    # out.logits is a tuple of T tensors of shape (1, V), pre-warper raw logits
    gen_tokens = out.sequences[0, prompt_len:].cpu()  # (T,)
    T = int(gen_tokens.shape[0])
    if T == 0:
        return None  # empty generation; treat as error so manifest records it

    # Stack logits → (T, V) and reduce to top-K
    logits = torch.stack([s[0] for s in out.logits], dim=0).float()  # (T, V)
    topk_vals, topk_idx = logits.topk(k=TOPK, dim=-1)  # (T, K) each

    # Sanity metrics
    probs = logits.softmax(-1)
    topk_mass = probs.gather(-1, topk_idx).sum(-1).mean().item()
    chosen_in_topk = (
        topk_idx == gen_tokens.unsqueeze(-1).to(topk_idx.device)
    ).any(-1).float().mean().item()

    # Decoded text for human inspection / downstream filtering
    gen_text = processor.batch_decode(
        [gen_tokens], skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()

    return {
        "gen_tokens": gen_tokens,                                      # (T,) int64
        "topk_values": topk_vals.cpu().to(torch.float16),              # (T, K) fp16
        "topk_indices": topk_idx.cpu().to(torch.int32),                # (T, K) int32
        "gen_text": gen_text,
        "topk_mass": topk_mass,
        "chosen_in_topk": chosen_in_topk,
        "seq_len": T,
        "wall_time_s": dt,
    }


# ---------- main loop ----------

def load_done_ids(manifest_path):
    """Read manifest jsonl, return set of qa_ids that have a successful entry."""
    done = set()
    if not manifest_path.exists():
        return done
    with manifest_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            # Only count successful entries (skip ones with an error field)
            if "qa_id" in d and "error" not in d:
                done.add(int(d["qa_id"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="both", choices=["train", "val", "both"])
    ap.add_argument("--out_dir", default="logits_cache_subset")
    ap.add_argument("--manifest", default="logits_cache_subset.manifest.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N rows total (debug only).")
    ap.add_argument("--model_id", default=MODEL_ID,
                    help=f"Override teacher model. Default {MODEL_ID}. "
                         "Use Qwen/Qwen3.5-4B for fast local pipeline validation.")
    args = ap.parse_args()
    model_id = args.model_id

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(exist_ok=True)
    manifest_path = ROOT / args.manifest

    done_ids = load_done_ids(manifest_path)
    print(f"[resume] {len(done_ids)} samples already in manifest")

    rows = []
    if args.split in ("train", "both"):
        train_rows = json.loads((SUBSET_DIR / "subset_train.json").read_text())
        rows.extend(flatten_subset(train_rows, "train"))
    if args.split in ("val", "both"):
        val_rows = json.loads((SUBSET_DIR / "subset_val.json").read_text())
        rows.extend(flatten_subset(val_rows, "val"))
    print(f"loaded {len(rows)} QA rows from subset (split={args.split})")

    if args.limit:
        rows = rows[: args.limit]

    # Filter pending and group by sport → video → questions for nested progress
    by_sport = defaultdict(lambda: defaultdict(list))
    skipped_done = 0
    skipped_missing = 0
    for r in rows:
        if r["qa_id"] in done_ids:
            skipped_done += 1
            continue
        vp = resolve_video_path(r["video_id"])
        if vp is None:
            skipped_missing += 1
            continue
        by_sport[r["sport"]][r["video_id"]].append((r, vp))

    sport_totals = {sp: sum(len(qs) for qs in vids.values()) for sp, vids in by_sport.items()}
    total_pending = sum(sport_totals.values())
    n_videos = sum(len(v) for v in by_sport.values())

    print(f"pending: {total_pending} questions across {n_videos} videos in "
          f"{len(by_sport)} sports (skipped: {skipped_done} done, "
          f"{skipped_missing} missing-video)")
    for sp, n in sport_totals.items():
        print(f"  {sp:<20s} {n:>5d} questions, {len(by_sport[sp]):>3d} videos")
    if total_pending == 0:
        print("nothing to do.")
        return

    print(f"Loading {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
    ).eval()
    print(f"loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB\n")

    answered = 0
    errors = 0
    sum_mass = 0.0
    sum_chosen_in = 0.0
    t_start = time.time()

    # Three nested progress bars:
    #  pos 0 — overall (persists for the whole run)
    #  pos 1 — current sport (resets per sport)
    #  pos 2 — current video's questions (resets per video, hidden when done)
    overall = tqdm(total=total_pending, desc="overall", unit="q",
                   position=0, dynamic_ncols=True)

    with manifest_path.open("a") as mout:
        for sport, by_video in by_sport.items():
            sport_total = sport_totals[sport]
            sport_bar = tqdm(total=sport_total, desc=f"  {sport:<14s}", unit="q",
                             position=1, leave=False, dynamic_ncols=True)

            for video_id, qs in by_video.items():
                video_short = video_id.rsplit('/', 1)[-1][:32]
                video_bar = tqdm(qs, desc=f"    {video_short:<32s}", unit="q",
                                 position=2, leave=False, dynamic_ncols=True)

                video_path = qs[0][1]
                # Decode the video once, reuse across its ~5-6 questions
                try:
                    _, cached_video_inputs = process_vision_info(
                        build_messages(video_path, qs[0][0]["question"])
                    )
                except Exception as e:
                    for r, _ in qs:
                        mout.write(json.dumps({
                            "qa_id": r["qa_id"], "video_id": video_id,
                            "error": f"decode: {type(e).__name__}: {e}",
                        }) + "\n")
                        errors += 1
                        overall.update(1)
                        sport_bar.update(1)
                        video_bar.update(1)
                    mout.flush()
                    video_bar.close()
                    continue

                for r, _vp in video_bar:
                    try:
                        rec = extract_one(model, processor, video_path,
                                          r["question"], cached_video_inputs)
                    except Exception as e:
                        mout.write(json.dumps({
                            "qa_id": r["qa_id"], "video_id": video_id,
                            "error": f"{type(e).__name__}: {e}",
                        }) + "\n")
                        mout.flush()
                        errors += 1
                        overall.update(1)
                        sport_bar.update(1)
                        continue

                    if rec is None:
                        mout.write(json.dumps({
                            "qa_id": r["qa_id"], "video_id": video_id,
                            "error": "empty_generation",
                        }) + "\n")
                        mout.flush()
                        errors += 1
                        overall.update(1)
                        sport_bar.update(1)
                        continue

                    save_path = out_dir / f"qa_{r['qa_id']:07d}.pt"
                    torch.save({
                        # metadata from the row
                        "qa_id":         r["qa_id"],
                        "video_id":      r["video_id"],
                        "split":         r["split"],
                        "sport":         r["sport"],
                        "question_id":   r["question_id"],
                        "question_type": r["question_type"],
                        "question":      r["question"],
                        "answer":        r["answer"],
                        # extraction outputs
                        "gen_tokens":    rec["gen_tokens"],
                        "topk_values":   rec["topk_values"],
                        "topk_indices":  rec["topk_indices"],
                        "gen_text":      rec["gen_text"],
                        "topk_mass":     rec["topk_mass"],
                        "chosen_in_topk": rec["chosen_in_topk"],
                        "seq_len":       rec["seq_len"],
                        "wall_time_s":   rec["wall_time_s"],
                    }, save_path)

                    mout.write(json.dumps({
                        "qa_id":           r["qa_id"],
                        "video_id":        video_id,
                        "split":           r["split"],
                        "sport":           r["sport"],
                        "question_type":   r["question_type"],
                        "seq_len":         rec["seq_len"],
                        "topk_mass":       rec["topk_mass"],
                        "chosen_in_topk":  rec["chosen_in_topk"],
                        "wall_time_s":     rec["wall_time_s"],
                        "path":            save_path.name,
                    }) + "\n")
                    mout.flush()

                    answered += 1
                    sum_mass += rec["topk_mass"]
                    sum_chosen_in += rec["chosen_in_topk"]
                    overall.update(1)
                    sport_bar.update(1)

                video_bar.close()

                # Live overall stats — refreshed once per video (not per question)
                elapsed = time.time() - t_start
                rate = answered / elapsed if elapsed else 0
                remaining = total_pending - answered - errors
                eta_min = (remaining / rate / 60) if rate > 0 else 0
                overall.set_postfix({
                    "ok":     answered,
                    "err":    errors,
                    "mass":   f"{sum_mass / max(1, answered):.3f}",
                    "chosen": f"{sum_chosen_in / max(1, answered):.3f}",
                    "rate":   f"{rate:.2f}/s",
                    "eta_h":  f"{eta_min/60:.1f}",
                })

            sport_bar.close()

    overall.close()
    print(f"\nDone. answered={answered} "
          f"avg_topk_mass={sum_mass / max(1, answered):.4f} "
          f"avg_chosen_in_topk={sum_chosen_in / max(1, answered):.4f}")
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
