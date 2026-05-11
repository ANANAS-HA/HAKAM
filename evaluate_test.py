import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel

# peft<->gptqmodel version skew workaround: peft's dispatch_awq for every
# target module imports a class that gptqmodel renamed; the 0.8B student is
# not AWQ-quantized, so neutralize the dispatcher.
import peft.tuners.lora.model as _peft_lora_model
def _noop_dispatch_awq(target, adapter_name, config, **kwargs): return None
_peft_lora_model.dispatch_awq = _noop_dispatch_awq

MODEL_ID = "Qwen/Qwen3.5-0.8B"
ROOT = Path(__file__).parent
VIDEOS = ROOT / "videos"
FINEGYM_DIRS = [VIDEOS / f"finegym_0{i}" for i in range(1, 6)]
EXTS = (".avi", ".mp4", ".mkv", ".webm")

SYSTEM_PROMPT = (
    "You answer questions about short sports videos. This is a classification task: "
    "every answer is a single label from a fixed vocabulary. "
    "Output ONLY the answer label, nothing else — no sentences, no explanations, no punctuation. "
    "Answers are typically 1 to 3 words. Examples of valid answers: "
    "'yes', 'no', '1', '2', '3', 'basketball', 'football', 'volleyball', 'aerobic gymnastics', "
    "'volleyball spike', 'basketball 2-point shot', 'basketball missed', 'left team', 'right team'."
)


def resolve_video_path(video_key: str) -> Path | None:
    sport, stem = video_key.split("/", 1)
    candidates = []
    if sport == "fg":
        for d in FINEGYM_DIRS:
            candidates.extend(d / f"{stem}{e}" for e in EXTS)
    elif sport == "volleyball":
        candidates.extend(VIDEOS / "volleyball" / "volleyball" / f"{stem}{e}" for e in EXTS)
        candidates.extend(VIDEOS / "volleyball" / f"{stem}{e}" for e in EXTS)
    else:
        candidates.extend(VIDEOS / sport / f"{stem}{e}" for e in EXTS)
    for p in candidates:
        if p.exists():
            return p
    return None


def build_inputs(processor, video_path: Path, question: str, fps: float, device):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video", "video": str(video_path), "fps": fps},
            {"type": "text", "text": question},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test.json")
    ap.add_argument("--out", default="predictions_test_0p8b_lora.jsonl")
    ap.add_argument("--adapter", default="student_lora",
                    help="LoRA adapter dir under project root. Empty string disables.")
    ap.add_argument("--limit", type=int, default=None, help="Only first N rows")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    args = ap.parse_args()

    out_path = ROOT / args.out
    done_ids = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try: done_ids.add(json.loads(line)["qa_id"])
                except Exception: pass
        print(f"[resume] {len(done_ids)} predictions already in {out_path.name}")

    rows = json.loads((ROOT / "meta-data" / args.split).read_text())
    if args.limit:
        rows = rows[: args.limit]

    by_video: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["qa_id"] in done_ids:
            continue
        by_video[r["video"]].append(r)

    total_pending = sum(len(v) for v in by_video.values())
    print(f"Loading {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    if args.adapter:
        adapter_path = ROOT / args.adapter
        print(f"Applying LoRA adapter from {adapter_path} ...")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    missing = correct = answered = 0
    t_start = time.time()
    with out_path.open("a") as fout:
        for vi, (video_key, qs) in enumerate(by_video.items()):
            video_path = resolve_video_path(video_key)
            if video_path is None:
                for r in qs:
                    fout.write(json.dumps({"qa_id": r["qa_id"], "video": video_key,
                                           "question": r["question"], "gold": r["answer"],
                                           "pred": None, "error": "video_missing"}) + "\n")
                missing += len(qs)
                continue

            for r in qs:
                try:
                    inputs = build_inputs(processor, video_path, r["question"], args.fps, model.device)
                    with torch.inference_mode():
                        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                    new_tokens = generated[:, inputs["input_ids"].shape[1]:]
                    pred = processor.batch_decode(new_tokens, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)[0].strip()
                except Exception as e:
                    pred, err = None, f"{type(e).__name__}: {e}"
                    fout.write(json.dumps({"qa_id": r["qa_id"], "video": video_key,
                                           "question": r["question"], "gold": r["answer"],
                                           "pred": None, "error": err}) + "\n")
                    answered += 1
                    continue

                gold = r["answer"].strip().lower()
                match = pred.strip().lower() == gold
                correct += int(match)
                answered += 1
                fout.write(json.dumps({"qa_id": r["qa_id"], "video": video_key,
                                       "question": r["question"], "gold": r["answer"],
                                       "pred": pred, "match": match}) + "\n")

            fout.flush()
            elapsed = time.time() - t_start
            rate = answered / elapsed if elapsed else 0
            remaining = total_pending - answered - missing
            eta = remaining / rate if rate else 0
            print(f"[{vi+1}/{len(by_video)}] answered={answered} missing={missing} "
                  f"acc={correct/max(1,answered-missing):.3f} rate={rate:.2f}/s eta={eta/60:.1f}min")

    print(f"\nDone. answered={answered} correct={correct} missing={missing} "
          f"exact_match_acc={correct/max(1,answered-missing):.3f}")
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")


if __name__ == "__main__":
    main()
