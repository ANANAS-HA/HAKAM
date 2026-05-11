"""Smoke test on Qwen3.5-27B-AWQ-4bit (compressed-tensors) — same prompt, same
sampling, same clip as _smoke_9b.py v3.

VRAM note: 27B AWQ Int4 loads at ~18.4 GB, but compressed-tensors decompresses
weights at first inference, which pushes total > 24 GB on a 3090. We use
device_map="auto" + a 22 GB GPU cap so accelerate spills excess layers to CPU.
Expect ~5–15× slower per-token generation than a fully-resident model.
"""
import os
# Help with fragmentation when memory is tight.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc, json, re, time
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, TextStreamer
from transformers.generation import StoppingCriteria, StoppingCriteriaList
from qwen_vl_utils import process_vision_info


class StopAfterThink(StoppingCriteria):
    """Stop generation `after_think` tokens past the first </think>, but not before
    `min_new` total new tokens have been produced. Self-resets per generate() call."""
    def __init__(self, tokenizer, after_think=150, min_new=300):
        self.after_think = after_think
        self.min_new = min_new
        self.close_ids = tokenizer.encode("</think>", add_special_tokens=False)
        self._prompt_len = None
        self._think_close_at = None

    def _reset(self, seq_len):
        # First call of a fresh generate(): one new token already appended.
        self._prompt_len = seq_len - 1
        self._think_close_at = None

    def __call__(self, input_ids, scores, **kw):
        seq_len = input_ids.shape[1]
        if self._prompt_len is None or seq_len <= self._prompt_len:
            self._reset(seq_len)
        new_len = seq_len - self._prompt_len
        if new_len < self.min_new:
            return False
        if self._think_close_at is None:
            tail = input_ids[0, self._prompt_len:].tolist()
            n = len(self.close_ids)
            for i in range(len(tail) - n + 1):
                if tail[i:i + n] == self.close_ids:
                    self._think_close_at = i + n
                    break
        if self._think_close_at is not None:
            if (new_len - self._think_close_at) >= self.after_think:
                return True
        return False


class TimedStreamer(TextStreamer):
    """TextStreamer that also records wall time and new-token count for tok/s."""
    def __init__(self, processor, **kw):
        super().__init__(processor, skip_prompt=True, skip_special_tokens=True, **kw)
        self.t0 = None
        self.n_tokens = 0

    def put(self, value):
        if not self.next_tokens_are_prompt:
            if self.t0 is None:
                self.t0 = time.perf_counter()
            self.n_tokens += value.shape[-1] if value.ndim > 0 else 1
        return super().put(value)

    def tok_per_sec(self):
        if self.t0 is None or self.n_tokens == 0:
            return 0.0
        return self.n_tokens / max(time.perf_counter() - self.t0, 1e-6)

MODEL_ID = "cyankiwi/Qwen3.5-27B-AWQ-4bit"  # was Qwen/Qwen3.5-27B-GPTQ-Int4 — Marlin/Triton kernels reject 48-dim layer
ROOT = Path(__file__).parent
FPS = 2.0
TOPK = 32

# Identical to _smoke_9b.py v3 prompt — only model changes.
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


def build_inputs(processor, video_path, question, gold=None):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video", "video": str(video_path), "fps": FPS,
             "max_frames": 32, "max_pixels": 256 * 28 * 28},
            {"type": "text", "text": question},
        ]},
    ]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    prompt_only = processor(text=[prompt_text], images=image_inputs, videos=video_inputs,
                            padding=True, return_tensors="pt")
    prompt_len = prompt_only["input_ids"].shape[1]
    full_text = prompt_text + gold if gold else prompt_text
    inputs = processor(text=[full_text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to("cuda")
    return inputs, prompt_len


def main():
    train = json.loads((ROOT / "meta-data" / "train.json").read_text())
    first_key = None
    for r in train:
        if r["video"].startswith("basketball/") and resolve_video_path(r["video"]):
            first_key = r["video"]; break
    rows = [r for r in train if r["video"] == first_key]
    video_path = resolve_video_path(first_key)
    print(f"video: {first_key}   ({len(rows)} questions)\n")

    print(f"Loading {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    # Compressed-tensors stores int4 weights; activations dtype must be a float type, set explicitly.
    # device_map="auto" + max_memory cap: accelerate spills layers to CPU when GPU is near full,
    # which is necessary because compressed-tensors' decompress-on-first-inference inflates VRAM.
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    ).eval()
    print(f"loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB\n")

    print("=" * 78)
    print("PART 1 — Answering questions (analyst prompt, thinking on, v3 sampling, 27B-Int4)")
    print("=" * 78)
    correct = 0
    n = len(rows)
    stopping = StoppingCriteriaList([
        StopAfterThink(processor.tokenizer, after_think=150, min_new=300)
    ])
    bar = tqdm(rows, desc=first_key.rsplit('/', 1)[-1], unit='q')
    for i, r in enumerate(bar, 1):
        print(f"\n[{i:>2d}/{n}] [{r['type']:<14s}] Q: {r['question']}")
        print(f"       gold: {r['answer']!r}")
        inputs, _ = build_inputs(processor, video_path, r["question"])
        streamer = TimedStreamer(processor)
        t0 = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=256,
                stopping_criteria=stopping,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                top_k=20,
                min_p=0.0,
                repetition_penalty=1.1,
                
                streamer=streamer,
            )
        dt = time.perf_counter() - t0
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(new_tokens, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0].strip()
        think_match = re.search(r"<think>(.*?)</think>\s*(.*)", raw, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            answer = think_match.group(2).strip() or "(no answer after </think>)"
        else:
            thinking = ""
            answer = raw
        match = r["answer"].strip().lower() in answer.lower()
        correct += int(match)
        mark = "✓" if match else "✗"
        tps = streamer.n_tokens / dt if dt > 0 else 0.0
        print(f"\n[{i:>2d}/{n}] {mark} [{r['type']:<14s}] {dt:.1f}s | {streamer.n_tokens} new tok | {tps:.1f} tok/s   gold={r['answer']!r}")
    bar.close()
    print(f"\n'gold label appears in answer' on this video: {correct}/{n} = {correct/n:.1%}")
    print(f"\npeak VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")


if __name__ == "__main__":
    main()
