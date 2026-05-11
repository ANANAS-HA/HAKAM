"""Shared correctness metric for the GP2 figures.

Mirrors `eval_compare.py:36-54` exactly so derived figure numbers match the
report. Also encodes the field-name mapping (`student_text` not `pred`,
`question_type` not `qtype`, sport-from-prefix fallback for older prediction
files that lack the `sport` field).
"""
from __future__ import annotations

import json
from pathlib import Path

import Levenshtein


THRESHOLD = 0.80


def best_window_similarity(text: str, gold: str) -> float:
    """Slide a window of len(gold) across text, return max Levenshtein.ratio."""
    text = (text or "").lower().strip()
    gold = (gold or "").lower().strip()
    if not text or not gold:
        return 0.0
    g = len(gold)
    if len(text) <= g:
        return Levenshtein.ratio(text, gold)
    best = 0.0
    for i in range(len(text) - g + 1):
        s = Levenshtein.ratio(text[i:i + g], gold)
        if s > best:
            best = s
        if best >= 0.999:
            break
    return best


def is_correct(text: str, gold: str, threshold: float = THRESHOLD) -> bool:
    return best_window_similarity(text, gold) >= threshold


def get_pred_text(row: dict) -> str:
    """The model's free-form answer is in `student_text`, not `pred`."""
    return row["student_text"]


def get_qtype(row: dict) -> str:
    """The question-type field is `question_type`, not `qtype`."""
    return row["question_type"]


def get_sport(row: dict) -> str:
    """Curated sport when present (3-epoch + teacher prediction files);
    fall back to the video_id prefix for older files (vanilla, 1-epoch).
    Mirrors the fallback at eval_compare.py:134.
    """
    return row.get("sport") or row["video_id"].split("/", 1)[0]


def load_jsonl(path: Path | str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def accuracy(rows: list[dict], threshold: float = THRESHOLD) -> tuple[int, int]:
    """Return (n_correct, n_total) using is_correct on each row."""
    n_total = len(rows)
    n_correct = sum(1 for r in rows if is_correct(get_pred_text(r), r["gold"], threshold))
    return n_correct, n_total
