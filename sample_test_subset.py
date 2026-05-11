"""Build a balanced Sports-QA *test* subset for downstream evaluation.

Mirrors the train/val sampler in sample_subset.py: stratify by logical
sport (8 categories — football, basketball, volleyball, Gym, and the 4
FineGym sub-events), pick N videos per sport, then apply the same
per-video question caps (3 Descriptive / 3 Temporal / 2 Causal / 1 CF)
with long-tail-answer preference inside each cap.

Default: 60 videos per sport × 8 sports = 480 videos.

Question counts are not balanced across sports — whatever the per-video
caps produce on the chosen videos is what we keep.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from sample_subset import (
    build_fg_sub_sport_map,
    build_video_entries,
    group_by_video,
    load_split,
    select_videos_per_sport,
)


def compute_stats(entries: list[dict]) -> dict:
    per_sport_videos = Counter(e["sport"] for e in entries)
    per_sport_questions: Counter = Counter()
    per_type: Counter = Counter()
    yes_no: Counter = Counter()
    total_q = 0
    for e in entries:
        for q in e["questions"]:
            total_q += 1
            per_sport_questions[e["sport"]] += 1
            per_type[q["question_type"]] += 1
            a = q["answer"].strip().lower()
            if a in ("yes", "no"):
                yes_no[a] += 1
    yn_total = sum(yes_no.values())
    return {
        "test": {
            "videos": len(entries),
            "questions": total_q,
            "per_sport_videos": dict(per_sport_videos),
            "per_sport_questions": dict(per_sport_questions),
            "per_question_type": dict(per_type),
            "yes_no": {
                "yes": yes_no.get("yes", 0),
                "no": yes_no.get("no", 0),
                "ratio": round(yes_no.get("yes", 0) / yn_total, 4) if yn_total else None,
            },
        }
    }


def print_summary(stats: dict, target: int) -> None:
    s = stats["test"]
    print("\n=== Test subset summary ===")
    print(f"Videos:    {s['videos']}  (target {target}/sport × 8 = {target * 8})")
    print(f"Questions: {s['questions']}")
    print("\nPer sport:")
    print(f"  {'sport':<22} {'videos':>8} {'questions':>10}")
    for sport in sorted(s["per_sport_videos"]):
        v = s["per_sport_videos"][sport]
        q = s["per_sport_questions"].get(sport, 0)
        print(f"  {sport:<22} {v:>8} {q:>10}")

    print("\nPer question type:")
    for qt in ("Descriptive", "Temporal", "Causal", "Counterfactual"):
        print(f"  {qt:<18} {s['per_question_type'].get(qt, 0):>5}")

    yn = s["yes_no"]
    print(f"\nYes/no:  yes={yn['yes']} no={yn['no']} ratio={yn['ratio']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--annotations-dir", type=Path, default=Path("meta-data"))
    p.add_argument("--out-dir", type=Path, default=Path("data/subsets"))
    p.add_argument("--per-sport", type=int, default=60,
                   help="Videos to keep per logical sport (default 60).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"Loading test annotations from {args.annotations_dir}/ ...")
    test_rows = load_split(args.annotations_dir, "test")
    print(f"  test: {len(test_rows)} QAs")

    by_video = group_by_video(test_rows)
    fg_map = build_fg_sub_sport_map(test_rows)
    n_fg_videos = sum(1 for v in by_video if v.startswith("fg/"))
    print(f"  fg sub-sport recovered: {len(fg_map)}/{n_fg_videos}")

    print(f"\nSelecting {args.per_sport} videos per sport ...")
    chosen = select_videos_per_sport(by_video, args.per_sport, rng, fg_map)

    entries = build_video_entries(chosen, by_video, "test", rng)
    stats = compute_stats(entries)

    out_path = args.out_dir / "subset_test.json"
    stats_path = args.out_dir / "subset_test_stats.json"
    with out_path.open("w") as f:
        json.dump(entries, f, indent=2)
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    by_split_test = sorted({e["video_id"] for e in entries})
    manifest = {
        "seed": args.seed,
        "source": f"meta-data/test.json (curated, {args.per_sport} videos/sport)",
        "videos": [
            {"video_id": e["video_id"], "split": "test", "sport": e["sport"]}
            for e in entries
        ],
        "by_split": {"test": by_split_test},
    }
    manifest_path = args.out_dir / "subset_test_video_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote:\n  {out_path}\n  {stats_path}\n  {manifest_path}")
    print_summary(stats, args.per_sport)


if __name__ == "__main__":
    main()
