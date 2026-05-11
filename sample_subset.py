"""Build a balanced Sports-QA subset (train + val) for downstream inference.

Stratifies by the sport prefix in the `video` field, prioritises rare question
types, caps questions per video, and softly balances yes/no answers.

See plan: ~/.claude/plans/subset-handling-read-the-compiled-planet.md
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

FG_SUB_SPORT_NORMALIZE = {
    # answers seen in train.json, mapped to canonical sport keys
    "Balance Beam": "Balance_Beam",
    "balance beam": "Balance_Beam",
    "Uneven Bar": "Uneven_Bar",
    "uneven bar": "Uneven_Bar",
    "uneven bars": "Uneven_Bar",
    "Floor Exercise": "Floor_Exercise",
    "floor exercise": "Floor_Exercise",
    "Vault": "Vault",
    "vault": "Vault",
}


QUESTION_TYPE_CAPS = {
    "Descriptive": 3,
    "Temporal": 3,
    "Causal": 2,
    "Counterfactual": 1,
}

RARE_TYPE_WEIGHTS = {
    "Counterfactual": 4,
    "Causal": 2,
    "Temporal": 1,
    "Descriptive": 1,
}

YES_NO = {"yes", "no"}
YES_NO_TOLERANCE = 0.05  # final ratio must be within 0.5 ± tolerance


def load_split(annotations_dir: Path, split: str) -> list[dict]:
    path = annotations_dir / f"{split}.json"
    with path.open() as f:
        return json.load(f)


def group_by_video(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["video"]].append(r)
    return groups


def video_score(rows: list[dict]) -> int:
    return sum(RARE_TYPE_WEIGHTS.get(r["type"], 0) for r in rows)


def raw_prefix(video_id: str) -> str:
    return video_id.split("/", 1)[0]


def build_fg_sub_sport_map(rows: list[dict]) -> dict[str, str]:
    """For every fg/* video, recover its sub-sport from the descriptive
    'What is the video about?' answer. Returns {video_id: sub_sport}."""
    out: dict[str, str] = {}
    for r in rows:
        if not r["video"].startswith("fg/"):
            continue
        if r["type"] != "Descriptive":
            continue
        if not r["question"].lower().startswith("what is the video about"):
            continue
        sub = FG_SUB_SPORT_NORMALIZE.get(r["answer"])
        if sub:
            out.setdefault(r["video"], sub)
    return out


def sport_of(video_id: str, fg_map: dict[str, str]) -> str:
    """Logical sport for stratification. fg/* videos resolve to their
    FineGym sub-event; aerobic_gymnastics is renamed to Gym for clarity."""
    pref = raw_prefix(video_id)
    if pref == "fg":
        return fg_map.get(video_id, "fg_UNKNOWN")
    if pref == "aerobic_gymnastics":
        return "Gym"
    return pref


def is_long_tail(answer: str) -> bool:
    """Prefer answers that aren't yes/no/digit fillers when capping."""
    a = answer.strip().lower()
    if a in YES_NO:
        return False
    if a.isdigit():
        return False
    return True


def select_videos_per_sport(
    by_video: dict[str, list[dict]],
    target_per_sport: int,
    rng: random.Random,
    fg_map: dict[str, str],
) -> dict[str, list[str]]:
    """Top-k videos per sport by rare-type score, deterministic tie-break."""
    by_sport: dict[str, list[str]] = defaultdict(list)
    for vid in by_video:
        sport = sport_of(vid, fg_map)
        if sport == "fg_UNKNOWN":
            # fg video without a recoverable sub-sport — skip with a warning
            continue
        by_sport[sport].append(vid)

    chosen: dict[str, list[str]] = {}
    for sport, vids in by_sport.items():
        scored = [(video_score(by_video[v]), rng.random(), v) for v in vids]
        # higher score first; rng.random() breaks ties deterministically
        scored.sort(key=lambda x: (-x[0], x[1]))
        if len(vids) < target_per_sport:
            print(
                f"  WARNING: sport '{sport}' has only {len(vids)} videos "
                f"(< target {target_per_sport}); taking all."
            )
            chosen[sport] = [v for _, _, v in scored]
        else:
            chosen[sport] = [v for _, _, v in scored[:target_per_sport]]
    return chosen


def cap_questions(rows: list[dict], rng: random.Random) -> list[dict]:
    """Apply per-type caps, preferring long-tail answers within each type."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    kept: list[dict] = []
    for qtype, cap in QUESTION_TYPE_CAPS.items():
        bucket = by_type.get(qtype, [])
        if not bucket:
            continue
        # sort: long-tail answers first, then deterministic shuffle within
        decorated = [
            (0 if is_long_tail(r["answer"]) else 1, rng.random(), r) for r in bucket
        ]
        decorated.sort(key=lambda x: (x[0], x[1]))
        kept.extend(r for _, _, r in decorated[:cap])
    return kept


def build_video_entries(
    chosen: dict[str, list[str]],
    by_video: dict[str, list[dict]],
    split_name: str,
    rng: random.Random,
) -> list[dict]:
    entries: list[dict] = []
    for sport, vids in chosen.items():
        for v in vids:
            kept = cap_questions(by_video[v], rng)
            entries.append(
                {
                    "video_id": v,
                    "sport": sport,
                    "split": split_name,
                    "questions": [
                        {
                            "question_id": r["qa_id"],
                            "question_type": r["type"],
                            "question_text": r["question"],
                            "answer": r["answer"],
                        }
                        for r in kept
                    ],
                }
            )
    return entries


def balance_yes_no(entries: list[dict], rng: random.Random) -> dict:
    """Soft 50/50 balancing across the whole subset (drop majority overflow)."""
    yes_refs, no_refs = [], []
    for entry in entries:
        for q in entry["questions"]:
            a = q["answer"].strip().lower()
            if a == "yes":
                yes_refs.append((entry, q))
            elif a == "no":
                no_refs.append((entry, q))

    total = len(yes_refs) + len(no_refs)
    if total == 0:
        return {"yes": 0, "no": 0, "ratio": None, "dropped": 0}

    ratio = len(yes_refs) / total
    if abs(ratio - 0.5) <= YES_NO_TOLERANCE:
        return {
            "yes": len(yes_refs),
            "no": len(no_refs),
            "ratio": round(ratio, 4),
            "dropped": 0,
        }

    if len(yes_refs) > len(no_refs):
        majority, minority = yes_refs, no_refs
    else:
        majority, minority = no_refs, yes_refs

    # target: bring ratio to exactly 0.5 (within 1 row); deterministic drop order
    target_majority = len(minority)
    drop_count = len(majority) - target_majority
    rng.shuffle(majority)
    drops = set(id(q) for _, q in majority[:drop_count])

    for entry in entries:
        entry["questions"] = [q for q in entry["questions"] if id(q) not in drops]

    final_yes = sum(
        1 for e in entries for q in e["questions"] if q["answer"].strip().lower() == "yes"
    )
    final_no = sum(
        1 for e in entries for q in e["questions"] if q["answer"].strip().lower() == "no"
    )
    final_total = final_yes + final_no
    return {
        "yes": final_yes,
        "no": final_no,
        "ratio": round(final_yes / final_total, 4) if final_total else None,
        "dropped": drop_count,
    }


def compute_stats(
    train_entries: list[dict],
    val_entries: list[dict],
    yn_train: dict,
    yn_val: dict,
) -> dict:
    def per_sport(entries):
        return dict(Counter(e["sport"] for e in entries))

    def per_type(entries):
        c = Counter()
        for e in entries:
            for q in e["questions"]:
                c[q["question_type"]] += 1
        return dict(c)

    def total_qs(entries):
        return sum(len(e["questions"]) for e in entries)

    return {
        "train": {
            "videos": len(train_entries),
            "questions": total_qs(train_entries),
            "per_sport": per_sport(train_entries),
            "per_question_type": per_type(train_entries),
            "yes_no": yn_train,
        },
        "val": {
            "videos": len(val_entries),
            "questions": total_qs(val_entries),
            "per_sport": per_sport(val_entries),
            "per_question_type": per_type(val_entries),
            "yes_no": yn_val,
        },
    }


def print_summary(stats: dict) -> None:
    sports = sorted(set(stats["train"]["per_sport"]) | set(stats["val"]["per_sport"]))
    print("\n=== Subset summary ===")
    print(f"{'sport':<22} {'train_vids':>10} {'val_vids':>10}")
    for s in sports:
        t = stats["train"]["per_sport"].get(s, 0)
        v = stats["val"]["per_sport"].get(s, 0)
        print(f"{s:<22} {t:>10} {v:>10}")
    print(f"{'TOTAL':<22} {stats['train']['videos']:>10} {stats['val']['videos']:>10}")

    print("\nQuestions per type:")
    qtypes = sorted(
        set(stats["train"]["per_question_type"]) | set(stats["val"]["per_question_type"])
    )
    print(f"{'type':<18} {'train':>8} {'val':>8}")
    for q in qtypes:
        t = stats["train"]["per_question_type"].get(q, 0)
        v = stats["val"]["per_question_type"].get(q, 0)
        print(f"{q:<18} {t:>8} {v:>8}")

    print("\nYes/no balance:")
    for split in ("train", "val"):
        yn = stats[split]["yes_no"]
        print(
            f"  {split}: yes={yn['yes']} no={yn['no']} "
            f"ratio={yn['ratio']} dropped={yn['dropped']}"
        )
    print(
        f"\nTotal questions: train={stats['train']['questions']} "
        f"val={stats['val']['questions']}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--annotations-dir", type=Path, default=Path("meta-data"))
    p.add_argument("--out-dir", type=Path, default=Path("data/subsets"))
    p.add_argument("--train-per-sport", type=int, default=200)
    p.add_argument("--val-per-sport", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"Loading annotations from {args.annotations_dir}/ ...")
    train_rows = load_split(args.annotations_dir, "train")
    val_rows = load_split(args.annotations_dir, "val")
    print(f"  train: {len(train_rows)} QAs, val: {len(val_rows)} QAs")

    train_by_video = group_by_video(train_rows)
    val_by_video = group_by_video(val_rows)

    fg_map_train = build_fg_sub_sport_map(train_rows)
    fg_map_val = build_fg_sub_sport_map(val_rows)
    n_fg_train = sum(1 for v in train_by_video if v.startswith("fg/"))
    n_fg_val = sum(1 for v in val_by_video if v.startswith("fg/"))
    print(f"  fg sub-sport recovered: train {len(fg_map_train)}/{n_fg_train}, "
          f"val {len(fg_map_val)}/{n_fg_val}")

    print(f"\nSelecting {args.train_per_sport} train + {args.val_per_sport} val "
          f"videos per sport ...")
    chosen_train = select_videos_per_sport(
        train_by_video, args.train_per_sport, rng, fg_map_train
    )
    chosen_val = select_videos_per_sport(
        val_by_video, args.val_per_sport, rng, fg_map_val
    )

    train_entries = build_video_entries(chosen_train, train_by_video, "train", rng)
    val_entries = build_video_entries(chosen_val, val_by_video, "val", rng)

    train_ids = {e["video_id"] for e in train_entries}
    val_ids = {e["video_id"] for e in val_entries}
    overlap = train_ids & val_ids
    assert not overlap, f"train/val overlap: {len(overlap)} videos"

    yn_train = balance_yes_no(train_entries, rng)
    yn_val = balance_yes_no(val_entries, rng)

    stats = compute_stats(train_entries, val_entries, yn_train, yn_val)

    train_path = args.out_dir / "subset_train.json"
    val_path = args.out_dir / "subset_val.json"
    stats_path = args.out_dir / "subset_stats.json"
    manifest_path = args.out_dir / "subset_video_manifest.json"

    with train_path.open("w") as f:
        json.dump(train_entries, f, indent=2)
    with val_path.open("w") as f:
        json.dump(val_entries, f, indent=2)
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    train_sport_lookup = {e["video_id"]: e["sport"] for e in train_entries}
    val_sport_lookup = {e["video_id"]: e["sport"] for e in val_entries}

    # Manifest of every selected video and its split — for downloading from
    # HuggingFace on a rented GPU. Includes both a flat list (one row per video)
    # and a split-keyed convenience map so consumers can pick whichever fits.
    manifest = {
        "seed": args.seed,
        "videos": (
            [{"video_id": v, "split": "train", "sport": train_sport_lookup.get(v) or val_sport_lookup.get(v) or raw_prefix(v)} for v in sorted(train_ids)]
            + [{"video_id": v, "split": "val", "sport": train_sport_lookup.get(v) or val_sport_lookup.get(v) or raw_prefix(v)} for v in sorted(val_ids)]
        ),
        "by_split": {
            "train": sorted(train_ids),
            "val": sorted(val_ids),
        },
    }
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote:\n  {train_path}\n  {val_path}\n  {stats_path}\n  {manifest_path}")
    print_summary(stats)


if __name__ == "__main__":
    main()
