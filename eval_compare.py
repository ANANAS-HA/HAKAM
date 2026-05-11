"""Flip-rate analysis on the before/after distillation eval jsonl files.

Reads `eval_before.jsonl` and `eval_after.jsonl`, joins on `qa_id`, and
classifies each held-out sample as `wrong→correct` / `correct→wrong` /
`correct→correct` / `wrong→wrong` using a fuzzy correctness criterion:
the gold label appears (or near-appears) inside the student's free-form
answer.

Correctness is decided by sliding-window Levenshtein:
    best_window_sim(student_text, gold) >= threshold
This catches near-misses like "aerobics gymnastics" vs "aerobic gymnastics"
that the strict substring check (used in train_student.py) misses.

Outputs:
  - stdout: aggregate buckets, similarity averages, sport / qtype breakdowns
  - eval_flip_report.md: full markdown side-by-side of every flip-to-correct
    and every regression, plus the aggregate tables

Usage:
    python eval_compare.py
    python eval_compare.py --threshold 0.85
    python eval_compare.py --before X.jsonl --after Y.jsonl --report Z.md
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import Levenshtein

ROOT = Path(__file__).parent


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


def lev_sim(a: str, b: str) -> float:
    """1 − normalized Levenshtein over the full strings. Mirrors the helper
    in train_student.py — used for teacher-text similarity, where both
    strings are paragraphs."""
    a, b = a or "", b or ""
    n = max(len(a), len(b))
    if n == 0:
        return 1.0
    return 1.0 - Levenshtein.distance(a, b) / n


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


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


def fmt_table(rows: list[tuple], headers: tuple, widths: tuple = None) -> str:
    if widths is None:
        widths = tuple(max(len(str(r[i])) for r in [headers] + rows)
                       for i in range(len(headers)))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    out = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    for r in rows:
        out.append(fmt.format(*r))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="eval_before.jsonl")
    ap.add_argument("--after", default="eval_after.jsonl")
    ap.add_argument("--report", default="eval_flip_report.md")
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="best_window_sim cutoff for 'correct'. Default 0.80.")
    args = ap.parse_args()

    before = load_jsonl(ROOT / args.before)
    after = load_jsonl(ROOT / args.after)
    after_by_id = {r["qa_id"]: r for r in after}

    rows = []
    for b in before:
        a = after_by_id.get(b["qa_id"])
        if a is None:
            continue
        gold = b["gold"]
        sim_b = best_window_sim(b["student_text"], gold)
        sim_a = best_window_sim(a["student_text"], gold)
        tsim_b = lev_sim(b["student_text"], b["teacher_gen_text"])
        tsim_a = lev_sim(a["student_text"], a["teacher_gen_text"])
        bucket = classify(sim_b, sim_a, args.threshold)
        # Prefer the curated `sport` field from the subset (set by
        # eval_test_predict.py from sample_test_subset.py's recovered FineGym
        # sub-events: Vault / Uneven_Bar / Balance_Beam / Floor_Exercise).
        # Fall back to the video_id prefix for older prediction files that
        # didn't carry sport metadata.
        video_id = b["video_id"]
        sport = b.get("sport") or a.get("sport") or video_id.split("/", 1)[0]
        rows.append({
            "qa_id":         b["qa_id"],
            "video_id":      video_id,
            "sport":         sport,
            "question_type": b["question_type"],
            "question":      b["question"],
            "gold":          gold,
            "before":        b["student_text"],
            "after":         a["student_text"],
            "teacher":       b["teacher_gen_text"],
            "sim_b":         sim_b,
            "sim_a":         sim_a,
            "tsim_b":        tsim_b,
            "tsim_a":        tsim_a,
            "delta_sim":     sim_a - sim_b,
            "bucket":        bucket,
        })

    n = len(rows)
    if n == 0:
        raise SystemExit("No paired rows — check qa_id alignment between the jsonl files.")

    # -------- aggregates --------
    bucket_counts = Counter(r["bucket"] for r in rows)
    fix = bucket_counts["fix"]
    regress = bucket_counts["regress"]
    persist_ok = bucket_counts["persist_ok"]
    persist_bad = bucket_counts["persist_bad"]
    correct_b_total = persist_ok + regress
    correct_a_total = persist_ok + fix
    avg_sim_b = sum(r["sim_b"] for r in rows) / n
    avg_sim_a = sum(r["sim_a"] for r in rows) / n
    avg_tsim_b = sum(r["tsim_b"] for r in rows) / n
    avg_tsim_a = sum(r["tsim_a"] for r in rows) / n

    print(f"\n=== FLIP-RATE REPORT  (threshold={args.threshold}, n={n}) ===\n")
    print(fmt_table(
        [
            ("wrong → correct (FIX)",      f"{fix:>3}", f"{fix/n:.1%}"),
            ("correct → wrong (REGRESS)",  f"{regress:>3}", f"{regress/n:.1%}"),
            ("correct → correct",          f"{persist_ok:>3}", f"{persist_ok/n:.1%}"),
            ("wrong → wrong",              f"{persist_bad:>3}", f"{persist_bad/n:.1%}"),
        ],
        ("bucket", "n", "share"),
    ))
    print()
    print(fmt_table(
        [
            ("correct rate (gold-fuzzy)",   f"{correct_b_total/n:.3f}",
             f"{correct_a_total/n:.3f}",
             f"{(correct_a_total - correct_b_total)/n:+.3f}"),
            ("avg gold best-window sim",    f"{avg_sim_b:.3f}",
             f"{avg_sim_a:.3f}", f"{avg_sim_a - avg_sim_b:+.3f}"),
            ("avg teacher-text sim",        f"{avg_tsim_b:.3f}",
             f"{avg_tsim_a:.3f}", f"{avg_tsim_a - avg_tsim_b:+.3f}"),
        ],
        ("metric", "before", "after", "Δ"),
    ))

    # -------- by sport --------
    print("\n--- by sport ---")
    sports = sorted({r["sport"] for r in rows})
    sport_rows = []
    for sp in sports:
        sub = [r for r in rows if r["sport"] == sp]
        ns = len(sub)
        bf = Counter(r["bucket"] for r in sub)
        sport_rows.append((
            sp, ns,
            bf["fix"], bf["regress"], bf["persist_ok"], bf["persist_bad"],
            f"{(bf['persist_ok'] + bf['fix']) / ns:.3f}",
        ))
    print(fmt_table(sport_rows,
                    ("sport", "n", "fix", "regress", "persist_ok", "persist_bad",
                     "correct_after")))

    # -------- by question_type --------
    print("\n--- by question_type ---")
    qtypes = sorted({r["question_type"] for r in rows})
    qt_rows = []
    for qt in qtypes:
        sub = [r for r in rows if r["question_type"] == qt]
        nq = len(sub)
        bf = Counter(r["bucket"] for r in sub)
        qt_rows.append((
            qt, nq,
            bf["fix"], bf["regress"], bf["persist_ok"], bf["persist_bad"],
            f"{(bf['persist_ok'] + bf['fix']) / nq:.3f}",
        ))
    print(fmt_table(qt_rows,
                    ("qtype", "n", "fix", "regress", "persist_ok", "persist_bad",
                     "correct_after")))

    # -------- markdown report --------
    fixes = sorted([r for r in rows if r["bucket"] == "fix"],
                   key=lambda r: r["delta_sim"], reverse=True)
    regs = sorted([r for r in rows if r["bucket"] == "regress"],
                  key=lambda r: r["delta_sim"])

    md = []
    md.append(f"# Flip-rate report — distillation before vs. after\n")
    md.append(f"Threshold (best_window_sim ≥): **{args.threshold}**  ·  "
              f"n = **{n}** held-out samples\n")
    md.append("## Buckets\n")
    md.append("| bucket | n | share |\n|---|---:|---:|\n")
    md.append(f"| **wrong → correct (FIX)** | {fix} | {fix/n:.1%} |\n")
    md.append(f"| correct → wrong (REGRESS) | {regress} | {regress/n:.1%} |\n")
    md.append(f"| correct → correct         | {persist_ok} | {persist_ok/n:.1%} |\n")
    md.append(f"| wrong → wrong             | {persist_bad} | {persist_bad/n:.1%} |\n\n")

    md.append("## Aggregate similarity\n")
    md.append("| metric | before | after | Δ |\n|---|---:|---:|---:|\n")
    md.append(f"| correct rate (gold-fuzzy ≥ {args.threshold}) "
              f"| {correct_b_total/n:.3f} | {correct_a_total/n:.3f} "
              f"| {(correct_a_total - correct_b_total)/n:+.3f} |\n")
    md.append(f"| avg gold best-window sim | {avg_sim_b:.3f} "
              f"| {avg_sim_a:.3f} | {avg_sim_a - avg_sim_b:+.3f} |\n")
    md.append(f"| avg teacher-text sim     | {avg_tsim_b:.3f} "
              f"| {avg_tsim_a:.3f} | {avg_tsim_a - avg_tsim_b:+.3f} |\n\n")

    md.append("## By sport\n")
    md.append("| sport | n | fix | regress | persist_ok | persist_bad | correct_after |\n"
              "|---|---:|---:|---:|---:|---:|---:|\n")
    for sp, ns, f_, rg, po, pb, ca in sport_rows:
        md.append(f"| {sp} | {ns} | {f_} | {rg} | {po} | {pb} | {ca} |\n")
    md.append("\n")

    md.append("## By question_type\n")
    md.append("| qtype | n | fix | regress | persist_ok | persist_bad | correct_after |\n"
              "|---|---:|---:|---:|---:|---:|---:|\n")
    for qt, nq, f_, rg, po, pb, ca in qt_rows:
        md.append(f"| {qt} | {nq} | {f_} | {rg} | {po} | {pb} | {ca} |\n")
    md.append("\n")

    md.append(f"## Flip-to-correct rows  ({len(fixes)})\n")
    md.append("Sorted by Δ gold-similarity (largest gain first).\n\n")
    for r in fixes:
        md.append(f"### qa_id={r['qa_id']}  ·  {r['sport']} / {r['question_type']}\n")
        md.append(f"- **Q:** {r['question']}\n")
        md.append(f"- **Gold:** `{r['gold']}`\n")
        md.append(f"- **sim:** {r['sim_b']:.3f} → **{r['sim_a']:.3f}**  "
                  f"(Δ {r['delta_sim']:+.3f})\n\n")
        md.append(f"  **BEFORE:** {r['before'][:600]}\n\n")
        md.append(f"  **AFTER:**  {r['after'][:600]}\n\n---\n\n")

    md.append(f"## Regressions  ({len(regs)})\n")
    md.append("Sorted by Δ gold-similarity (worst regression first).\n\n")
    for r in regs:
        md.append(f"### qa_id={r['qa_id']}  ·  {r['sport']} / {r['question_type']}\n")
        md.append(f"- **Q:** {r['question']}\n")
        md.append(f"- **Gold:** `{r['gold']}`\n")
        md.append(f"- **sim:** {r['sim_b']:.3f} → **{r['sim_a']:.3f}**  "
                  f"(Δ {r['delta_sim']:+.3f})\n\n")
        md.append(f"  **BEFORE:** {r['before'][:600]}\n\n")
        md.append(f"  **AFTER:**  {r['after'][:600]}\n\n---\n\n")

    (ROOT / args.report).write_text("".join(md))
    print(f"\nReport written to: {args.report}")


if __name__ == "__main__":
    main()
