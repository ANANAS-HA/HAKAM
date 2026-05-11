"""Download only the Sports-QA videos listed in the subset manifest from a
HuggingFace dataset repo.

Strategy:
1. Read `subset_video_manifest.json` (produced by `sample_subset.py`).
2. List every file in the HF repo once (`HfApi().list_repo_files`).
3. For each subset video stem, find the matching repo file by suffix
   (the `<sport>/<stem>.<ext>` suffix is stable across plausible layouts:
   `videos/<sport>/<stem>.mp4`, `videos/finegym_0X/<stem>.mp4`, etc.).
4. Download matched files in parallel via `hf_hub_download`.

Usage:
    python download_subset_videos.py --repo-id YOUR_USER/sports-qa-videos
    python download_subset_videos.py --repo-id ... --splits train
    python download_subset_videos.py --repo-id ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".webm", ".mov")
FG_PREFIX = "fg/"


def load_manifest(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def index_repo_files(repo_files: list[str]) -> dict[str, list[str]]:
    """Map `<sport>/<stem>` (no ext) -> list of matching repo paths.

    fg videos live under `finegym_01..05/<stem>.<ext>` in the original repo
    layout, so we also bucket every finegym_0X file under the synthetic
    `fg/<stem>` key to match the manifest's `fg/...` ids.
    """
    idx: dict[str, list[str]] = {}
    for rf in repo_files:
        if not rf.lower().endswith(VIDEO_EXTS):
            continue
        # strip extension
        no_ext = rf.rsplit(".", 1)[0]
        # tail = path components excluding any leading "videos/" prefix
        parts = no_ext.split("/")
        if len(parts) < 2:
            continue
        # rebuild possible keys: every contiguous 2-segment tail
        # (sport/stem) that the manifest could reference.
        sport, stem = parts[-2], parts[-1]
        # fg synonym: any finegym_0X bucket → fg
        if sport.startswith("finegym"):
            idx.setdefault(f"fg/{stem}", []).append(rf)
        idx.setdefault(f"{sport}/{stem}", []).append(rf)
        # also tolerate doubled volleyball (videos/volleyball/volleyball/...)
        if len(parts) >= 3 and parts[-3] == "volleyball" and sport == "volleyball":
            idx.setdefault(f"volleyball/{stem}", []).append(rf)
    return idx


def resolve_targets(
    video_ids: list[str], idx: dict[str, list[str]]
) -> tuple[list[tuple[str, str]], list[str]]:
    """Returns (matched: [(video_id, repo_path)], missing: [video_id])."""
    matched, missing = [], []
    for vid in video_ids:
        candidates = idx.get(vid)
        if not candidates:
            missing.append(vid)
            continue
        # Prefer .mp4, then first by sort
        candidates_sorted = sorted(
            candidates,
            key=lambda p: (0 if p.lower().endswith(".mp4") else 1, p),
        )
        matched.append((vid, candidates_sorted[0]))
    return matched, missing


def download_one(
    repo_id: str,
    repo_path: str,
    out_dir: Path,
    repo_type: str,
    revision: str | None,
    token: str | None,
) -> tuple[str, Path | None, str | None]:
    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            repo_type=repo_type,
            revision=revision,
            token=token,
            local_dir=str(out_dir),
            # cache_dir left as default so we benefit from HF's content-hash cache
        )
        return repo_path, Path(local), None
    except HfHubHTTPError as e:
        return repo_path, None, str(e)
    except Exception as e:  # noqa: BLE001
        return repo_path, None, f"{type(e).__name__}: {e}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo-id", required=True,
                   help="HuggingFace repo, e.g. 'username/KnowledgeDistillationVLMsQA'")
    p.add_argument("--repo-type", default="dataset",
                   choices=("dataset", "model", "space"))
    p.add_argument("--revision", default=None,
                   help="Branch / tag / commit (default: main)")
    p.add_argument("--manifest", type=Path,
                   default=Path("data/subsets/subset_video_manifest.json"))
    p.add_argument("--out-dir", type=Path, default=Path("videos"),
                   help="Local destination root (default: videos/)")
    p.add_argument("--splits", nargs="+", default=["train", "val"],
                   choices=["train", "val", "test"])
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel download workers")
    p.add_argument("--token", default=None,
                   help="HF token (else uses HF_TOKEN env / cached login)")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve & report what would be downloaded; don't fetch")
    args = p.parse_args()

    if not args.manifest.exists():
        sys.exit(f"Manifest not found: {args.manifest}\n"
                 "Run sample_subset.py first.")

    manifest = load_manifest(args.manifest)
    wanted: list[str] = []
    for split in args.splits:
        wanted.extend(manifest["by_split"].get(split, []))
    wanted = sorted(set(wanted))
    print(f"Manifest: {len(wanted)} unique videos requested "
          f"(splits={args.splits}).")

    print(f"Listing files in {args.repo_type} '{args.repo_id}' ...")
    api = HfApi(token=args.token)
    try:
        repo_files = api.list_repo_files(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
        )
    except HfHubHTTPError as e:
        sys.exit(f"Failed to list repo files: {e}")
    print(f"  {len(repo_files)} files in repo.")

    idx = index_repo_files(repo_files)
    matched, missing = resolve_targets(wanted, idx)
    print(f"  matched: {len(matched)}, missing: {len(missing)}")

    if missing:
        print(f"\nWARNING: {len(missing)} video(s) not found in repo. "
              "First 10:")
        for v in missing[:10]:
            print(f"  - {v}")
        miss_path = args.out_dir.parent / "subset_missing_videos.txt"
        miss_path.parent.mkdir(parents=True, exist_ok=True)
        miss_path.write_text("\n".join(missing) + "\n")
        print(f"  full list -> {miss_path}")

    if args.dry_run:
        print("\nDry run — example resolutions:")
        for vid, rp in matched[:10]:
            print(f"  {vid}  ->  {rp}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {len(matched)} files into {args.out_dir}/ "
          f"with {args.workers} workers ...")

    successes, failures = 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(
                download_one,
                args.repo_id, rp, args.out_dir,
                args.repo_type, args.revision, args.token,
            ): (vid, rp)
            for vid, rp in matched
        }
        for i, fut in enumerate(as_completed(futs), 1):
            vid, rp = futs[fut]
            _, local, err = fut.result()
            if err:
                failures.append((vid, rp, err))
                print(f"  [{i}/{len(matched)}] FAIL {rp}: {err}")
            else:
                successes += 1
                if i % 50 == 0 or i == len(matched):
                    print(f"  [{i}/{len(matched)}] ok ({successes} done)")

    print(f"\nDone. Successes: {successes}, Failures: {len(failures)}, "
          f"Missing-from-repo: {len(missing)}")
    if failures:
        fail_path = args.out_dir.parent / "subset_failed_downloads.txt"
        with fail_path.open("w") as f:
            for vid, rp, err in failures:
                f.write(f"{vid}\t{rp}\t{err}\n")
        print(f"Failure log -> {fail_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
