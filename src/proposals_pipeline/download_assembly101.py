#!/usr/bin/env python3
"""
Assembly101 selective downloader (Hugging Face mirror: cvml-nus/assembly101).

The HF repo mirrors the original Google Drive layout: recordings are stored as
flat per-recording folders, NOT split into train/val/test directories. Splits
exist only inside `annotations/`. This script therefore:

  1. downloads `annotations/**` (small),
  2. resolves a recording list -- either from a split file or by random sample,
  3. builds allow_patterns restricted to the views you asked for,
  4. reports the exact byte size BEFORE transferring anything,
  5. downloads.

Prereqs:
    pip install "huggingface_hub[hf_transfer]"
    hf auth login          # repo is gated (CC BY-NC 4.0) -- accept terms on the
                           # dataset page first

Examples:
    # 10 random recordings, one fixed view each, size check only
    python download_assembly101.py --n-random 10 --views v1 --dry-run

    # same, actually download
    python download_assembly101.py --n-random 10 --views v1

    # every recording in the validation split, all 4 egocentric streams
    python download_assembly101.py --split validation --views ego

    # just the annotations
    python download_assembly101.py --annotations-only
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
except ImportError:
    sys.exit("huggingface_hub is required:  pip install 'huggingface_hub[hf_transfer]'")

REPO_ID = "cvml-nus/assembly101"

# Camera-ID -> view mapping (from the official download-scripts README).
FIXED_VIEWS = {
    "v1": "C10095_rgb",
    "v2": "C10115_rgb",
    "v3": "C10118_rgb",
    "v4": "C10119_rgb",
    "v5": "C10379_rgb",
    "v6": "C10390_rgb",
    "v7": "C10395_rgb",
    "v8": "C10404_rgb",
}

# Egocentric camera IDs differ between recordings (two hardware generations);
# both aliases are emitted as patterns and the non-matching one simply
# resolves to nothing.
EGO_VIEWS = {
    "e1": ["HMC_84346135_mono10bit", "HMC_21176875_mono10bit"],
    "e2": ["HMC_84347414_mono10bit", "HMC_21176623_mono10bit"],
    "e3": ["HMC_84355350_mono10bit", "HMC_21110305_mono10bit"],
    "e4": ["HMC_84358933_mono10bit", "HMC_21179183_mono10bit"],
}

RECORDING_RE = re.compile(r"nusar-\d{4}_action_[a-z]+_[A-Za-z0-9_.\-]+")

# Rough per-file expectation, used only to sanity-check the reported total.
GB = 1_000_000_000


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def resolve_view_patterns(views: list[str]) -> list[str]:
    """Turn view specs into filename globs relative to a recording folder."""
    stems: list[str] = []
    for v in views:
        v = v.strip().lower()
        if v == "all":
            return ["*.mp4"]
        elif v == "fixed":
            stems.extend(FIXED_VIEWS.values())
        elif v == "ego":
            for aliases in EGO_VIEWS.values():
                stems.extend(aliases)
        elif v in FIXED_VIEWS:
            stems.append(FIXED_VIEWS[v])
        elif v in EGO_VIEWS:
            stems.extend(EGO_VIEWS[v])
        else:
            # allow a raw camera ID, e.g. C10095_rgb
            stems.append(v.replace(".mp4", ""))
    return [f"{s}.mp4" for s in dict.fromkeys(stems)]


def list_repo_recordings(api: HfApi, repo_id: str) -> list[str]:
    """All recording folder names present under recordings/ in the repo."""
    files = api.list_repo_files(repo_id, repo_type="dataset")
    recs = {f.split("/")[1] for f in files if f.startswith("recordings/") and "/" in f[11:]}
    return sorted(recs)


def recordings_from_split(ann_dir: Path, split: str) -> list[str]:
    """
    Scrape recording names out of whichever annotation files belong to `split`.

    Handles both granularities without hardcoding filenames: fine-grained
    splits are CSVs (train.csv / validation.csv / test.csv), coarse splits are
    txt lists under coarse_splits/.
    """
    if not ann_dir.is_dir():
        sys.exit(f"annotations not found at {ann_dir} -- run without --skip-annotations")

    candidates = [
        p
        for p in ann_dir.rglob("*")
        if p.is_file()
        and split in p.name.lower()
        and p.suffix.lower() in {".csv", ".txt"}
    ]
    if not candidates:
        listing = "\n  ".join(sorted(str(p.relative_to(ann_dir)) for p in ann_dir.rglob("*") if p.is_file())[:40])
        sys.exit(f"no annotation file matching split '{split}'. Files present:\n  {listing}")

    print(f"[split] reading {len(candidates)} file(s):")
    for p in candidates:
        print(f"        {p.relative_to(ann_dir)}")

    recs: set[str] = set()
    for p in candidates:
        text = p.read_text(errors="ignore")
        recs.update(m.group(0).split("/")[0] for m in RECORDING_RE.finditer(text))
    return sorted(recs)


# --------------------------------------------------------------------------- #
# sizing
# --------------------------------------------------------------------------- #
def measure(api: HfApi, repo_id: str, recordings: list[str], view_globs: list[str]) -> tuple[int, int]:
    """Return (n_files, total_bytes) for the exact selection, via the repo tree."""
    all_files = api.list_repo_files(repo_id, repo_type="dataset")
    wanted_prefixes = {f"recordings/{r}/" for r in recordings}

    def matches(path: str) -> bool:
        if not any(path.startswith(p) for p in wanted_prefixes):
            return False
        name = path.rsplit("/", 1)[-1]
        return any(
            name == g or (g == "*.mp4" and name.endswith(".mp4"))
            for g in view_globs
        )

    selected = [f for f in all_files if matches(f)]
    if not selected:
        return 0, 0

    total = 0
    for i in range(0, len(selected), 400):  # API caps paths per request
        chunk = selected[i : i + 400]
        for info in api.get_paths_info(repo_id, chunk, repo_type="dataset"):
            total += getattr(info, "size", 0) or 0
    return len(selected), total


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Selectively download Assembly101 from the Hugging Face mirror.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--out", default="./assembly101", help="local target directory")

    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--split", help="train | validation | test (matched against annotation filenames)")
    sel.add_argument("--n-random", type=int, help="sample N recordings uniformly from the repo")
    sel.add_argument("--recordings-file", help="newline-separated recording names to fetch")

    ap.add_argument("--seed", type=int, default=0, help="RNG seed for --n-random (reproducible draws)")
    ap.add_argument(
        "--views",
        nargs="+",
        default=["v1"],
        help="v1..v8, e1..e4, 'fixed' (8 RGB), 'ego' (4 mono), 'all' (12), or a raw camera ID",
    )
    ap.add_argument("--annotations-only", action="store_true", help="fetch annotations/ and exit")
    ap.add_argument("--skip-annotations", action="store_true", help="assume annotations/ is already local")
    ap.add_argument("--dry-run", action="store_true", help="report file count and size, download nothing")
    ap.add_argument("--yes", action="store_true", help="skip the size confirmation prompt")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--warn-gb", type=float, default=20.0, help="prompt for confirmation above this size")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    api = HfApi()
    out = Path(args.out)

    # --- annotations ------------------------------------------------------- #
    if not args.skip_annotations:
        print("[1/4] annotations ...")
        try:
            snapshot_download(
                args.repo,
                repo_type="dataset",
                allow_patterns=["annotations/**"],
                local_dir=str(out),
                max_workers=args.max_workers,
            )
        except GatedRepoError:
            sys.exit(
                f"{args.repo} is gated. Accept the licence at "
                f"https://huggingface.co/datasets/{args.repo} and run `hf auth login`."
            )
        except HfHubHTTPError as e:
            sys.exit(f"download failed: {e}")

    if args.annotations_only:
        print(f"done -> {out/'annotations'}")
        return

    # --- recording selection ----------------------------------------------- #
    print("[2/4] resolving recordings ...")
    if args.split:
        recordings = recordings_from_split(out / "annotations", args.split.lower())
    elif args.recordings_file:
        recordings = [
            ln.strip()
            for ln in Path(args.recordings_file).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        recordings = sorted(dict.fromkeys(recordings))
    elif args.n_random:
        pool = list_repo_recordings(api, args.repo)
        print(f"        {len(pool)} recordings in repo")
        if args.n_random > len(pool):
            sys.exit(f"asked for {args.n_random}, only {len(pool)} available")
        rng = random.Random(args.seed)
        recordings = sorted(rng.sample(pool, args.n_random))
    else:
        recordings = list_repo_recordings(api, args.repo)

    view_globs = resolve_view_patterns(args.views)
    print(f"        {len(recordings)} recording(s) x {len(view_globs)} view pattern(s)")
    for r in recordings[:10]:
        print(f"        - {r}")
    if len(recordings) > 10:
        print(f"        ... and {len(recordings) - 10} more")

    (out / "selected_recordings.txt").parent.mkdir(parents=True, exist_ok=True)
    (out / "selected_recordings.txt").write_text("\n".join(recordings) + "\n")

    # --- size check --------------------------------------------------------- #
    print("[3/4] measuring ...")
    n_files, n_bytes = measure(api, args.repo, recordings, view_globs)
    if n_files == 0:
        sys.exit("selection matched 0 files -- check --views and the recording names")
    print(f"        {n_files} files, {n_bytes/GB:.2f} GB ({n_bytes/n_files/1e6:.0f} MB/file avg)")

    if args.dry_run:
        print("dry run -- nothing downloaded")
        return

    if n_bytes / GB > args.warn_gb and not args.yes:
        if input(f"        proceed with {n_bytes/GB:.1f} GB? [y/N] ").strip().lower() not in {"y", "yes"}:
            sys.exit("aborted")

    # --- download ----------------------------------------------------------- #
    print("[4/4] downloading ...")
    allow = [f"recordings/{r}/{g}" for r in recordings for g in view_globs]
    snapshot_download(
        args.repo,
        repo_type="dataset",
        allow_patterns=allow,
        local_dir=str(out),
        max_workers=args.max_workers,
    )
    print(f"done -> {out/'recordings'}")


if __name__ == "__main__":
    main()
