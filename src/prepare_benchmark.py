"""Audit labeled videos and create a source-disjoint train/val/test manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.video_groups import infer_video_group

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _probe(path: Path) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "json", str(path)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return False
    try:
        streams = json.loads(result.stdout).get("streams", [])
        return bool(streams and streams[0].get("width", 0) > 0 and streams[0].get("height", 0) > 0)
    except (ValueError, TypeError):
        return False


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(data_dir: Path, manifest_path: Path, report_path: Path, seed: int = 42) -> dict:
    files = [
        (path, label)
        for label in ("accident", "normal")
        for path in (data_dir / label).iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not files:
        raise RuntimeError(f"No labeled videos found in {data_dir}/accident and normal")

    pointer_files = []
    candidates = []
    for path, label in files:
        with path.open("rb") as handle:
            is_pointer = handle.read(80).startswith(b"version https://git-lfs.github.com/spec/")
        if is_pointer:
            pointer_files.append(str(path))
        else:
            candidates.append((path, label))

    with ThreadPoolExecutor(max_workers=12) as pool:
        validity = list(pool.map(_probe, (path for path, _ in candidates)))
    unreadable_files = [str(path) for (path, _), valid in zip(candidates, validity) if not valid]
    valid_files = [(path, label) for (path, label), valid in zip(candidates, validity) if valid]

    by_hash: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for path, label in valid_files:
        by_hash[_hash_file(path)].append((path, label))

    duplicates = []
    conflicting_labels = []
    unique_files = []
    for items in by_hash.values():
        if len({label for _, label in items}) > 1:
            conflicting_labels.extend(str(path) for path, _ in items)
            continue
        items.sort(key=lambda pair: ("(" in pair[0].stem, str(pair[0])))
        unique_files.append(items[0])
        duplicates.extend(str(path) for path, _ in items[1:])

    groups: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for path, label in unique_files:
        groups[infer_video_group(path)].append((path, label))

    rng = random.Random(seed)
    assigned: dict[str, str] = {}
    for kind in ("normal", "accident", "mixed"):
        keys = sorted(
            group for group, items in groups.items()
            if ("mixed" if len({label for _, label in items}) == 2 else items[0][1]) == kind
        )
        rng.shuffle(keys)
        n_test = round(len(keys) * 0.15) if len(keys) >= 3 else 0
        n_val = round(len(keys) * 0.15) if len(keys) >= 3 else 0
        if n_test + n_val >= len(keys):
            n_val = max(0, len(keys) - n_test - 1)
        for index, group in enumerate(keys):
            assigned[group] = "test" if index < n_test else ("val" if index < n_test + n_val else "train")

    rows = sorted(
        (
            str(path.relative_to(data_dir)), label, assigned[group], group
        )
        for group, items in groups.items()
        for path, label in items
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "split", "video_group"])
        writer.writerows(rows)

    counts = {split: dict(Counter(label for _, label, row_split, _ in rows if row_split == split))
              for split in ("train", "val", "test")}
    report = {
        "source_dir": str(data_dir), "seed": seed, "raw_files": len(files),
        "usable_clips": len(rows), "source_groups": len(groups), "split_counts": counts,
        "lfs_pointers": pointer_files, "unreadable_files": unreadable_files,
        "exact_duplicates_skipped": duplicates, "conflicting_labels_skipped": conflicting_labels,
        "manifest": str(manifest_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if not isinstance(value, list)}, indent=2))
    print(f"Audit details: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("data/benchmark_manifest.csv"))
    parser.add_argument("--report", type=Path, default=Path("outputs/eval/data_audit.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.data_dir, args.manifest, args.report, args.seed)


if __name__ == "__main__":
    main()
