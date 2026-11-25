"""SCAD dataset discovery, leak-free splitting, and 16-frame clip sampling."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

POSITIVE_ALIASES = {"accident", "accidents", "positive", "crash", "collision", "1", "true"}
NEGATIVE_ALIASES = {"normal", "non_accident", "non-accidents", "negative", "no_accident", "0", "false"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class ClipSample:
    """Single labeled clip sample (video file or directory of ordered frames)."""

    path: Path
    label: int  # 1 = accident, 0 = normal
    split: str  # "train", "val", or "test"
    video_group: str


def _parse_label(raw: str) -> int:
    key = raw.strip().lower()
    if key in POSITIVE_ALIASES:
        return 1
    if key in NEGATIVE_ALIASES:
        return 0
    raise ValueError(f"Unrecognized class label '{raw}'")


def _is_frame_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in path.iterdir())


def _infer_video_group(path: Path) -> str:
    """Extract parent scene/video identifier to avoid clip leakage across splits."""
    stem = path.stem
    if "_clip_" in stem:
        return stem.split("_clip_", 1)[0]
    if "_seg_" in stem:
        return stem.split("_seg_", 1)[0]
    return stem


def _collect_class_items(class_dir: Path, label: int, split: str) -> list[ClipSample]:
    items: list[ClipSample] = []
    if not class_dir.is_dir():
        return items

    for entry in sorted(class_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() in VIDEO_EXTENSIONS:
            items.append(ClipSample(entry, label, split, _infer_video_group(entry)))
        elif _is_frame_dir(entry):
            items.append(ClipSample(entry, label, split, _infer_video_group(entry)))
        elif entry.is_dir():
            for sub in sorted(entry.iterdir()):
                if sub.is_file() and sub.suffix.lower() in VIDEO_EXTENSIONS:
                    items.append(ClipSample(sub, label, split, entry.name))
                elif _is_frame_dir(sub):
                    items.append(ClipSample(sub, label, split, entry.name))
    return items


def _split_by_group(
    samples: list[ClipSample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[ClipSample]]:
    """Stratified group-level split so clips from the same source video stay in one split."""
    if not (0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0 and val_ratio + test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must be in [0, 1) and sum to < 1")

    rng = random.Random(seed)
    splits: dict[str, list[ClipSample]] = {"train": [], "val": [], "test": []}

    for label in (0, 1):
        grouped: dict[str, list[ClipSample]] = defaultdict(list)
        for sample in samples:
            if sample.label == label:
                grouped[sample.video_group].append(sample)

        group_keys = sorted(grouped.keys())
        rng.shuffle(group_keys)
        n_groups = len(group_keys)
        n_test = round(n_groups * test_ratio) if test_ratio > 0 and n_groups >= 3 else 0
        n_val = round(n_groups * val_ratio) if val_ratio > 0 and n_groups >= 3 else 0
        if n_test + n_val >= n_groups and n_groups > 0:
            n_test = min(n_test, max(0, n_groups - 2))
            n_val = min(n_val, max(0, n_groups - 1 - n_test))

        test_groups = set(group_keys[:n_test])
        val_groups = set(group_keys[n_test : n_test + n_val])

        for gkey, group_items in grouped.items():
            target_split = (
                "test" if gkey in test_groups else ("val" if gkey in val_groups else "train")
            )
            for item in group_items:
                splits[target_split].append(
                    ClipSample(item.path, item.label, target_split, item.video_group)
                )

    for split_list in splits.values():
        split_list.sort(key=lambda s: str(s.path))
    return splits


def discover_scad_samples(
    data_root: Optional[Path] = None,
    manifest_csv: Optional[Path] = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[ClipSample]]:
    """Discover SCAD clips from either a CSV manifest or directory tree."""
    if manifest_csv is not None:
        if not manifest_csv.is_file():
            raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")
        splits: dict[str, list[ClipSample]] = {"train": [], "val": [], "test": []}
        unsplit: list[ClipSample] = []
        with manifest_csv.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not {"path", "label"}.issubset(reader.fieldnames):
                raise ValueError("Manifest CSV must contain at least 'path' and 'label' columns")
            for row in reader:
                raw_path = Path(row["path"])
                resolved = raw_path if raw_path.is_absolute() or data_root is None else data_root / raw_path
                label = _parse_label(row["label"])
                group = row.get("video_group") or _infer_video_group(resolved)
                raw_split = (row.get("split") or "").strip().lower()
                if raw_split in SPLIT_ALIASES:
                    sp = SPLIT_ALIASES[raw_split]
                    splits[sp].append(ClipSample(resolved, label, sp, group))
                else:
                    unsplit.append(ClipSample(resolved, label, "train", group))
        if unsplit:
            auto = _split_by_group(unsplit, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
            for sp in ("train", "val", "test"):
                splits[sp].extend(auto[sp])
        return splits

    if data_root is None or not data_root.is_dir():
        raise FileNotFoundError(f"SCAD dataset directory not found: {data_root}")

    # Case 1: {data_root}/{train,val,test}/{accident,normal}/...
    splits = {"train": [], "val": [], "test": []}
    found_explicit_splits = False
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        split_name = SPLIT_ALIASES.get(child.name.lower())
        if split_name is None:
            continue
        found_explicit_splits = True
        for class_dir in sorted(child.iterdir()):
            if not class_dir.is_dir():
                continue
            try:
                label = _parse_label(class_dir.name)
            except ValueError:
                continue
            splits[split_name].extend(_collect_class_items(class_dir, label, split_name))

    if found_explicit_splits:
        if not splits["val"] and splits["train"] and val_ratio > 0:
            # Carve validation split deterministically from train when dataset only has train+test.
            re_split = _split_by_group(splits["train"], val_ratio=val_ratio, test_ratio=0.0, seed=seed)
            splits["train"] = re_split["train"]
            splits["val"] = re_split["val"]
        return splits

    # Case 2: {data_root}/{accident,normal}/...
    all_samples: list[ClipSample] = []
    for class_dir in sorted(data_root.iterdir()):
        if not class_dir.is_dir():
            continue
        try:
            label = _parse_label(class_dir.name)
        except ValueError:
            continue
        all_samples.extend(_collect_class_items(class_dir, label, "train"))

    if not all_samples:
        raise RuntimeError(
            f"No video clips or frame folders discovered under {data_root}. "
            "Expected class folders (e.g. accident/ and normal/) or train/val/test subdirs."
        )

    return _split_by_group(all_samples, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)


def _compute_sample_indices(
    total_frames: int,
    num_frames: int,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    augment: bool = False,
    rng: Optional[random.Random] = None,
) -> list[int]:
    """Uniformly sample num_frames indices across [start_idx, end_idx]."""
    last = (total_frames - 1) if end_idx is None else min(total_frames - 1, end_idx)
    first = max(0, min(start_idx, last))
    span = last - first + 1
    if span <= 0:
        raise ValueError(f"Empty frame range [{first}, {last}]")

    if span == 1:
        return [first] * num_frames

    raw = np.linspace(first, last, num_frames)
    if augment and span > num_frames:
        r = rng or random
        step = (last - first) / num_frames
        jitter = np.array([r.uniform(-0.35 * step, 0.35 * step) for _ in range(num_frames)])
        raw = np.clip(raw + jitter, first, last)
    return [int(round(v)) for v in raw]


def _read_frames_from_video(
    video_path: Path,
    num_frames: int,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    augment: bool = False,
    rng: Optional[random.Random] = None,
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video clip: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            # Fallback sequential read when container lacks frame count metadata.
            all_frames: list[np.ndarray] = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                all_frames.append(frame)
            if not all_frames:
                raise RuntimeError(f"No readable frames in {video_path}")
            start_idx = max(0, int(round(start_sec * fps))) if (start_sec is not None and fps > 0) else 0
            end_idx = (
                min(len(all_frames) - 1, int(round(end_sec * fps)))
                if (end_sec is not None and fps > 0)
                else len(all_frames) - 1
            )
            indices = _compute_sample_indices(len(all_frames), num_frames, start_idx, end_idx, augment, rng)
            return [all_frames[i] for i in indices]

        start_idx = max(0, int(round(start_sec * fps))) if (start_sec is not None and fps > 0) else 0
        end_idx = (
            min(total - 1, int(round(end_sec * fps)))
            if (end_sec is not None and fps > 0)
            else total - 1
        )
        indices = _compute_sample_indices(total, num_frames, start_idx, end_idx, augment, rng)

        unique_sorted = sorted(set(indices))
        frame_cache: dict[int, np.ndarray] = {}
        first_target = unique_sorted[0]
        last_target = unique_sorted[-1]
        if first_target > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_target)
        pos = first_target
        target_ptr = 0
        while pos <= last_target and target_ptr < len(unique_sorted):
            ok, frame = cap.read()
            if not ok:
                break
            if pos == unique_sorted[target_ptr]:
                frame_cache[pos] = frame
                target_ptr += 1
            pos += 1

        if not frame_cache:
            raise RuntimeError(f"Failed to decode frames from {video_path}")

        available_keys = sorted(frame_cache.keys())
        sampled: list[np.ndarray] = []
        for idx in indices:
            if idx in frame_cache:
                sampled.append(frame_cache[idx])
            else:
                nearest = min(available_keys, key=lambda k: abs(k - idx))
                sampled.append(frame_cache[nearest])
        return sampled
    finally:
        cap.release()


def _read_frames_from_dir(
    frame_dir: Path,
    num_frames: int,
    augment: bool = False,
    rng: Optional[random.Random] = None,
) -> list[np.ndarray]:
    files = sorted(
        p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"No image frames found in {frame_dir}")
    indices = _compute_sample_indices(len(files), num_frames, augment=augment, rng=rng)
    frames: list[np.ndarray] = []
    for idx in indices:
        img = cv2.imread(str(files[idx]))
        if img is None:
            raise RuntimeError(f"Failed to read frame image: {files[idx]}")
        frames.append(img)
    return frames


def frames_to_tensor(
    frames_bgr: list[np.ndarray],
    input_size: int = 224,
    augment: bool = False,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """Resize BGR frames, apply clip-consistent augmentation, and normalize to (T, C, H, W)."""
    r = rng or random
    do_flip = augment and (r.random() < 0.5)
    brightness_scale = r.uniform(0.85, 1.15) if augment else 1.0

    processed: list[np.ndarray] = []
    for frame in frames_bgr:
        resized = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if do_flip:
            rgb = rgb[:, ::-1, :]
        if brightness_scale != 1.0:
            rgb = np.clip(rgb * brightness_scale, 0.0, 1.0)
        norm = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        processed.append(np.transpose(norm, (2, 0, 1)))  # (C, H, W)

    stacked = np.stack(processed, axis=0).astype(np.float32)
    return torch.from_numpy(stacked)


def sample_video_frames(
    source_path: Path,
    num_frames: int = 16,
    input_size: int = 224,
    augment: bool = False,
    rng: Optional[random.Random] = None,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> torch.Tensor:
    """Load num_frames uniformly sampled frames from a video file or frame directory."""
    if source_path.is_dir():
        frames = _read_frames_from_dir(source_path, num_frames=num_frames, augment=augment, rng=rng)
    elif source_path.is_file():
        frames = _read_frames_from_video(
            source_path,
            num_frames=num_frames,
            start_sec=start_sec,
            end_sec=end_sec,
            augment=augment,
            rng=rng,
        )
    else:
        raise FileNotFoundError(f"Clip source not found: {source_path}")
    return frames_to_tensor(frames, input_size=input_size, augment=augment, rng=rng)


class SCADClipDataset(Dataset):
    """PyTorch Dataset returning (clip_tensor, label_float) for SCAD video classification."""

    def __init__(
        self,
        samples: list[ClipSample],
        num_frames: int = 16,
        input_size: int = 224,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.samples = samples
        self.num_frames = num_frames
        self.input_size = input_size
        self.augment = augment
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        clip = sample_video_frames(
            sample.path,
            num_frames=self.num_frames,
            input_size=self.input_size,
            augment=self.augment,
            rng=self._rng,
        )
        label = torch.tensor(float(sample.label), dtype=torch.float32)
        return clip, label
