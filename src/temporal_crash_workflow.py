"""Evaluate the published MobileNetV2 temporal crash model on labeled video clips.

Architecture and weights: beaunix/car-crash-detector (CC BY-NC-ND 4.0).
https://huggingface.co/beaunix/car-crash-detector
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import mobilenet_v2

from src.vit_workflow import _read_manifest, binary_metrics

MODEL_REPO = "beaunix/car-crash-detector"
WEIGHTS_NAME = "best_model.pt"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TemporalAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(1280, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(frames), dim=1)
        return (frames * weights).sum(dim=1)


class CrashDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = mobilenet_v2(weights=None)
        self.feature_extractor = nn.Sequential(
            backbone.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.temporal_attention = TemporalAttention()
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = clips.shape
        features = self.feature_extractor(clips.reshape(batch * steps, channels, height, width))
        return self.classifier(self.temporal_attention(features.reshape(batch, steps, 1280)))


def clip_tensor(path: Path, frames: int = 16) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < frames:
            raise RuntimeError(f"Video has fewer than {frames} frames: {path}")
        indices = np.linspace(0, total - 1, frames, dtype=np.int32)
        images = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {index}: {path}")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
            images.append(((rgb - MEAN) / STD).transpose(2, 0, 1))
        return torch.from_numpy(np.stack(images)).unsqueeze(0)
    finally:
        capture.release()


class TemporalCrashClassifier:
    def __init__(self, weights: Path, device: str = "cpu") -> None:
        if not weights.is_file():
            raise FileNotFoundError(f"Weights missing: {weights}")
        self.device = torch.device(device)
        self.model = CrashDetector()
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def score(self, path: Path) -> float:
        return float(self.model(clip_tensor(path).to(self.device)).item())


def _score_split(classifier: TemporalCrashClassifier, manifest: Path, data_dir: Path,
                 split: str, output_dir: Path) -> list[dict]:
    samples = _read_manifest(manifest, data_dir, split)
    records = []
    for index, (path, label) in enumerate(samples, 1):
        records.append({"video": str(path), "label": label, "score": classifier.score(path)})
        if index % 20 == 0:
            print(f"{split}: video model scored {index}/{len(samples)} clips", flush=True)
    (output_dir / f"{split}_raw_scores.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def _metrics(records: list[dict], threshold: float) -> dict:
    return binary_metrics([int(row["label"]) for row in records],
                          [int(row["score"] >= threshold) for row in records])


def _write_predictions(records: list[dict], threshold: float, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video", "actual", "p_accident", "predicted", "correct"])
        for row in records:
            predicted = int(row["score"] >= threshold)
            writer.writerow([row["video"], row["label"], round(row["score"], 6), predicted,
                             int(predicted == row["label"])])


def evaluate(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(args.threads, torch.get_num_threads())))
    classifier = TemporalCrashClassifier(args.weights, args.device)
    validation = _score_split(classifier, args.manifest, args.data_dir, "val", args.output_dir)
    options = []
    for step in range(1, 20):
        threshold = round(step * 0.05, 2)
        metrics = _metrics(validation, threshold)
        options.append((metrics["accuracy"], metrics["f1"], metrics["recall"],
                        -abs(threshold - 0.5), threshold))
    threshold = max(options)[4]
    test = _score_split(classifier, args.manifest, args.data_dir, "test", args.output_dir)
    val_metrics = _metrics(validation, threshold)
    test_metrics = _metrics(test, threshold)
    _write_predictions(validation, threshold, args.output_dir / "val_predictions.csv")
    _write_predictions(test, threshold, args.output_dir / "test_predictions.csv")
    report = {
        "model": MODEL_REPO, "weights": str(args.weights), "threshold": threshold,
        "frames_per_clip": 16, "validation": val_metrics, "test": test_metrics,
        "manifest": str(args.manifest), "license": "CC BY-NC-ND 4.0",
    }
    (args.output_dir / "temporal_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/benchmark_manifest.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--weights", type=Path,
                        default=Path("outputs/model_cache/beaunix_crash/best_model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval/temporal_crash"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
