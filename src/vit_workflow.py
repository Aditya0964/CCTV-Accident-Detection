"""Evaluate and run the pretrained accident-image ViT on short CCTV clips."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "tiya1012/vit-accident-image"


@dataclass(frozen=True)
class DecisionRule:
    aggregation: str
    threshold: float


def binary_metrics(labels: list[int], predictions: list[int]) -> dict[str, Any]:
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    n = len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": round((tp + tn) / n, 4) if n else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "support_pos": sum(labels), "support_neg": n - sum(labels),
    }


def aggregate(probabilities: list[float], name: str) -> float:
    ordered = sorted(probabilities, reverse=True)
    if name == "mean":
        return sum(ordered) / len(ordered)
    if name == "top2":
        return sum(ordered[:2]) / min(2, len(ordered))
    if name == "top3":
        return sum(ordered[:3]) / min(3, len(ordered))
    raise ValueError(f"Unknown aggregation: {name}")


def select_rule(records: list[dict[str, Any]]) -> tuple[DecisionRule, dict[str, Any]]:
    labels = [int(record["label"]) for record in records]
    if not labels or not (0 in labels and 1 in labels):
        raise ValueError("Validation requires both accident and normal clips")
    options = []
    for aggregation in ("mean", "top2", "top3"):
        scores = [aggregate(record["frame_probabilities"], aggregation) for record in records]
        for step in range(1, 20):
            threshold = round(step * 0.05, 2)
            metrics = binary_metrics(labels, [int(score >= threshold) for score in scores])
            options.append((metrics["accuracy"], metrics["f1"], metrics["recall"],
                            -abs(threshold - 0.5), aggregation, threshold, metrics))
    best = max(options)
    return DecisionRule(best[4], best[5]), best[6]


def sample_frames(video_path: Path, num_frames: int = 8,
                  start_sec: float | None = None, end_sec: float | None = None
                  ) -> tuple[list[Image.Image], list[float], float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if total <= 0 or fps <= 0:
            raise RuntimeError(f"Invalid frame count or FPS: {video_path}")
        first = max(0, int(round(start_sec * fps))) if start_sec is not None else 0
        last = min(total - 1, int(round(end_sec * fps))) if end_sec is not None else total - 1
        if first > last:
            raise ValueError(f"Empty video range in {video_path}: {start_sec} to {end_sec}")
        indices = np.linspace(first, last, num_frames + 2)[1:-1].round().astype(int)
        frames: list[Image.Image] = []
        times = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {index} in {video_path}")
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            times.append(round(float(index) / fps, 3))
        return frames, times, (last - first + 1) / fps
    finally:
        capture.release()


class ViTAccidentClassifier:
    def __init__(self, model_id: str = MODEL_ID, cache_dir: Path = Path("outputs/model_cache"),
                 device: str = "cpu") -> None:
        self.model_id = model_id
        self.device = torch.device(device)
        # Use already downloaded weights without making a network HEAD request on every run.
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                model_id, cache_dir=str(cache_dir), local_files_only=True)
            self.model = AutoModelForImageClassification.from_pretrained(
                model_id, cache_dir=str(cache_dir), local_files_only=True)
        except OSError:
            self.processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=str(cache_dir))
            self.model = AutoModelForImageClassification.from_pretrained(model_id, cache_dir=str(cache_dir))
        self.model.to(self.device).eval()
        labels = {int(key): str(value).lower() for key, value in self.model.config.id2label.items()}
        positive = [key for key, value in labels.items() if "accident" in value and not
                    any(term in value for term in ("non", "no ", "normal"))]
        if not positive and model_id == MODEL_ID and labels == {0: "label_0", 1: "label_1"}:
            # This model's config omits label names; its model card defines class 1 as accident.
            positive = [1]
        if len(positive) != 1:
            raise RuntimeError(f"Cannot identify accident class from model labels: {labels}")
        self.positive_index = positive[0]

    @torch.inference_mode()
    def score(self, video_path: Path, num_frames: int = 8,
              start_sec: float | None = None, end_sec: float | None = None) -> dict[str, Any]:
        frames, times, duration = sample_frames(video_path, num_frames, start_sec, end_sec)
        inputs = self.processor(images=frames, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        logits = self.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[:, self.positive_index].cpu().tolist()
        peak_index = int(np.argmax(probabilities))
        return {
            "video": str(video_path), "duration_sec": round(duration, 3),
            "sample_times_sec": times,
            "frame_probabilities": [round(float(value), 6) for value in probabilities],
            "peak_time_sec": times[peak_index],
            "peak_probability": round(float(probabilities[peak_index]), 6),
        }

    @torch.inference_mode()
    def features(self, video_path: Path, num_frames: int = 8) -> np.ndarray:
        """Mean and variation of ViT class-token features across sampled frames."""
        frames, _, _ = sample_frames(video_path, num_frames)
        inputs = self.processor(images=frames, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        tokens = self.model.vit(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
        return np.concatenate([tokens.mean(axis=0), tokens.std(axis=0)]).astype(np.float32)


def _read_manifest(manifest: Path, data_dir: Path, split: str) -> list[tuple[Path, int]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [(data_dir / row["path"], int(row["label"] == "accident"))
                for row in rows if row["split"] == split]
    if not selected:
        raise RuntimeError(f"No '{split}' clips in {manifest}")
    return selected


def _score_split(model: ViTAccidentClassifier, manifest: Path, data_dir: Path,
                 split: str, output_dir: Path, num_frames: int) -> list[dict[str, Any]]:
    records = []
    samples = _read_manifest(manifest, data_dir, split)
    for index, (path, label) in enumerate(samples, 1):
        record = model.score(path, num_frames)
        record["label"] = label
        records.append(record)
        if index % 20 == 0:
            print(f"{split}: scored {index}/{len(samples)} clips", flush=True)
    raw_path = output_dir / f"{split}_frame_scores.json"
    raw_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def _write_predictions(records: list[dict[str, Any]], rule: DecisionRule, path: Path) -> dict[str, Any]:
    labels, predictions = [], []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video", "actual", "p_accident", "predicted", "correct",
                         "peak_time_sec", "peak_frame_probability"])
        for record in records:
            score = aggregate(record["frame_probabilities"], rule.aggregation)
            label = int(record["label"])
            predicted = int(score >= rule.threshold)
            labels.append(label)
            predictions.append(predicted)
            writer.writerow([record["video"], label, round(score, 6), predicted,
                             int(predicted == label), record["peak_time_sec"],
                             record["peak_probability"]])
    return binary_metrics(labels, predictions)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(args.threads, torch.get_num_threads())))
    model = ViTAccidentClassifier(args.model, args.cache_dir, args.device)
    validation = _score_split(model, args.manifest, args.data_dir, "val", args.output_dir, args.frames)
    rule, val_metrics = select_rule(validation)
    test = _score_split(model, args.manifest, args.data_dir, "test", args.output_dir, args.frames)
    val_metrics = _write_predictions(validation, rule, args.output_dir / "val_predictions.csv")
    test_metrics = _write_predictions(test, rule, args.output_dir / "test_predictions.csv")
    report = {
        "model": args.model, "rule": asdict(rule), "frames_per_clip": args.frames,
        "model_revision": getattr(model.model.config, "_commit_hash", None),
        "validation": val_metrics, "test": test_metrics,
        "manifest": str(args.manifest), "data_dir": str(args.data_dir),
    }
    report_path = args.output_dir / "vit_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def _render_preview(video: Path, outcome: dict[str, Any], output: Path) -> None:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_width = min(width, 960)
    out_height = max(2, round(height * out_width / width))
    out_height += out_height % 2
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (out_width, out_height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot write preview: {output}")
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.resize(frame, (out_width, out_height))
            predicted = outcome["predicted_accident"]
            color = (0, 0, 255) if predicted else (0, 180, 0)
            label = "ACCIDENT" if predicted else "NORMAL"
            cv2.rectangle(frame, (0, 0), (out_width, 78), (0, 0, 0), -1)
            cv2.putText(frame, f"{label}  P={outcome['p_accident']:.2f}", (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2)
            cv2.putText(frame, f"Time {index / fps:.2f}s  Peak evidence {outcome['peak_time_sec']:.2f}s",
                        (15, 63), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
            writer.write(frame)
            index += 1
    finally:
        capture.release()
        writer.release()


def predict(args: argparse.Namespace) -> dict[str, Any]:
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    rule = DecisionRule(**calibration["rule"])
    torch.set_num_threads(max(1, min(args.threads, torch.get_num_threads())))
    model = ViTAccidentClassifier(calibration["model"], args.cache_dir, args.device)
    record = model.score(args.video, int(calibration["frames_per_clip"]))
    score = aggregate(record["frame_probabilities"], rule.aggregation)
    outcome = {
        **record, "p_accident": round(score, 6),
        "predicted_accident": score >= rule.threshold,
        "threshold": rule.threshold, "aggregation": rule.aggregation,
        "model": model.model_id,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"{args.video.stem}_result.json"
    result_path.write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    preview_path = args.output_dir / f"{args.video.stem}_preview.mp4"
    _render_preview(args.video, outcome, preview_path)
    print(json.dumps({"result": str(result_path), "preview": str(preview_path), **outcome}, indent=2))
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--cache-dir", type=Path, default=Path("outputs/model_cache"))
    common.add_argument("--device", default="cpu")
    common.add_argument("--threads", type=int, default=6)
    evaluate_parser = sub.add_parser("evaluate", parents=[common])
    evaluate_parser.add_argument("--manifest", type=Path, default=Path("data/benchmark_manifest.csv"))
    evaluate_parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    evaluate_parser.add_argument("--model", default=MODEL_ID)
    evaluate_parser.add_argument("--frames", type=int, default=8)
    evaluate_parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval/vit"))
    predict_parser = sub.add_parser("predict", parents=[common])
    predict_parser.add_argument("--video", type=Path, required=True)
    predict_parser.add_argument("--calibration", type=Path,
                                default=Path("outputs/eval/vit/vit_evaluation.json"))
    predict_parser.add_argument("--output-dir", type=Path, default=Path("outputs/vit_demo"))
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
