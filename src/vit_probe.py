"""Train a small source-disjoint classifier on frozen pretrained ViT video features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.vit_workflow import MODEL_ID, ViTAccidentClassifier, _read_manifest, binary_metrics


def _features_for_split(model: ViTAccidentClassifier, manifest: Path, data_dir: Path,
                        split: str, cache_dir: Path, frames: int
                        ) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    samples = _read_manifest(manifest, data_dir, split)
    cache_dir.mkdir(parents=True, exist_ok=True)
    vectors, labels, paths = [], [], []
    for index, (path, label) in enumerate(samples, 1):
        key = hashlib.sha256(f"{model.model_id}:{frames}:{path}".encode()).hexdigest()[:20]
        feature_path = cache_dir / f"{key}.npy"
        if feature_path.is_file():
            vector = np.load(feature_path)
        else:
            vector = model.features(path, frames)
            np.save(feature_path, vector)
        vectors.append(vector)
        labels.append(label)
        paths.append(path)
        if index % 25 == 0:
            print(f"{split}: ViT features {index}/{len(samples)}", flush=True)
    return np.stack(vectors), np.asarray(labels, dtype=np.int64), paths


def _write_predictions(path: Path, videos: list[Path], labels: np.ndarray,
                       scores: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(np.int64)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video", "actual", "p_accident", "predicted", "correct"])
        for video, label, score, prediction in zip(videos, labels, scores, predictions):
            writer.writerow([str(video), int(label), round(float(score), 6),
                             int(prediction), int(label == prediction)])
    return binary_metrics(labels.tolist(), predictions.tolist())


def fit(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(args.threads, torch.get_num_threads())))
    vit = ViTAccidentClassifier(args.model, args.model_cache, args.device)
    feature_cache = args.output_dir / "features"
    train_x, train_y, _ = _features_for_split(
        vit, args.manifest, args.data_dir, "train", feature_cache, args.frames)
    val_x, val_y, val_paths = _features_for_split(
        vit, args.manifest, args.data_dir, "val", feature_cache, args.frames)
    if len(set(train_y)) != 2 or len(set(val_y)) != 2:
        raise RuntimeError("Train and validation splits must each contain both classes")

    candidates = []
    for c in (0.001, 0.01, 0.1, 1.0):
        classifier = make_pipeline(
            StandardScaler(), LogisticRegression(C=c, max_iter=1000, solver="liblinear")
        )
        classifier.fit(train_x, train_y)
        scores = classifier.predict_proba(val_x)[:, 1]
        for step in range(1, 20):
            threshold = round(step * 0.05, 2)
            metrics = binary_metrics(val_y.tolist(), (scores >= threshold).astype(int).tolist())
            rank = (metrics["accuracy"], metrics["f1"], metrics["recall"],
                    -abs(threshold - 0.5), -c)
            candidates.append((rank, classifier, c, threshold, scores, metrics))
    _, classifier, c, threshold, val_scores, val_metrics = max(candidates, key=lambda item: item[0])

    test_x, test_y, test_paths = _features_for_split(
        vit, args.manifest, args.data_dir, "test", feature_cache, args.frames)
    test_scores = classifier.predict_proba(test_x)[:, 1]
    val_metrics = _write_predictions(args.output_dir / "val_predictions.csv",
                                     val_paths, val_y, val_scores, threshold)
    test_metrics = _write_predictions(args.output_dir / "test_predictions.csv",
                                      test_paths, test_y, test_scores, threshold)
    model_path = args.output_dir / "probe.joblib"
    joblib.dump(classifier, model_path)
    report = {
        "base_model": args.model,
        "base_model_revision": getattr(vit.model.config, "_commit_hash", None),
        "classifier": "standardized L2 logistic regression on mean and standard deviation of ViT class tokens",
        "C": c, "threshold": threshold, "frames_per_clip": args.frames,
        "train_clips": len(train_y), "validation": val_metrics, "test": test_metrics,
        "manifest": str(args.manifest), "model_path": str(model_path),
    }
    (args.output_dir / "probe_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/benchmark_manifest.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-cache", type=Path, default=Path("outputs/model_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval/vit_probe"))
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    fit(parser.parse_args())


if __name__ == "__main__":
    main()
