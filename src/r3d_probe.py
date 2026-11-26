"""Benchmark a frozen pretrained video encoder on source-disjoint CCTV clips."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision.models.video import R3D_18_Weights, r3d_18

from src.vit_probe import _write_predictions
from src.vit_workflow import _read_manifest, binary_metrics, sample_frames


def encode_video(path: Path, model: torch.nn.Module, frames: int) -> np.ndarray:
    images, _, _ = sample_frames(path, frames)
    rgb = np.stack([cv2.resize(np.asarray(image), (112, 112)) for image in images])
    x = torch.from_numpy(rgb).permute(3, 0, 1, 2).float().div_(255)
    mean = torch.tensor([0.43216, 0.394666, 0.37645])[:, None, None, None]
    std = torch.tensor([0.22803, 0.22145, 0.216989])[:, None, None, None]
    x = (x - mean) / std
    with torch.inference_mode():
        return model(x.unsqueeze(0)).squeeze(0).numpy().astype(np.float32)


def features(split: str, args: argparse.Namespace, model: torch.nn.Module):
    samples = _read_manifest(args.manifest, args.data_dir, split)
    cache = args.output_dir / "features"
    cache.mkdir(parents=True, exist_ok=True)
    vectors, labels, paths = [], [], []
    for index, (path, label) in enumerate(samples, 1):
        key = hashlib.sha256(f"r3d18:{args.frames}:{path}".encode()).hexdigest()[:20]
        target = cache / f"{key}.npy"
        if target.exists():
            vector = np.load(target)
        else:
            vector = encode_video(path, model, args.frames)
            np.save(target, vector)
        vectors.append(vector)
        labels.append(label)
        paths.append(path)
        if index % 25 == 0:
            print(f"{split}: video features {index}/{len(samples)}", flush=True)
    return np.stack(vectors), np.asarray(labels), paths


def fit(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    encoder = r3d_18(weights=R3D_18_Weights.DEFAULT)
    encoder.fc = torch.nn.Identity()
    encoder.eval()
    train_x, train_y, _ = features("train", args, encoder)
    val_x, val_y, val_paths = features("val", args, encoder)
    candidates = []
    for name, estimator in (
        ("logistic_C0.01", LogisticRegression(C=.01, max_iter=2000)),
        ("logistic_C0.1", LogisticRegression(C=.1, max_iter=2000)),
        ("logistic_C1", LogisticRegression(C=1, max_iter=2000)),
        ("rbf_C0.1", SVC(C=.1, probability=True)),
        ("rbf_C1", SVC(C=1, probability=True)),
        ("rbf_C10", SVC(C=10, probability=True)),
    ):
        classifier = make_pipeline(StandardScaler(), estimator)
        classifier.fit(train_x, train_y)
        scores = classifier.predict_proba(val_x)[:, 1]
        for threshold in np.arange(.1, .96, .05):
            threshold = round(float(threshold), 2)
            metrics = binary_metrics(val_y.tolist(), (scores >= threshold).astype(int).tolist())
            rank = (metrics["accuracy"], metrics["f1"], -abs(threshold - .5))
            candidates.append((rank, name, classifier, threshold, scores))
    _, name, classifier, threshold, val_scores = max(candidates, key=lambda row: row[0])
    test_x, test_y, test_paths = features("test", args, encoder)
    test_scores = classifier.predict_proba(test_x)[:, 1]
    val_metrics = _write_predictions(args.output_dir / "val_predictions.csv", val_paths,
                                     val_y, val_scores, threshold)
    test_metrics = _write_predictions(args.output_dir / "test_predictions.csv", test_paths,
                                      test_y, test_scores, threshold)
    model_path = args.output_dir / "probe.joblib"
    joblib.dump(classifier, model_path)
    report = {"base_model": "torchvision R3D-18 Kinetics-400", "classifier": name,
              "threshold": threshold, "frames_per_clip": args.frames,
              "train_clips": len(train_y), "validation": val_metrics,
              "test": test_metrics, "manifest": str(args.manifest),
              "model_path": str(model_path)}
    (args.output_dir / "probe_evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


def predict(args: argparse.Namespace) -> dict:
    report = json.loads(args.report.read_text())
    torch.set_num_threads(args.threads)
    encoder = r3d_18(weights=R3D_18_Weights.DEFAULT)
    encoder.fc = torch.nn.Identity()
    encoder.eval()
    vector = encode_video(args.video, encoder, report["frames_per_clip"])
    classifier = joblib.load(report["model_path"])
    score = float(classifier.predict_proba(vector.reshape(1, -1))[0, 1])
    result = {"video": str(args.video), "predicted_accident": score >= report["threshold"],
              "p_accident": round(score, 6), "threshold": report["threshold"]}
    target = args.output_dir / f"{args.video.stem}_result.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/benchmark_manifest.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--report", type=Path,
                        default=Path("outputs/eval/r3d_probe/probe_evaluation.json"))
    args = parser.parse_args()
    if args.video is not None:
        args.output_dir = args.output_dir or Path("outputs/r3d_demo")
        predict(args)
    else:
        args.output_dir = args.output_dir or Path("outputs/eval/r3d_probe")
        fit(args)


if __name__ == "__main__":
    main()
