"""Combine video embeddings and localized accident boxes using validation-only tuning."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _read_scores(split: str, vit_dir: Path, r3d_dir: Path, yolo_dir: Path) -> list[dict]:
    def index(path: Path) -> dict:
        with path.open(newline="") as handle:
            return {row["video"]: row for row in csv.DictReader(handle)}

    vit = index(vit_dir / f"{split}_predictions.csv")
    r3d = index(r3d_dir / f"{split}_predictions.csv")
    yolo = json.loads((yolo_dir / f"{split}_raw_scores.json").read_text())
    if set(vit) != set(r3d) or set(vit) != {row["video"] for row in yolo}:
        raise RuntimeError(f"The three {split} prediction sets do not contain the same clips")
    rows = []
    for row in yolo:
        path = row["video"]
        label = int(row["label"])
        if label != int(vit[path]["actual"]) or label != int(r3d[path]["actual"]):
            raise RuntimeError(f"Label mismatch for {path}")
        rows.append({"video": path, "actual": label,
                     "vit": float(vit[path]["p_accident"]),
                     "r3d": float(r3d[path]["p_accident"]),
                     "yolo": float(np.mean(row["frame_scores"]))})
    return rows


def _score(row: dict, weights: dict[str, float]) -> float:
    return sum(weights[name] * row[name] for name in ("vit", "r3d", "yolo"))


def _metrics(rows: list[dict], weights: dict[str, float], threshold: float) -> dict:
    labels = [row["actual"] for row in rows]
    predictions = [int(_score(row, weights) >= threshold) for row in rows]
    tp = sum(a == 1 and p == 1 for a, p in zip(labels, predictions))
    tn = sum(a == 0 and p == 0 for a, p in zip(labels, predictions))
    fp = sum(a == 0 and p == 1 for a, p in zip(labels, predictions))
    fn = sum(a == 1 and p == 0 for a, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": round((tp + tn) / len(rows), 4),
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * tp / (2 * tp + fp + fn), 4) if tp else 0.0,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "support_pos": sum(labels), "support_neg": len(rows) - sum(labels)}


def _select_rule(rows: list[dict]) -> tuple[dict[str, float], float, dict]:
    candidates = []
    for vit_tenths in range(1, 9):
        for r3d_tenths in range(1, 10 - vit_tenths):
            yolo_tenths = 10 - vit_tenths - r3d_tenths
            if yolo_tenths < 1:
                continue
            weights = {"vit": vit_tenths / 10, "r3d": r3d_tenths / 10,
                       "yolo": yolo_tenths / 10}
            for step in range(4, 37):
                threshold = round(step * .025, 3)
                metrics = _metrics(rows, weights, threshold)
                rank = (metrics["accuracy"], metrics["f1"], metrics["recall"],
                        -abs(threshold - .5))
                candidates.append((rank, weights, threshold, metrics))
    _, weights, threshold, metrics = max(candidates, key=lambda item: item[0])
    return weights, threshold, metrics


def _write_predictions(rows: list[dict], weights: dict[str, float], threshold: float,
                       path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video", "actual", "vit_score", "r3d_score", "yolo_mean_box_score",
                         "fusion_score", "predicted", "correct"])
        for row in rows:
            score = _score(row, weights)
            predicted = int(score >= threshold)
            writer.writerow([row["video"], row["actual"], row["vit"], row["r3d"],
                             row["yolo"], round(score, 6), predicted,
                             int(predicted == row["actual"])])


def evaluate(args: argparse.Namespace) -> dict:
    val = _read_scores("val", args.vit_dir, args.r3d_dir, args.yolo_dir)
    test = _read_scores("test", args.vit_dir, args.r3d_dir, args.yolo_dir)
    weights, threshold, val_metrics = _select_rule(val)
    test_metrics = _metrics(test, weights, threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_predictions(val, weights, threshold, args.output_dir / "val_predictions.csv")
    _write_predictions(test, weights, threshold, args.output_dir / "test_predictions.csv")
    report = {"model": "validation-tuned fusion of ViT, R3D-18, and YOLO11x",
              "weights": weights, "threshold": threshold, "validation": val_metrics,
              "test": test_metrics, "vit_report": str(args.vit_dir / "probe_evaluation.json"),
              "r3d_report": str(args.r3d_dir / "probe_evaluation.json"),
              "yolo_report": str(args.yolo_dir / "evaluation.json")}
    (args.output_dir / "fusion_evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


def predict(args: argparse.Namespace) -> dict:
    import cv2
    import joblib
    import torch
    from torchvision.models.video import R3D_18_Weights, r3d_18
    from ultralytics import YOLO

    from src.r3d_probe import encode_video
    from src.vit_workflow import ViTAccidentClassifier, sample_frames
    from src.yolo_accident_workflow import score_video

    report = json.loads(args.report.read_text())
    vit_report = json.loads(Path(report["vit_report"]).read_text())
    r3d_report = json.loads(Path(report["r3d_report"]).read_text())
    yolo_report = json.loads(Path(report["yolo_report"]).read_text())
    torch.set_num_threads(args.threads)

    vit = ViTAccidentClassifier(vit_report["base_model"], args.model_cache, "cpu")
    vit_vector = vit.features(args.video, vit_report["frames_per_clip"])
    vit_classifier = joblib.load(vit_report["model_path"])
    vit_score = float(vit_classifier.predict_proba(vit_vector.reshape(1, -1))[0, 1])

    r3d = r3d_18(weights=R3D_18_Weights.DEFAULT)
    r3d.fc = torch.nn.Identity()
    r3d.eval()
    r3d_vector = encode_video(args.video, r3d, r3d_report["frames_per_clip"])
    r3d_classifier = joblib.load(r3d_report["model_path"])
    r3d_score = float(r3d_classifier.predict_proba(r3d_vector.reshape(1, -1))[0, 1])

    yolo = YOLO(yolo_report["weights"])
    detection = score_video(yolo, args.video, yolo_report["frames_per_clip"],
                            yolo_report["imgsz"])
    yolo_score = float(np.mean(detection["frame_scores"]))
    scores = {"vit": vit_score, "r3d": r3d_score, "yolo": yolo_score}
    fused = _score(scores, report["weights"])
    predicted = fused >= report["threshold"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images, _, _ = sample_frames(args.video, yolo_report["frames_per_clip"])
    tiles = []
    for index, image in enumerate(images):
        frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR).copy()
        for box in detection["frame_boxes"][index]:
            x1, y1, x2, y2 = [int(value) for value in box["xyxy"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"ACCIDENT {box['confidence']:.2f}",
                        (x1, max(y1 - 8, 22)), cv2.FONT_HERSHEY_SIMPLEX,
                        .65, (0, 0, 255), 2)
        h = round(frame.shape[0] * 640 / frame.shape[1])
        tile = cv2.resize(frame, (640, h))
        canvas = np.zeros((h + 45, 640, 3), dtype=np.uint8)
        canvas[45:] = tile
        caption = f"{detection['times_sec'][index]:.2f}s  detector score {detection['frame_scores'][index]:.2f}"
        cv2.putText(canvas, caption, (10, 29), cv2.FONT_HERSHEY_SIMPLEX,
                    .6, (255, 255, 255), 2)
        tiles.append(canvas)
    evidence = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]) if len(tiles) == 4 else np.vstack(tiles)
    evidence_path = args.output_dir / f"{args.video.stem}_evidence.jpg"
    cv2.imwrite(str(evidence_path), evidence)
    peak_index = int(np.argmax(detection["frame_scores"]))
    peak_box = max(detection["frame_boxes"][peak_index], key=lambda box: box["confidence"], default=None)
    result = {"video": str(args.video), "predicted_accident": predicted,
              "fusion_score": round(fused, 6), "threshold": report["threshold"],
              "component_scores": {name: round(value, 6) for name, value in scores.items()},
              "peak_detector_time_sec": detection["times_sec"][peak_index],
              "peak_detector_box_xyxy": peak_box["xyxy"] if peak_box else None,
              "peak_detector_box_score": peak_box["confidence"] if peak_box else None,
              "evidence_image": str(evidence_path)}
    (args.output_dir / f"{args.video.stem}_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit-dir", type=Path, default=Path("outputs/eval/vit_probe"))
    parser.add_argument("--r3d-dir", type=Path, default=Path("outputs/eval/r3d_probe"))
    parser.add_argument("--yolo-dir", type=Path, default=Path("outputs/eval/enos_yolo11x"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--report", type=Path,
                        default=Path("outputs/eval/fusion/fusion_evaluation.json"))
    parser.add_argument("--model-cache", type=Path, default=Path("outputs/model_cache"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.video:
        args.output_dir = args.output_dir or Path("outputs/fusion_demo")
        predict(args)
    else:
        args.output_dir = args.output_dir or Path("outputs/eval/fusion")
        evaluate(args)


if __name__ == "__main__":
    main()
