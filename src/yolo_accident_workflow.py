"""Benchmark the Enos YOLO11x accident detector on labeled CCTV videos."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.vit_workflow import _read_manifest, binary_metrics, sample_frames


def score_video(model: YOLO, path: Path, frames: int, imgsz: int) -> dict:
    images, times, _ = sample_frames(path, frames)
    bgr = [cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR) for image in images]
    results = model.predict(bgr, imgsz=imgsz, conf=.01, device="cpu", verbose=False)
    frame_scores = []
    frame_boxes = []
    for result in results:
        boxes = []
        for box in result.boxes:
            if "accident" not in model.names[int(box.cls[0])].lower():
                continue
            confidence = float(box.conf[0])
            boxes.append({"xyxy": [round(float(x), 1) for x in box.xyxy[0]],
                          "confidence": round(confidence, 6)})
        frame_scores.append(max((item["confidence"] for item in boxes), default=0.0))
        frame_boxes.append(boxes)
    return {"video": str(path), "times_sec": times,
            "frame_scores": frame_scores, "frame_boxes": frame_boxes}


def aggregate(record: dict, method: str) -> float:
    values = sorted(record["frame_scores"], reverse=True)
    if method == "max":
        return values[0]
    if method == "top2":
        return float(np.mean(values[:2]))
    if method == "mean":
        return float(np.mean(values))
    raise ValueError(method)


def score_split(model: YOLO, args: argparse.Namespace, split: str) -> list[dict]:
    raw_path = args.output_dir / f"{split}_raw_scores.json"
    expected = _read_manifest(args.manifest, args.data_dir, split)
    records = []
    if raw_path.exists():
        records = json.loads(raw_path.read_text())
        if not all(
                row["video"] == str(path) and row["label"] == label
                for row, (path, label) in zip(records, expected)):
            records = []
    for index, (path, label) in enumerate(expected[len(records):], len(records) + 1):
        row = score_video(model, path, args.frames, args.imgsz)
        row["label"] = label
        records.append(row)
        if index % 10 == 0:
            raw_path.write_text(json.dumps(records))
            print(f"{split}: YOLO scored {index} clips", flush=True)
    raw_path.write_text(json.dumps(records))
    return records


def select_rule(records: list[dict]) -> tuple[str, float, dict]:
    labels = [row["label"] for row in records]
    options = []
    for method in ("max", "top2", "mean"):
        scores = [aggregate(row, method) for row in records]
        for threshold in (.01, .025, .05, .075, .1, .15, .2, .25, .3, .4, .5, .6, .7, .8):
            metrics = binary_metrics(labels, [int(score >= threshold) for score in scores])
            options.append(((metrics["accuracy"], metrics["f1"], metrics["recall"],
                             -abs(threshold - .2)), method, threshold, metrics))
    _, method, threshold, metrics = max(options, key=lambda row: row[0])
    return method, threshold, metrics


def write_predictions(records: list[dict], method: str, threshold: float,
                      path: Path) -> dict:
    labels, predictions = [], []
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video", "actual", "score", "predicted", "correct",
                         "peak_time_sec", "peak_box"])
        for row in records:
            score = aggregate(row, method)
            predicted = int(score >= threshold)
            labels.append(row["label"])
            predictions.append(predicted)
            peak = int(np.argmax(row["frame_scores"]))
            boxes = row["frame_boxes"][peak]
            writer.writerow([row["video"], row["label"], round(score, 6), predicted,
                             int(predicted == row["label"]), row["times_sec"][peak],
                             boxes[0]["xyxy"] if boxes else ""])
    return binary_metrics(labels, predictions)


def evaluate(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    print("Model classes:", model.names, flush=True)
    val = score_split(model, args, "val")
    method, threshold, _ = select_rule(val)
    test = score_split(model, args, "test")
    val_metrics = write_predictions(val, method, threshold,
                                    args.output_dir / "val_predictions.csv")
    test_metrics = write_predictions(test, method, threshold,
                                     args.output_dir / "test_predictions.csv")
    report = {"model": "Enos-123/traffic-accident-detection-yolo11x",
              "weights": str(args.weights), "frames_per_clip": args.frames,
              "imgsz": args.imgsz, "aggregation": method, "threshold": threshold,
              "validation": val_metrics, "test": test_metrics,
              "manifest": str(args.manifest)}
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


def predict(args: argparse.Namespace) -> dict:
    calibration = json.loads(args.calibration.read_text())
    model = YOLO(str(args.weights))
    row = score_video(model, args.video, calibration["frames_per_clip"], calibration["imgsz"])
    score = aggregate(row, calibration["aggregation"])
    predicted = score >= calibration["threshold"]
    peak = int(np.argmax(row["frame_scores"]))
    out = args.output_dir or Path("outputs/yolo_demo")
    out.mkdir(parents=True, exist_ok=True)

    images, _, _ = sample_frames(args.video, calibration["frames_per_clip"])
    annotated = []
    for index, image in enumerate(images):
        frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR).copy()
        for box in row["frame_boxes"][index]:
            x1, y1, x2, y2 = [int(value) for value in box["xyxy"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"ACCIDENT {box['confidence']:.2f}",
                        (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        .65, (0, 0, 255), 2)
        width = 640
        height = round(frame.shape[0] * width / frame.shape[1])
        tile = cv2.resize(frame, (width, height))
        canvas = np.zeros((height + 48, width, 3), dtype=np.uint8)
        canvas[48:] = tile
        caption = f"Sample {index + 1}: {row['times_sec'][index]:.2f}s | accident box {row['frame_scores'][index]:.2f}"
        cv2.putText(canvas, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    .57, (255, 255, 255), 2)
        annotated.append(canvas)
    evidence_path = out / f"{args.video.stem}_evidence.jpg"
    if len(annotated) == 4:
        contact_sheet = np.vstack([np.hstack(annotated[:2]), np.hstack(annotated[2:])])
    else:
        contact_sheet = np.vstack(annotated)
    cv2.imwrite(str(evidence_path), contact_sheet)

    result = {"video": str(args.video), "predicted_accident": predicted,
              "decision_score": round(score, 6), "threshold": calibration["threshold"],
              "aggregation": calibration["aggregation"],
              "peak_time_sec": row["times_sec"][peak],
              "frame_scores": row["frame_scores"],
              "sample_times_sec": row["times_sec"],
              "evidence_image": str(evidence_path)}
    result_path = out / f"{args.video.stem}_result.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=Path("outputs/model_cache/enos_yolo11x/weights/epoch61.pt"))
    parser.add_argument("--manifest", type=Path, default=Path("data/benchmark_manifest.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--video", type=Path, default=None,
                        help="Predict one video using a completed validation report")
    parser.add_argument("--calibration", type=Path,
                        default=Path("outputs/eval/enos_yolo11x/evaluation.json"))
    args = parser.parse_args()
    if args.video is not None:
        predict(args)
    else:
        args.output_dir = args.output_dir or Path("outputs/eval/enos_yolo11x")
        evaluate(args)


if __name__ == "__main__":
    main()
