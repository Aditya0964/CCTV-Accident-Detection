"""Stage 1: CCTV ingestion, YOLO vehicle detection, ByteTrack, and track export."""

import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO class IDs: car=2, motorcycle=3, bus=5, truck=7.
VEHICLE_CLASSES = [2, 3, 5, 7]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input CCTV video")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained detection weights")
    parser.add_argument("--width", type=int, default=960, help="Resize width, preserving aspect ratio")
    parser.add_argument("--conf", type=float, default=0.15, help="Detection confidence threshold")
    parser.add_argument("--trail-length", type=int, default=30)
    return parser.parse_args()


def run(args):
    if not args.input.is_file():
        raise FileNotFoundError(f"Video not found: {args.input}")
    if args.width <= 0 or args.trail_length <= 0:
        raise ValueError("--width and --trail-length must be positive")

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise RuntimeError("Source video FPS unavailable; use a valid video with FPS metadata")

    input_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if input_width <= 0 or input_height <= 0:
        capture.release()
        raise RuntimeError("Invalid video dimensions")

    height = max(2, round(input_height * args.width / input_width))
    height += height % 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_video = args.output_dir / "tracked_vehicles.mp4"
    output_csv = args.output_dir / "tracks.csv"
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create output video: {output_video}")

    model = YOLO(args.model)
    trails = defaultdict(lambda: deque(maxlen=args.trail_length))
    frame_index = 0

    try:
        with output_csv.open("w", newline="", encoding="utf-8") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["frame", "timestamp_sec", "track_id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"])

            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                frame = cv2.resize(frame, (args.width, height))
                timestamp = frame_index / fps
                result = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=VEHICLE_CLASSES,
                    conf=args.conf,
                    verbose=False,
                )[0]

                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    coordinates = boxes.xyxy.cpu().numpy()
                    ids = boxes.id.int().cpu().tolist()
                    classes = boxes.cls.int().cpu().tolist()
                    confidences = boxes.conf.cpu().tolist()

                    for (x1, y1, x2, y2), track_id, class_id, confidence in zip(coordinates, ids, classes, confidences):
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        class_name = model.names[class_id]
                        csv_writer.writerow([frame_index, f"{timestamp:.3f}", track_id, class_id, class_name, f"{confidence:.4f}", f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}", f"{cx:.2f}", f"{cy:.2f}"])
                        trails[track_id].append((int(cx), int(cy)))

                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
                        cv2.putText(frame, f"{class_name} ID {track_id}", (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
                        points = list(trails[track_id])
                        for start, end in zip(points, points[1:]):
                            cv2.line(frame, start, end, (0, 255, 255), 2)

                cv2.putText(frame, f"Time {timestamp:.2f}s", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                writer.write(frame)
                frame_index += 1
                if frame_index % 100 == 0:
                    print(f"Processed {frame_index} frames")
    finally:
        capture.release()
        writer.release()

    print(f"Processed {frame_index} frames")
    print(f"Annotated video: {output_video}")
    print(f"Trajectory CSV: {output_csv}")


if __name__ == "__main__":
    run(parse_args())
