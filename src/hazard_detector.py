"""Scan post-incident footage with Firewatch's two-class YOLO11s model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.clip_buffer import compute_resized_dimensions
from src.risk_assessment import compute_visual_risk


CLASSES = ("fire", "smoke")
Box = tuple[float, float, float, float]


def load_firewatch_model(weights: Path):
    if weights.suffix.lower() != ".onnx" or not weights.is_file():
        raise ValueError(f"Expected a local Firewatch ONNX file: {weights}")
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install requirements-phase4.txt to use Firewatch") from exc
    from ultralytics import YOLO

    return YOLO(str(weights), task="detect")


class FirewatchDetector:
    def __init__(self, weights: Path, fire_conf: float, smoke_conf: float):
        self.model = load_firewatch_model(weights)
        self.thresholds = {"fire": fire_conf, "smoke": smoke_conf}

    def detect(self, frame) -> list[tuple[str, float, Box]]:
        result = self.model.predict(
            frame, imgsz=640, rect=False, device="cpu",
            conf=min(self.thresholds.values()), verbose=False,
        )[0]
        names = {i: name.strip().lower() for i, name in result.names.items()}
        if sorted(names.values()) != sorted(CLASSES):
            raise ValueError(f"Expected fire and smoke model classes, got {names}")

        detections = []
        for box, score, class_id in zip(
            result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()
        ):
            label = names[int(class_id)]
            if score >= self.thresholds[label]:
                detections.append((label, score, tuple(box)))
        return detections


@dataclass(frozen=True)
class ScanSettings:
    window_sec: float = 5.0
    sample_fps: float = 5.0
    min_persistence: float = 0.5
    min_iou: float = 0.02
    region_padding: float = 0.25
    tracking_width: int = 960


def box_iou(a: Box, b: Box) -> float:
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0.0


def scan_video(
    video: Path, incidents: list[dict], detector: FirewatchDetector, settings: ScanSettings,
    *, manual: bool = False,
) -> list[dict]:
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or min(frame_count, width, height) <= 0:
            raise ValueError(f"Video must provide valid FPS, frame count and dimensions: {video}")
        tracking_width, tracking_height = compute_resized_dimensions(width, height, settings.tracking_width)
        results = []
        for incident in incidents:
            raw_box = incident.get("candidate_region")
            full_frame = raw_box is None and (manual or incident.get("scan_full_frame") is True)
            source_pixels = manual or incident.get("region_space") == "source"
            scale = (1.0, 1.0) if source_pixels else (width / tracking_width, height / tracking_height)
            region = _scan_region(raw_box, scale, width, height, settings.region_padding)
            result = _scan_incident(
                capture, fps, frame_count, incident, detector, settings, region, full_frame,
            )
            if incident.get("confirmation_policy") in ("vit_impact_review", "fusion_localized_review"):
                result["accident_region_xyxy_source_pixels"] = _scan_region(raw_box, scale, width, height, 0.0)
            results.append(result)
        return results
    finally:
        capture.release()


def _scan_region(raw_box, scale, width, height, padding) -> Box | None:
    if not isinstance(raw_box, (tuple, list)) or len(raw_box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in raw_box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x1 >= x2 or y1 >= y2:
        return None
    sx, sy = scale
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    region = (max(0.0, (x1 - dx) * sx), max(0.0, (y1 - dy) * sy),
              min(width, (x2 + dx) * sx), min(height, (y2 + dy) * sy))
    return region if region[0] < region[2] and region[1] < region[3] else None


def _scan_incident(capture, fps, frame_count, incident, detector, settings, region, full_frame):
    timestamp = float(incident["timestamp_sec"])
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError(f"Invalid incident timestamp: {timestamp}")
    scan_start = incident.get("scan_start_sec") if full_frame else None
    scan_start = timestamp if scan_start is None else float(scan_start)
    if not math.isfinite(scan_start) or not 0 <= scan_start <= timestamp:
        raise ValueError(f"Invalid hazard scan start time: {scan_start}")
    end_sec = min(scan_start + settings.window_sec, frame_count / fps)
    hits = dict.fromkeys(CLASSES, 0)
    best = dict.fromkeys(CLASSES)
    sampled = 0
    requested = 0
    detections = []
    stride = max(1, round(fps / settings.sample_fps))
    if region is not None or full_frame:
        start = math.ceil(scan_start * fps - 1e-6)
        stop = math.ceil(end_sec * fps - 1e-6)
        for index in range(start, stop, stride):
            requested += 1
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                continue
            ok, frame = capture.read()
            if not ok:
                continue
            sampled += 1
            seen = set()
            for label, score, box in detector.detect(frame):
                if region is not None and box_iou(region, box) < settings.min_iou:
                    continue
                seen.add(label)
                detections.append({
                    "frame_index": index, "label": label, "confidence": round(score, 4),
                    "box_xyxy_source_pixels": [round(float(value), 2) for value in box],
                })
                if best[label] is None or score > best[label][0]:
                    best[label] = (score, index / fps)
            for label in seen:
                hits[label] += 1

    if region is None and not full_frame:
        status = "missing_or_invalid_region"
    elif sampled < 3 or sampled * 2 <= requested:
        status = "insufficient_frames"
    else:
        status = "scanned"
    flags = {
        f"{label}_detected": hits[label] / sampled >= settings.min_persistence if status == "scanned" else None
        for label in CLASSES
    }
    vehicles = incident.get("vehicles")
    visual_risk, risk_reason = compute_visual_risk(
        fire=flags["fire_detected"], smoke=flags["smoke_detected"],
        motion_score=incident.get("motion_score"),
        num_vehicles=len(vehicles) if isinstance(vehicles, list) else None,
    )
    return {
        **incident, **flags,
        "visual_risk": visual_risk,
        "visual_risk_reason": risk_reason,
        "hazard_scan": {
            "status": status,
            "scope": "full_frame" if full_frame else "incident_region",
            "region_xyxy_source_pixels": region,
            "sampled_frames": sampled,
            "requested_frames": requested,
            "window_start_sec": scan_start,
            "window_end_sec": max(scan_start, end_sec),
            "hits": hits,
            "sample_stride_frames": stride,
            "detections": detections,
            "strongest_detection": {
                label: {"confidence": round(value[0], 4), "timestamp_sec": round(value[1], 3)}
                if value is not None else None for label, value in best.items()
            },
        },
    }
