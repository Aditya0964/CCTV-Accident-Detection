"""Render a post-scan incident summary on the source video."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2


def render_hazard_preview(
    video: Path, incidents: list[dict], output: Path, *, show_summary: bool = True,
    accident_only: bool = False, show_hazards: bool = False,
) -> None:
    """Mark each incident's scan region and optionally show its summary."""
    ordered = sorted(incidents, key=lambda item: item["hazard_scan"]["window_start_sec"])
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or min(width, height) < 2:
            raise ValueError(f"Video must provide valid FPS and dimensions: {video}")
        output.parent.mkdir(parents=True, exist_ok=True)
        # Pad odd dimensions rather than cropping away source pixels.
        pad_x, pad_y = width % 2, height % 2
        accidents = [
            (row["timestamp_sec"], row["accident_region_xyxy_source_pixels"])
            for row in incidents if row.get("accident_region_xyxy_source_pixels") is not None
        ]
        hazards_by_frame = _hazard_overlays(incidents, fps) if show_hazards else {}
        with TemporaryDirectory(dir=output.parent) as folder:
            temporary = Path(folder) / "preview.mp4"
            writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (width + pad_x, height + pad_y))
            frame_index = 0
            incident_index = -1
            try:
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create MP4 preview: {output}")
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    timestamp = frame_index / fps
                    if accident_only:
                        for peak, box in accidents:
                            if peak - 0.15 <= timestamp <= peak + 1.5:
                                _draw_accident(frame, box)
                    else:
                        while (incident_index + 1 < len(ordered)
                               and ordered[incident_index + 1]["hazard_scan"]["window_start_sec"] <= timestamp + 1e-6):
                            incident_index += 1
                        if incident_index >= 0:
                            incident = ordered[incident_index]
                            if timestamp < incident["hazard_scan"]["window_end_sec"]:
                                _draw_region(frame, incident["hazard_scan"]["region_xyxy_source_pixels"])
                                if show_summary:
                                    _draw_summary(frame, incident)
                    for (label, box), persistent in hazards_by_frame.get(frame_index, {}).items():
                        color = (0, 135, 255) if label == "fire" else (210, 173, 85)
                        text = label.upper() if persistent else f"{label.upper()} CUE"
                        _draw_labeled_box(frame, box, text, color)
                    if pad_x or pad_y:
                        frame = cv2.copyMakeBorder(frame, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT)
                    writer.write(frame)
                    frame_index += 1
            finally:
                writer.release()
            if not frame_index:
                raise ValueError(f"No video frames could be decoded: {video}")
            temporary.replace(output)
    finally:
        capture.release()


def _draw_region(frame, region) -> None:
    if region is not None:
        x1, y1, x2, y2 = (round(x) for x in region)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 205, 82), 2)


def _hazard_overlays(incidents: list[dict], fps: float) -> dict:
    by_frame = defaultdict(dict)
    for incident in incidents:
        scan = incident["hazard_scan"]
        stride = int(scan.get("sample_stride_frames", 1))
        first = math.ceil(scan["window_start_sec"] * fps - 1e-6)
        stop = math.ceil(scan["window_end_sec"] * fps - 1e-6)
        for item in scan.get("detections", []):
            index = item["frame_index"]
            label = item["label"]
            key = label, tuple(item["box_xyxy_source_pixels"])
            persistent = incident.get(f"{label}_detected") is True
            # Show sampled boxes briefly and never carry them across an empty sample.
            start = max(first, index - (stride - 1) // 2)
            end = min(stop, index + stride // 2 + 1)
            for frame in range(start, end):
                by_frame[frame][key] = by_frame[frame].get(key, False) or persistent
    return by_frame


def _draw_accident(frame, region) -> None:
    _draw_labeled_box(frame, region, "ACCIDENT", (30, 30, 240))


def _draw_labeled_box(frame, region, label: str, color: tuple[int, int, int]) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (min(limit - 1, max(0, round(value))) for value, limit in
                      zip(region, (width, height, width, height)))
    if x1 >= x2 or y1 >= y2:
        return
    scale = max(0.35, min(0.85, width / 1150))
    line = max(1, round(scale * 2))
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, line)
    label_x = max(0, min(x1, width - text_width - 10))
    label_y = max(text_height + 4, min(height - baseline - 4, y1 - baseline - 4))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, line + 1)
    cv2.rectangle(frame, (label_x, label_y - text_height - 4),
                  (label_x + text_width + 8, label_y + baseline + 2), color, cv2.FILLED)
    cv2.putText(frame, label, (label_x + 4, label_y), font, scale, (255, 255, 255), line, cv2.LINE_AA)


def _draw_summary(frame, incident: dict) -> None:
    height, width = frame.shape[:2]
    scan = incident["hazard_scan"]
    incident_id = incident.get("incident_id", "Unknown ID")
    source = "Manual incident" if incident.get("incident_source") == "manual" else "Incident"
    if incident.get("candidate_source") == "fallback_window":
        source = "Fallback window"
    white = (242, 247, 252)
    lines = [(f"{source} {incident_id} | {scan['window_start_sec']:.2f}s", white)]
    for label in ("smoke", "fire"):
        flag = incident.get(f"{label}_detected")
        if flag is None:
            lines.append((f"{label.title()}: Scan incomplete", white))
        else:
            status = "Detected" if flag else "Below threshold"
            count = f"{scan['hits'][label]}/{scan['sampled_frames']} frames"
            color = (75, 210, 255) if flag else white
            lines.append((f"{label.title()}: {status} ({count})", color))
    if incident.get("visual_risk"):
        lines.append((f"Visual risk: {incident['visual_risk']}", white))
    incident_metrics = []
    if incident.get("accident_confidence") is not None:
        incident_metrics.append(f"Accident confidence: {incident['accident_confidence']:.0%}")
    if incident.get("motion_score") is not None:
        incident_metrics.append(f"Motion score: {incident['motion_score']}/9")
    if incident_metrics:
        lines.append((" | ".join(incident_metrics), white))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = min(1.0, width / 900)
    thickness = 2 if font_size >= 0.65 else 1
    margin = max(4, round(font_size * 15))
    measurements = [cv2.getTextSize(line, font, font_size, thickness)[0] for line, _ in lines]
    line_height = max(h for _, h in measurements) + margin
    panel = (min(width - 1, max(w for w, _ in measurements) + margin * 3),
             min(height - 1, margin * 2 + line_height * len(lines)))
    cv2.rectangle(frame, (margin, margin), panel, (24, 21, 17), cv2.FILLED)
    for index, (line, color) in enumerate(lines, start=1):
        cv2.putText(frame, line, (margin * 2, margin + line_height * index), font,
                    font_size, color, thickness, cv2.LINE_AA)
