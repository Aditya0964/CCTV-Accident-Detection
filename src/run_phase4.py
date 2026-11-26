"""Scan the ViT or fusion workflow's incident evidence for fire and smoke."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from src.hazard_detector import FirewatchDetector, ScanSettings, scan_video
from src.hazard_preview import render_hazard_preview


def incident_evidence(clip: dict, localized: list[dict]) -> list[dict]:
    """Prefer localized incidents and retain a scene-wide scan for clip-only alerts."""
    if localized or not clip["predicted_accident"]:
        return localized
    fusion = "fusion_score" in clip
    box_score = clip.get("peak_detector_box_score")
    box = clip.get("peak_detector_box_xyxy")
    use_box = fusion and isinstance(box_score, (int, float)) and box_score >= 0.25 and box is not None
    if fusion:
        peak = clip.get("peak_detector_time_sec") if use_box else None
    else:
        peak = clip.get("peak_time_sec")
    return [{
        "incident_id": "clip_001",
        "timestamp_sec": 0.0 if peak is None else peak,
        "timestamp_is_approximate": True,
        "timestamp_source": "detector_sample" if use_box else "vit_sample" if peak is not None else "clip_start",
        "incident_source": "clip_verdict",
        "candidate_region": box if use_box else None,
        "region_space": "source",
        "scan_full_frame": not use_box,
        "scan_start_sec": max(0.0, peak - 2.0) if not fusion and peak is not None else None,
        "confirmation_policy": "fusion_localized_review" if use_box and box_score >= 0.80 else None,
        "clip_score": clip["fusion_score"] if fusion else clip["p_accident"],
    }]


def run(video: Path, clip: dict, localized: list[dict], weights: Path, output: Path) -> dict:
    if type(clip.get("predicted_accident")) is not bool:
        raise ValueError("Expected a Phase 3 clip verdict with predicted_accident")
    if not isinstance(localized, list) or any(
        not isinstance(row, dict) or "timestamp_sec" not in row for row in localized
    ):
        raise ValueError("Expected a Phase 3 incident list with timestamp_sec")
    incidents = incident_evidence(clip, localized)
    for item in incidents:
        timestamp = item["timestamp_sec"]
        if not isinstance(timestamp, (float, int)) or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Expected a finite, nonnegative incident timestamp")

    preview = output / "final_preview.mp4"
    if preview.resolve() == video.resolve():
        raise ValueError("The preview path must differ from the source video")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    if incidents:
        detector = FirewatchDetector(weights, fire_conf=0.40, smoke_conf=0.50)
        results = scan_video(video, incidents, detector, ScanSettings())
    render_hazard_preview(video, results, preview, show_summary=False, accident_only=True, show_hazards=True)
    (output / "hazard_incidents.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    columns = ("incident_id", "timestamp_sec", "timestamp_is_approximate", "timestamp_source",
               "scope", "status", "fire_detected", "smoke_detected", "visual_risk", "visual_risk_reason")
    with (output / "hazard_incidents.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow({**result, **result["hazard_scan"]})

    fusion = "fusion_score" in clip
    summary = {
        "backend": "fusion" if fusion else "vit",
        "video": str(video),
        "predicted_accident": clip["predicted_accident"],
        "score": clip["fusion_score"] if fusion else clip["p_accident"],
        "threshold": clip["threshold"],
        "localized_incidents": len(localized),
        "scan_count": len(results),
    }
    (output / "review_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Firewatch scanned {sum(row['hazard_scan']['status'] == 'scanned' for row in results)} of "
          f"{len(results)} incident windows. Review saved to {output}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--clip-result", type=Path, required=True)
    parser.add_argument("--incidents", type=Path, help="Optional localized ViT incident JSON")
    parser.add_argument("--weights", type=Path, default=Path("outputs/models/firewatch-yolo11s.onnx"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase4"))
    args = parser.parse_args()
    try:
        clip = json.loads(args.clip_result.read_text(encoding="utf-8-sig"))
        incidents = json.loads(args.incidents.read_text(encoding="utf-8-sig")) if args.incidents else []
        if not isinstance(clip, dict):
            raise ValueError("Expected a Phase 3 clip verdict object")
        run(args.video, clip, incidents, args.weights, args.output_dir)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Phase 4 failed: {exc}\n")


if __name__ == "__main__":
    main()
