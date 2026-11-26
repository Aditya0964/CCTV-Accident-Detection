"""Run tracking, motion candidates, and calibrated ViT confirmation on one video."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def stage_commands(video: Path, calibration: Path, output_dir: Path, yolo_model: str) -> list[list[str]]:
    if not video.is_file():
        raise FileNotFoundError(video)
    if not calibration.is_file():
        raise FileNotFoundError(
            f"Calibrated ViT report not found: {calibration}. Run 'python -m src.vit_workflow evaluate' first."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        [sys.executable, "-m", "src.track_vehicles", "--input", str(video),
         "--output-dir", str(output_dir), "--model", yolo_model],
        [sys.executable, "-m", "src.detect_anomalies", "--tracks", str(output_dir / "tracks.csv"),
         "--video", str(video), "--preserve-moving-contacts", "--recover-single-vehicle",
         "--extract-clips", "--output-dir", str(output_dir)],
        [sys.executable, "-m", "src.vit_workflow", "predict", "--video", str(video),
         "--calibration", str(calibration), "--output-dir", str(output_dir / "vit")],
        [sys.executable, "-m", "src.confirm_accidents", "--candidates",
         str(output_dir / "candidates.json"), "--vit-calibration", str(calibration),
         "--localized-review", "--video", str(video), "--render-video", "--output-dir", str(output_dir)],
    ]


def run(video: Path, calibration: Path, output_dir: Path, yolo_model: str) -> dict:
    commands = stage_commands(video, calibration, output_dir, yolo_model)
    log_path = output_dir / "pipeline.log"
    with log_path.open("w", encoding="utf-8") as log:
        for command in commands:
            print("Running:", " ".join(command), flush=True)
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f"Pipeline stage failed (exit {result.returncode}). See {log_path}")

    clip_result = json.loads((output_dir / "vit" / f"{video.stem}_result.json").read_text())
    candidates = json.loads((output_dir / "candidates.json").read_text())
    incidents = json.loads((output_dir / "confirmed_incidents.json").read_text())
    summary = {
        "video": str(video),
        "video_level_accident": clip_result["predicted_accident"],
        "video_level_probability": clip_result["p_accident"],
        "approximate_peak_time_sec": clip_result["peak_time_sec"],
        "motion_candidates": len(candidates),
        "localized_confirmed_incidents": len(incidents),
        "incident_times_sec": [item["timestamp_sec"] for item in incidents],
        "clip_preview": str(output_dir / "vit" / f"{video.stem}_preview.mp4"),
        "confirmed_preview": str(output_dir / "confirmed_preview.mp4"),
        "log": str(log_path),
    }
    (output_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--calibration", type=Path,
                        default=Path("outputs/eval/vit/vit_evaluation.json"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--yolo-model", default="yolo11n.pt")
    args = parser.parse_args()
    output_dir = args.output_dir or Path("outputs/vit_pipeline") / args.video.stem
    run(args.video, args.calibration, output_dir, args.yolo_model)


if __name__ == "__main__":
    main()
