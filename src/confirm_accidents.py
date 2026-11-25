"""Stage 11 & 12: Video confirmation of motion candidates and cooldown incident management."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import cv2
import torch

from src.clip_buffer import compute_resized_dimensions
from src.event_manager import AccidentEventManager, ConfirmedIncident, write_confirmed_incidents_json
from src.scad_dataset import sample_video_frames
from src.video_classifier import load_checkpoint, resolve_device


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Candidates JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    return data


def _score_candidate_clip(
    candidate: dict[str, Any],
    model: torch.nn.Module,
    num_frames: int,
    input_size: int,
    device: torch.device,
    fallback_video: Optional[Path] = None,
) -> float:
    raw_clip = candidate.get("clip_path")
    clip_path = Path(raw_clip) if raw_clip else None

    if clip_path is not None and clip_path.is_file():
        tensor = sample_video_frames(
            clip_path, num_frames=num_frames, input_size=input_size, augment=False
        )
    elif fallback_video is not None and fallback_video.is_file():
        tensor = sample_video_frames(
            fallback_video,
            num_frames=num_frames,
            input_size=input_size,
            augment=False,
            start_sec=float(candidate.get("clip_start_sec", 0.0)),
            end_sec=float(candidate.get("clip_end_sec", 0.0)),
        )
    else:
        raise FileNotFoundError(
            f"Candidate #{candidate.get('candidate_id')} has no clip_path and no valid --video fallback"
        )

    batch = tensor.unsqueeze(0).to(device)
    prob = float(model.predict_proba(batch).item())
    return prob


def evaluate_experiment3(
    confirmation_results: list[dict[str, Any]],
    confirmed_incidents: list[ConfirmedIncident],
    ground_truth_path: Path,
    match_tolerance_sec: float = 3.0,
) -> dict[str, Any]:
    """Compare Experiment 3: motion rules only vs. motion rules -> video classifier."""
    if not ground_truth_path.is_file():
        raise FileNotFoundError(f"Ground truth JSON not found: {ground_truth_path}")
    gt_items = json.loads(ground_truth_path.read_text(encoding="utf-8"))

    gt_windows: list[tuple[float, float]] = []
    for item in gt_items:
        if "start_sec" in item and "end_sec" in item:
            gt_windows.append((float(item["start_sec"]) - match_tolerance_sec, float(item["end_sec"]) + match_tolerance_sec))
        elif "timestamp_sec" in item:
            t = float(item["timestamp_sec"])
            gt_windows.append((t - match_tolerance_sec, t + match_tolerance_sec))

    def _score_events(timestamps: list[float]) -> dict[str, Any]:
        matched_gt: set[int] = set()
        tp = 0
        fp = 0
        for ts in timestamps:
            hit_idx = next(
                (idx for idx, (s, e) in enumerate(gt_windows) if s <= ts <= e and idx not in matched_gt),
                None,
            )
            if hit_idx is not None:
                matched_gt.add(hit_idx)
                tp += 1
            else:
                # Multiple triggers inside an already matched GT window count as duplicate FP.
                fp += 1
        fn = len(gt_windows) - len(matched_gt)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return {
            "total_alerts": len(timestamps),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    motion_only_ts = [float(r["timestamp_sec"]) for r in confirmation_results]
    confirmed_ts = [float(inc.timestamp_sec) for inc in confirmed_incidents]
    m_stats = _score_events(motion_only_ts)
    c_stats = _score_events(confirmed_ts)
    fp_reduction = (
        (m_stats["fp"] - c_stats["fp"]) / m_stats["fp"] if m_stats["fp"] > 0 else 0.0
    )

    return {
        "ground_truth_count": len(gt_windows),
        "match_tolerance_sec": match_tolerance_sec,
        "motion_rules_only": m_stats,
        "motion_rules_plus_classifier": c_stats,
        "false_positive_reduction_rate": round(fp_reduction, 4),
    }


def render_confirmed_preview(
    video_path: Path,
    confirmation_results: list[dict[str, Any]],
    confirmed_incidents: list[ConfirmedIncident],
    output_path: Path,
    target_width: int = 960,
) -> Path:
    """Render video preview showing confirmed incidents vs. rejected motion candidates."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or in_w <= 0 or in_h <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata in {video_path}")

    out_w, out_h = compute_resized_dimensions(in_w, in_h, target_width)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create confirmed preview video: {output_path}")

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            canvas = cv2.resize(frame, (out_w, out_h))
            ts = frame_index / fps

            active_inc = [
                inc for inc in confirmed_incidents if inc.timestamp_sec - 0.5 <= ts <= inc.timestamp_sec + 2.0
            ]
            active_rej = [
                res
                for res in confirmation_results
                if not res["confirmed"] and res["timestamp_sec"] - 0.5 <= ts <= res["timestamp_sec"] + 1.5
            ]

            for rej in active_rej:
                rx1, ry1, rx2, ry2 = (int(round(v)) for v in rej["candidate_region"])
                cv2.rectangle(canvas, (max(0, rx1), max(0, ry1)), (min(out_w - 1, rx2), min(out_h - 1, ry2)), (150, 150, 150), 2)
                cv2.putText(
                    canvas,
                    f"REJECTED #{rej['candidate_id']} P={rej['p_accident']:.2f}",
                    (max(0, rx1), max(20, ry1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (180, 180, 180),
                    2,
                )

            for inc in active_inc:
                rx1, ry1, rx2, ry2 = (int(round(v)) for v in inc.candidate_region)
                cv2.rectangle(canvas, (max(0, rx1), max(0, ry1)), (min(out_w - 1, rx2), min(out_h - 1, ry2)), (0, 0, 255), 3)
                veh_desc = " + ".join(f"{v['class_name']} ID {v['track_id']}" for v in inc.vehicles)
                cv2.putText(
                    canvas,
                    f"INCIDENT {inc.incident_id} | P={inc.accident_confidence:.2f} | {veh_desc}",
                    (max(0, rx1), max(24, ry1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 0, 255),
                    2,
                )

            banner_color = (0, 0, 255) if active_inc else (255, 255, 255)
            banner = f"Time {ts:.2f}s"
            if active_inc:
                banner += f" | ACCIDENT CONFIRMED (ID {active_inc[0].incident_id}, P={active_inc[0].accident_confidence:.2f})"
            cv2.putText(canvas, banner, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner_color, 2)

            writer.write(canvas)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("outputs/candidates.json"))
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained classifier checkpoint (.pt)")
    parser.add_argument("--threshold", type=float, default=None, help="Override P(accident) confirmation cutoff T_c")
    parser.add_argument("--cooldown-sec", type=float, default=5.0)
    parser.add_argument("--region-iou-threshold", type=float, default=0.10)
    parser.add_argument("--video", type=Path, default=None, help="Source video for clip slicing fallback or --render-video")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--ground-truth", type=Path, default=None, help="Optional ground-truth JSON for Experiment 3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.render_video and args.video is None:
        raise ValueError("--video is required when --render-video is set")

    candidates = _load_candidates(args.candidates)
    device = resolve_device(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device=device)
    cutoff = args.threshold if args.threshold is not None else cfg.threshold

    manager = AccidentEventManager(
        cooldown_sec=args.cooldown_sec,
        region_iou_threshold=args.region_iou_threshold,
    )

    confirmation_results: list[dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda c: float(c["timestamp_sec"])):
        p_acc = _score_candidate_clip(
            cand,
            model=model,
            num_frames=cfg.num_frames,
            input_size=cfg.input_size,
            device=device,
            fallback_video=args.video,
        )
        confirmed = p_acc > cutoff
        incident_id: Optional[str] = None
        is_new_incident = False
        if confirmed:
            inc, is_new_incident = manager.register_confirmed(cand, accident_confidence=p_acc)
            incident_id = inc.incident_id

        confirmation_results.append(
            {
                "candidate_id": cand.get("candidate_id"),
                "timestamp_sec": cand.get("timestamp_sec"),
                "frame_index": cand.get("frame_index"),
                "involved_track_ids": cand.get("involved_track_ids", []),
                "candidate_region": cand.get("candidate_region", []),
                "motion_score": cand.get("motion_score"),
                "p_accident": round(p_acc, 4),
                "threshold": cutoff,
                "confirmed": confirmed,
                "incident_id": incident_id,
                "new_incident": is_new_incident,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    incidents_path = args.output_dir / "confirmed_incidents.json"
    results_path = args.output_dir / "confirmation_results.json"
    write_confirmed_incidents_json(manager.incidents, incidents_path)
    results_path.write_text(json.dumps(confirmation_results, indent=2), encoding="utf-8")

    exp3_report = None
    if args.ground_truth is not None:
        exp3_report = evaluate_experiment3(confirmation_results, manager.incidents, args.ground_truth)
        exp3_path = args.output_dir / "experiment3_comparison.json"
        exp3_path.write_text(json.dumps(exp3_report, indent=2), encoding="utf-8")
        print(f"Experiment 3 report: {exp3_path}")

    if args.render_video and args.video is not None:
        preview_path = args.output_dir / "confirmed_preview.mp4"
        render_confirmed_preview(
            video_path=args.video,
            confirmation_results=confirmation_results,
            confirmed_incidents=manager.incidents,
            output_path=preview_path,
            target_width=args.width,
        )
        print(f"Confirmed preview video: {preview_path}")

    print(
        f"Evaluated {len(candidates)} candidate(s) -> "
        f"{sum(1 for r in confirmation_results if r['confirmed'])} confirmed, "
        f"{len(manager.incidents)} deduplicated incident(s)"
    )
    print(f"Confirmed incidents JSON: {incidents_path}")
    print(f"Confirmation audit JSON: {results_path}")

    return {
        "confirmed_incidents": [inc.to_dict() for inc in manager.incidents],
        "confirmation_results": confirmation_results,
        "experiment3": exp3_report,
    }


if __name__ == "__main__":
    run(parse_args())
