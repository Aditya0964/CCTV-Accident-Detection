"""Standalone CLI for Stage 12 accident event management and persistent event logging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.event_manager import (
    AccidentEventLogger,
    AccidentEventManager,
    ConfirmedIncident,
    write_confirmed_incidents_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/confirmation_results.json"),
        help="Input JSON (confirmation_results.json, confirmed_incidents.json, or candidates.json)",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="Optional candidates.json to enrich vehicle classes and clip/keyframe paths",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--cooldown-sec", type=float, default=5.0)
    parser.add_argument("--region-iou-threshold", type=float, default=0.10)
    parser.add_argument(
        "--accept-all-candidates",
        action="store_true",
        help="Treat raw candidates.json entries as confirmed events (e.g. for motion-only logging)",
    )
    parser.add_argument(
        "--min-motion-score",
        type=int,
        default=5,
        help="Minimum motion score when --accept-all-candidates is used",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing events.log and events.jsonl instead of overwriting",
    )
    parser.add_argument(
        "--no-individual-files",
        action="store_true",
        help="Skip creating outputs/incidents/incident_XXX.{json,mp4,jpg} bundles",
    )
    parser.add_argument(
        "--no-copy-media",
        action="store_true",
        help="Reference existing clip/keyframe paths without copying into outputs/incidents/",
    )
    parser.add_argument("--echo", action="store_true", help="Print formatted event blocks to stdout")
    return parser.parse_args(argv)


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    return data


def _enrich_from_candidates(
    records: list[dict[str, Any]],
    candidates_path: Path | None,
) -> list[dict[str, Any]]:
    if candidates_path is None or not candidates_path.is_file():
        return records
    by_id = {
        int(c["candidate_id"]): c
        for c in _load_json_list(candidates_path)
        if "candidate_id" in c
    }
    enriched: list[dict[str, Any]] = []
    for rec in records:
        cid = rec.get("candidate_id")
        merged = dict(rec)
        if cid is not None and int(cid) in by_id:
            src = by_id[int(cid)]
            for key in ("involved_classes", "clip_path", "keyframe_path", "candidate_region"):
                if not merged.get(key) and src.get(key) is not None:
                    merged[key] = src[key]
        enriched.append(merged)
    return enriched


def run(args: argparse.Namespace) -> dict[str, Any]:
    records = _load_json_list(args.input)
    records = _enrich_from_candidates(records, args.candidates)

    logger = AccidentEventLogger(
        output_dir=args.output_dir,
        append=args.append,
        echo_console=args.echo,
    )
    manager = AccidentEventManager(
        cooldown_sec=args.cooldown_sec,
        region_iou_threshold=args.region_iou_threshold,
        logger=logger,
    )

    if records and "incident_id" in records[0] and "vehicles" in records[0]:
        # Already deduplicated ConfirmedIncident records: log and export directly.
        incidents = [ConfirmedIncident(**rec) for rec in records]
        for inc in incidents:
            logger.log_event(
                event_type="INCIDENT_CREATED",
                incident_id=inc.incident_id,
                candidate_id=inc.candidate_ids[0] if inc.candidate_ids else None,
                timestamp_sec=inc.timestamp_sec,
                frame_index=inc.frame_index,
                vehicles=inc.vehicles,
                candidate_region=inc.candidate_region,
                motion_score=inc.motion_score,
                accident_confidence=inc.accident_confidence,
                smoke_detected=inc.smoke_detected,
                fire_detected=inc.fire_detected,
                visual_risk=inc.visual_risk,
                reason="Logged from confirmed incidents JSON",
            )
    else:
        for rec in sorted(records, key=lambda r: float(r.get("timestamp_sec", 0.0))):
            if "confirmed" in rec:
                p_acc = float(rec.get("p_accident", 0.0))
                threshold = float(rec.get("threshold", 0.5))
                if rec["confirmed"]:
                    manager.register_confirmed(rec, accident_confidence=p_acc)
                else:
                    manager.record_rejected(rec, accident_confidence=p_acc, threshold=threshold)
            elif args.accept_all_candidates:
                m_score = int(rec.get("motion_score", 0))
                if m_score >= args.min_motion_score:
                    proxy_conf = float(rec.get("p_accident", min(1.0, m_score / 9.0)))
                    manager.register_confirmed(rec, accident_confidence=proxy_conf)
                else:
                    manager.record_rejected(
                        rec,
                        accident_confidence=0.0,
                        threshold=float(args.min_motion_score),
                    )
            else:
                raise ValueError(
                    "Input JSON lacks 'confirmed' or 'incident_id'. "
                    "Pass --accept-all-candidates to log raw candidates.json entries."
                )
        incidents = manager.incidents

    exported_files = logger.write_incidents_summary(
        incidents,
        export_individual=not args.no_individual_files,
        copy_media=not args.no_copy_media,
    )
    confirmed_json = args.output_dir / "confirmed_incidents.json"
    write_confirmed_incidents_json(incidents, confirmed_json)

    print(
        f"Logged {len(logger.entries)} event(s) -> {len(incidents)} incident(s) "
        f"| log={logger.log_path} | jsonl={logger.jsonl_path} | csv={logger.csv_path}"
    )
    return {
        "incidents": [inc.to_dict() for inc in incidents],
        "logged_events_count": len(logger.entries),
        "log_path": str(logger.log_path),
        "jsonl_path": str(logger.jsonl_path),
        "csv_path": str(logger.csv_path),
        "exported_incident_files": [str(p) for p in exported_files],
    }


if __name__ == "__main__":
    run(parse_args())
