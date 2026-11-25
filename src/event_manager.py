"""Stage 12: Cooldown-based accident event manager, incident serialization, and event logging."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.motion_anomaly import compute_iou


def format_hms(seconds: float) -> str:
    """Format non-negative seconds as HH:MM:SS."""
    whole = max(0, int(round(seconds)))
    hours = whole // 3600
    minutes = (whole % 3600) // 60
    secs = whole % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ConfirmedIncident:
    """Deduplicated confirmed traffic accident event."""

    incident_id: str
    timestamp_sec: float
    timestamp_hms: str
    frame_index: int
    vehicles: list[dict[str, Any]]
    candidate_region: list[float]
    motion_score: int
    accident_confidence: float
    candidate_ids: list[int]
    clip_path: Optional[str] = None
    keyframe_path: Optional[str] = None
    smoke_detected: Optional[bool] = None
    fire_detected: Optional[bool] = None
    visual_risk: Optional[str] = None
    incident_json_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def format_incident_block(
    event_type: str,
    logged_at_utc: str,
    timestamp_sec: float,
    timestamp_hms: str,
    frame_index: int,
    vehicles: list[dict[str, Any]],
    motion_score: int,
    confidence: float,
    incident_id: Optional[str] = None,
    candidate_id: Optional[int] = None,
    reason: Optional[str] = None,
    smoke_detected: Optional[bool] = None,
    fire_detected: Optional[bool] = None,
    visual_risk: Optional[str] = None,
) -> str:
    """Format a human-readable event log block matching the pipeline report style."""
    header_parts = [f"[{logged_at_utc}] {event_type}"]
    if incident_id is not None:
        header_parts.append(f"Incident ID: {incident_id}")
    if candidate_id is not None:
        header_parts.append(f"Candidate: #{candidate_id}")

    lines = [
        "-" * 48,
        " | ".join(header_parts),
        f"Time: {timestamp_hms} ({timestamp_sec:.3f}s, frame {frame_index})",
        "Vehicles:",
    ]
    if vehicles:
        for veh in vehicles:
            lines.append(f"  ID {veh['track_id']} - {veh['class_name']}")
    else:
        lines.append("  (none)")

    lines.append(f"Motion anomaly score: {motion_score}/9")
    lines.append(f"Accident confidence: {confidence:.4f}")
    if smoke_detected is not None:
        lines.append(f"Smoke: {'Detected' if smoke_detected else 'Not detected'}")
    if fire_detected is not None:
        lines.append(f"Fire: {'Detected' if fire_detected else 'Not detected'}")
    if visual_risk is not None:
        lines.append(f"Visual Risk: {visual_risk}")
    if reason:
        lines.append(f"Note: {reason}")
    lines.append("-" * 48)
    return "\n".join(lines)


class AccidentEventLogger:
    """Persist structured human-readable, JSONL, CSV, and per-incident event logs."""

    def __init__(
        self,
        output_dir: Path,
        append: bool = False,
        echo_console: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "events.log"
        self.jsonl_path = self.output_dir / "events.jsonl"
        self.csv_path = self.output_dir / "incidents.csv"
        self.incidents_dir = self.output_dir / "incidents"
        self.append = append
        self.echo_console = echo_console
        self.entries: list[dict[str, Any]] = []

        if not append:
            self.log_path.write_text("", encoding="utf-8")
            self.jsonl_path.write_text("", encoding="utf-8")

    def log_event(
        self,
        *,
        event_type: str,
        timestamp_sec: float,
        frame_index: int,
        vehicles: list[dict[str, Any]],
        motion_score: int,
        accident_confidence: float,
        incident_id: Optional[str] = None,
        candidate_id: Optional[int] = None,
        candidate_region: Optional[list[float]] = None,
        reason: Optional[str] = None,
        smoke_detected: Optional[bool] = None,
        fire_detected: Optional[bool] = None,
        visual_risk: Optional[str] = None,
    ) -> dict[str, Any]:
        """Append one event entry to memory, events.log, and events.jsonl."""
        logged_at = utc_now_iso()
        hms = format_hms(timestamp_sec)
        entry: dict[str, Any] = {
            "logged_at_utc": logged_at,
            "event_type": event_type,
            "incident_id": incident_id,
            "candidate_id": candidate_id,
            "timestamp_sec": round(float(timestamp_sec), 3),
            "timestamp_hms": hms,
            "frame_index": int(frame_index),
            "vehicles": vehicles,
            "candidate_region": candidate_region or [],
            "motion_score": int(motion_score),
            "accident_confidence": round(float(accident_confidence), 4),
            "smoke_detected": smoke_detected,
            "fire_detected": fire_detected,
            "visual_risk": visual_risk,
            "reason": reason,
        }
        self.entries.append(entry)

        block = format_incident_block(
            event_type=event_type,
            logged_at_utc=logged_at,
            timestamp_sec=entry["timestamp_sec"],
            timestamp_hms=hms,
            frame_index=entry["frame_index"],
            vehicles=vehicles,
            motion_score=entry["motion_score"],
            confidence=entry["accident_confidence"],
            incident_id=incident_id,
            candidate_id=candidate_id,
            reason=reason,
            smoke_detected=smoke_detected,
            fire_detected=fire_detected,
            visual_risk=visual_risk,
        )
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(block + "\n")
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        if self.echo_console:
            print(block)
        return entry

    def write_incidents_summary(
        self,
        incidents: list[ConfirmedIncident],
        export_individual: bool = True,
        copy_media: bool = True,
    ) -> list[Path]:
        """Export incidents.csv and optional per-incident incident_XXX.{json,mp4,jpg} bundles."""
        exported_paths: list[Path] = []
        if export_individual and incidents:
            self.incidents_dir.mkdir(parents=True, exist_ok=True)
            for inc in incidents:
                stem = f"incident_{inc.incident_id}"
                json_out = self.incidents_dir / f"{stem}.json"

                if copy_media:
                    if inc.clip_path:
                        src_clip = Path(inc.clip_path)
                        if src_clip.is_file():
                            dst_clip = self.incidents_dir / f"{stem}{src_clip.suffix}"
                            if src_clip.resolve() != dst_clip.resolve():
                                shutil.copy2(src_clip, dst_clip)
                            inc.clip_path = str(dst_clip)
                    if inc.keyframe_path:
                        src_img = Path(inc.keyframe_path)
                        if src_img.is_file():
                            dst_img = self.incidents_dir / f"{stem}{src_img.suffix}"
                            if src_img.resolve() != dst_img.resolve():
                                shutil.copy2(src_img, dst_img)
                            inc.keyframe_path = str(dst_img)

                inc.incident_json_path = str(json_out)
                json_out.write_text(json.dumps(inc.to_dict(), indent=2), encoding="utf-8")
                exported_paths.append(json_out)

        write_incidents_csv(incidents, self.csv_path)
        return exported_paths


class AccidentEventManager:
    """Prevent duplicate incident alerts for the same vehicles/region within cooldown."""

    def __init__(
        self,
        cooldown_sec: float = 5.0,
        region_iou_threshold: float = 0.10,
        logger: Optional[AccidentEventLogger] = None,
    ) -> None:
        if cooldown_sec < 0:
            raise ValueError("cooldown_sec must be non-negative")
        self.cooldown_sec = cooldown_sec
        self.region_iou_threshold = region_iou_threshold
        self.logger = logger
        self._incidents: list[ConfirmedIncident] = []

    @property
    def incidents(self) -> list[ConfirmedIncident]:
        return list(self._incidents)

    def _matches_incident(
        self,
        existing: ConfirmedIncident,
        timestamp_sec: float,
        track_ids: set[int],
        region: tuple[float, float, float, float],
    ) -> bool:
        if abs(timestamp_sec - existing.timestamp_sec) > self.cooldown_sec:
            return False
        existing_ids = {int(v["track_id"]) for v in existing.vehicles}
        same_vehicles = bool(track_ids & existing_ids)
        existing_box = tuple(float(v) for v in existing.candidate_region)
        same_region = (
            len(existing_box) == 4
            and compute_iou(region, existing_box) >= self.region_iou_threshold
        )
        return same_vehicles or same_region

    @staticmethod
    def _extract_vehicles(candidate: dict[str, Any]) -> list[dict[str, Any]]:
        if "vehicles" in candidate and isinstance(candidate["vehicles"], list):
            return [
                {
                    "track_id": int(v["track_id"]),
                    "class_name": str(v.get("class_name", "vehicle")),
                }
                for v in candidate["vehicles"]
            ]
        track_ids = [int(t) for t in candidate.get("involved_track_ids", [])]
        classes = [str(c) for c in candidate.get("involved_classes", [])]
        while len(classes) < len(track_ids):
            classes.append("vehicle")
        return [
            {"track_id": tid, "class_name": cls_name}
            for tid, cls_name in zip(track_ids, classes)
        ]

    def register_confirmed(
        self,
        candidate: dict[str, Any],
        accident_confidence: float,
    ) -> tuple[ConfirmedIncident, bool]:
        """Register a confirmed candidate; returns (incident, is_new_incident)."""
        ts = float(candidate["timestamp_sec"])
        vehicles = self._extract_vehicles(candidate)
        track_ids = {int(v["track_id"]) for v in vehicles}
        region_list = [float(v) for v in candidate.get("candidate_region", [0.0, 0.0, 0.0, 0.0])]
        region_tuple = (
            (region_list[0], region_list[1], region_list[2], region_list[3])
            if len(region_list) == 4
            else (0.0, 0.0, 0.0, 0.0)
        )
        cand_id = int(candidate.get("candidate_id", 0))
        frame_idx = int(candidate.get("frame_index", 0))
        motion_score = int(candidate.get("motion_score", 0))

        for existing in reversed(self._incidents):
            if self._matches_incident(existing, ts, track_ids, region_tuple):
                if cand_id and cand_id not in existing.candidate_ids:
                    existing.candidate_ids.append(cand_id)
                if accident_confidence > existing.accident_confidence:
                    existing.accident_confidence = round(float(accident_confidence), 4)
                    existing.timestamp_sec = round(ts, 3)
                    existing.timestamp_hms = format_hms(ts)
                    existing.frame_index = frame_idx
                    existing.motion_score = motion_score
                    existing.candidate_region = region_list
                    if candidate.get("clip_path"):
                        existing.clip_path = candidate["clip_path"]
                    if candidate.get("keyframe_path"):
                        existing.keyframe_path = candidate["keyframe_path"]

                known_ids = {int(v["track_id"]) for v in existing.vehicles}
                for veh in vehicles:
                    if veh["track_id"] not in known_ids:
                        existing.vehicles.append(veh)
                        known_ids.add(veh["track_id"])

                if self.logger is not None:
                    self.logger.log_event(
                        event_type="INCIDENT_MERGED",
                        incident_id=existing.incident_id,
                        candidate_id=cand_id,
                        timestamp_sec=ts,
                        frame_index=frame_idx,
                        vehicles=vehicles,
                        candidate_region=region_list,
                        motion_score=motion_score,
                        accident_confidence=accident_confidence,
                        reason=f"Merged into Incident {existing.incident_id} within {self.cooldown_sec:.1f}s cooldown",
                    )
                return existing, False

        incident = ConfirmedIncident(
            incident_id=f"{len(self._incidents) + 1:03d}",
            timestamp_sec=round(ts, 3),
            timestamp_hms=format_hms(ts),
            frame_index=frame_idx,
            vehicles=vehicles,
            candidate_region=region_list,
            motion_score=motion_score,
            accident_confidence=round(float(accident_confidence), 4),
            candidate_ids=[cand_id] if cand_id else [],
            clip_path=candidate.get("clip_path"),
            keyframe_path=candidate.get("keyframe_path"),
            smoke_detected=candidate.get("smoke_detected"),
            fire_detected=candidate.get("fire_detected"),
            visual_risk=candidate.get("visual_risk"),
        )
        self._incidents.append(incident)

        if self.logger is not None:
            self.logger.log_event(
                event_type="INCIDENT_CREATED",
                incident_id=incident.incident_id,
                candidate_id=cand_id,
                timestamp_sec=incident.timestamp_sec,
                frame_index=incident.frame_index,
                vehicles=incident.vehicles,
                candidate_region=incident.candidate_region,
                motion_score=incident.motion_score,
                accident_confidence=incident.accident_confidence,
                smoke_detected=incident.smoke_detected,
                fire_detected=incident.fire_detected,
                visual_risk=incident.visual_risk,
                reason="Confirmed accident event logged",
            )
        return incident, True

    def record_rejected(
        self,
        candidate: dict[str, Any],
        accident_confidence: float,
        threshold: float,
    ) -> None:
        """Log a candidate rejected by the video classifier for auditability."""
        if self.logger is None:
            return
        self.logger.log_event(
            event_type="CANDIDATE_REJECTED",
            incident_id=None,
            candidate_id=int(candidate.get("candidate_id", 0)),
            timestamp_sec=float(candidate.get("timestamp_sec", 0.0)),
            frame_index=int(candidate.get("frame_index", 0)),
            vehicles=self._extract_vehicles(candidate),
            candidate_region=[float(v) for v in candidate.get("candidate_region", [])],
            motion_score=int(candidate.get("motion_score", 0)),
            accident_confidence=accident_confidence,
            reason=f"P(accident)={accident_confidence:.4f} <= threshold {threshold:.4f}",
        )


def write_confirmed_incidents_json(
    incidents: list[ConfirmedIncident],
    output_path: Path,
) -> None:
    """Write deduplicated confirmed incidents to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [inc.to_dict() for inc in incidents]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_incidents_csv(
    incidents: list[ConfirmedIncident],
    output_path: Path,
) -> None:
    """Write tabular summary of deduplicated confirmed incidents to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "incident_id",
                "timestamp_sec",
                "timestamp_hms",
                "frame_index",
                "vehicles",
                "motion_score",
                "accident_confidence",
                "candidate_ids",
                "smoke_detected",
                "fire_detected",
                "visual_risk",
                "clip_path",
                "keyframe_path",
                "incident_json_path",
            ]
        )
        for inc in incidents:
            veh_str = "; ".join(f"ID {v['track_id']} ({v['class_name']})" for v in inc.vehicles)
            cand_str = ",".join(str(cid) for cid in inc.candidate_ids)
            writer.writerow(
                [
                    inc.incident_id,
                    f"{inc.timestamp_sec:.3f}",
                    inc.timestamp_hms,
                    inc.frame_index,
                    veh_str,
                    inc.motion_score,
                    f"{inc.accident_confidence:.4f}",
                    cand_str,
                    "" if inc.smoke_detected is None else int(inc.smoke_detected),
                    "" if inc.fire_detected is None else int(inc.fire_detected),
                    inc.visual_risk or "",
                    inc.clip_path or "",
                    inc.keyframe_path or "",
                    inc.incident_json_path or "",
                ]
            )
