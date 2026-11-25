"""Stage 12: Cooldown-based accident event manager and incident serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AccidentEventManager:
    """Prevent duplicate incident alerts for the same vehicles/region within cooldown."""

    def __init__(self, cooldown_sec: float = 5.0, region_iou_threshold: float = 0.10) -> None:
        if cooldown_sec < 0:
            raise ValueError("cooldown_sec must be non-negative")
        self.cooldown_sec = cooldown_sec
        self.region_iou_threshold = region_iou_threshold
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

    def register_confirmed(
        self,
        candidate: dict[str, Any],
        accident_confidence: float,
    ) -> tuple[ConfirmedIncident, bool]:
        """Register a confirmed candidate; returns (incident, is_new_incident)."""
        ts = float(candidate["timestamp_sec"])
        track_ids = [int(t) for t in candidate.get("involved_track_ids", [])]
        classes = [str(c) for c in candidate.get("involved_classes", [])]
        while len(classes) < len(track_ids):
            classes.append("vehicle")

        vehicles = [
            {"track_id": tid, "class_name": cls_name}
            for tid, cls_name in zip(track_ids, classes)
        ]
        region_list = [float(v) for v in candidate.get("candidate_region", [0.0, 0.0, 0.0, 0.0])]
        region_tuple = (region_list[0], region_list[1], region_list[2], region_list[3])
        cand_id = int(candidate.get("candidate_id", 0))

        for existing in reversed(self._incidents):
            if self._matches_incident(existing, ts, set(track_ids), region_tuple):
                if cand_id not in existing.candidate_ids:
                    existing.candidate_ids.append(cand_id)
                if accident_confidence > existing.accident_confidence:
                    existing.accident_confidence = round(float(accident_confidence), 4)
                    existing.timestamp_sec = round(ts, 3)
                    existing.timestamp_hms = format_hms(ts)
                    existing.frame_index = int(candidate.get("frame_index", existing.frame_index))
                    existing.motion_score = int(candidate.get("motion_score", existing.motion_score))
                    existing.candidate_region = region_list
                    if candidate.get("clip_path"):
                        existing.clip_path = candidate["clip_path"]
                    if candidate.get("keyframe_path"):
                        existing.keyframe_path = candidate["keyframe_path"]
                # Merge any newly observed vehicles from the same collision window.
                known_ids = {int(v["track_id"]) for v in existing.vehicles}
                for veh in vehicles:
                    if veh["track_id"] not in known_ids:
                        existing.vehicles.append(veh)
                        known_ids.add(veh["track_id"])
                return existing, False

        incident = ConfirmedIncident(
            incident_id=f"{len(self._incidents) + 1:03d}",
            timestamp_sec=round(ts, 3),
            timestamp_hms=format_hms(ts),
            frame_index=int(candidate.get("frame_index", 0)),
            vehicles=vehicles,
            candidate_region=region_list,
            motion_score=int(candidate.get("motion_score", 0)),
            accident_confidence=round(float(accident_confidence), 4),
            candidate_ids=[cand_id],
            clip_path=candidate.get("clip_path"),
            keyframe_path=candidate.get("keyframe_path"),
        )
        self._incidents.append(incident)
        return incident, True


def write_confirmed_incidents_json(
    incidents: list[ConfirmedIncident],
    output_path: Path,
) -> None:
    """Write deduplicated confirmed incidents to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [inc.to_dict() for inc in incidents]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
