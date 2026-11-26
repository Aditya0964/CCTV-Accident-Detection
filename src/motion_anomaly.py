"""Stage 4-8: Motion anomaly, vehicle interaction, stopping, scoring, and candidate generation."""

from __future__ import annotations

import csv
import json
import math
from bisect import bisect_left
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


SCORE_WEIGHTS: dict[str, int] = {
    "sudden_speed_drop": 2,
    "sharp_direction_change": 2,
    "very_close_interaction": 2,
    "bounding_box_overlap": 1,
    "post_interaction_stopping": 2,
}


@dataclass(frozen=True)
class TrackObservation:
    """Single tracked vehicle detection at one frame."""

    frame: int
    timestamp_sec: float
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def diagonal(self) -> float:
        return math.hypot(self.width, self.height)

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> tuple[float, float]:
        return (self.center_x, self.center_y)


@dataclass
class AnomalyConfig:
    """Configurable thresholds and temporal windows for motion anomaly detection."""

    fps: Optional[float] = None
    mode: str = "full"  # "full" or "motion_only" (Experiment 2 ablation)
    recent_window_sec: float = 0.35
    prev_window_sec: float = 1.20
    speed_drop_ratio: float = 0.25
    min_prev_speed_px_s: float = 25.0
    max_frame_gap: int = 5
    angle_stride_sec: float = 0.15
    direction_change_deg: float = 50.0
    min_vector_norm_px: float = 6.0
    proximity_norm_threshold: float = 1.25
    iou_threshold: float = 0.05
    approach_lookback_sec: float = 0.75
    stationary_speed_px_s: float = 15.0
    post_stop_lookahead_sec: float = 2.0
    min_stop_duration_sec: float = 0.6
    candidate_threshold: Optional[int] = None
    merge_window_sec: float = 5.0
    merge_region_iou: float = 0.10
    clip_before_sec: float = 2.0
    clip_after_sec: float = 2.0
    region_padding_px: float = 24.0
    preserve_moving_contacts: bool = False
    tracking_size: tuple[int, int] | None = None
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.mode not in ("full", "motion_only"):
            raise ValueError(f"Unsupported mode '{self.mode}'; expected 'full' or 'motion_only'")
        if self.recent_window_sec <= 0 or self.prev_window_sec <= 0:
            raise ValueError("recent_window_sec and prev_window_sec must be positive")
        if self.max_frame_gap < 1:
            raise ValueError("max_frame_gap must be >= 1")
        if self.speed_drop_ratio <= 0 or self.direction_change_deg <= 0:
            raise ValueError("speed_drop_ratio and direction_change_deg must be positive")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("fps must be positive when specified")

    def effective_candidate_threshold(self) -> int:
        if self.candidate_threshold is not None:
            return self.candidate_threshold
        return 5 if self.mode == "full" else 4

    def max_possible_score(self) -> int:
        if self.mode == "motion_only":
            return SCORE_WEIGHTS["sudden_speed_drop"] + SCORE_WEIGHTS["sharp_direction_change"]
        return sum(SCORE_WEIGHTS.values())


@dataclass
class FrameTrackScore:
    """Motion anomaly diagnostic and score for one vehicle at one frame."""

    frame: int
    timestamp_sec: float
    track_id: int
    class_name: str
    box: tuple[float, float, float, float]
    speed_px_s: float
    recent_speed_px_s: Optional[float]
    prev_speed_px_s: Optional[float]
    speed_ratio: Optional[float]
    direction_change_deg: Optional[float]
    partner_track_id: Optional[int]
    partner_box: Optional[tuple[float, float, float, float]]
    normalized_distance: Optional[float]
    pair_iou: float
    approaching: bool
    sudden_speed_drop: bool
    sharp_direction_change: bool
    very_close_interaction: bool
    bounding_box_overlap: bool
    post_interaction_stopping: bool
    motion_score: int
    status: str


@dataclass
class CandidateEvent:
    """Grouped accident candidate emitted when motion score exceeds threshold."""

    candidate_id: int
    timestamp_sec: float
    frame_index: int
    clip_start_sec: float
    clip_end_sec: float
    involved_track_ids: list[int]
    involved_classes: list[str]
    candidate_region: list[float]
    motion_score: int
    max_score: int
    evidence: dict[str, bool]
    metrics: dict[str, Any]
    clip_path: Optional[str] = None
    keyframe_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute intersection-over-union between two xyxy bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0.0 else 0.0


def union_boxes(
    boxes: Iterable[tuple[float, float, float, float]],
    padding_px: float = 0.0,
) -> tuple[float, float, float, float]:
    """Compute padded bounding box enclosing all given xyxy boxes."""
    box_list = list(boxes)
    if not box_list:
        raise ValueError("Cannot compute union of empty box sequence")
    x1 = min(b[0] for b in box_list) - padding_px
    y1 = min(b[1] for b in box_list) - padding_px
    x2 = max(b[2] for b in box_list) + padding_px
    y2 = max(b[3] for b in box_list) + padding_px
    return (max(0.0, x1), max(0.0, y1), x2, y2)


def load_tracks_csv(csv_path: Path) -> list[TrackObservation]:
    """Load Phase 1 track detections from tracks.csv."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Tracks CSV not found: {csv_path}")

    observations: list[TrackObservation] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "frame",
            "timestamp_sec",
            "track_id",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "center_x",
            "center_y",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Invalid tracks CSV header in {csv_path}: {reader.fieldnames}")

        for row in reader:
            observations.append(
                TrackObservation(
                    frame=int(row["frame"]),
                    timestamp_sec=float(row["timestamp_sec"]),
                    track_id=int(row["track_id"]),
                    class_id=int(row["class_id"]),
                    class_name=row["class_name"],
                    confidence=float(row["confidence"]),
                    x1=float(row["x1"]),
                    y1=float(row["y1"]),
                    x2=float(row["x2"]),
                    y2=float(row["y2"]),
                    center_x=float(row["center_x"]),
                    center_y=float(row["center_y"]),
                )
            )
    return observations


def infer_fps(observations: Iterable[TrackObservation], default_fps: float = 30.0) -> float:
    """Estimate source FPS from frame indices and timestamps."""
    frame_times = sorted({(obs.frame, obs.timestamp_sec) for obs in observations})
    if len(frame_times) < 2:
        return default_fps
    first_frame, first_time = frame_times[0]
    last_frame, last_time = frame_times[-1]
    if last_frame > first_frame and last_time > first_time:
        return (last_frame - first_frame) / (last_time - first_time)
    return default_fps


@dataclass
class _TrackStepKinematics:
    speed_px_s: float
    segment_valid: bool
    recent_speed_px_s: Optional[float]
    prev_speed_px_s: Optional[float]
    speed_ratio: Optional[float]
    sudden_speed_drop: bool
    direction_change_deg: Optional[float]
    sharp_direction_change: bool
    stops_ahead: bool


def _compute_track_kinematics(
    track_obs: list[TrackObservation],
    fps: float,
    config: AnomalyConfig,
) -> list[_TrackStepKinematics]:
    """Compute per-step speed, Rule A speed drop, Rule B direction change, and forward stopping."""
    n = len(track_obs)
    speeds = [0.0] * n
    valid_seg = [False] * n

    for i in range(1, n):
        df = track_obs[i].frame - track_obs[i - 1].frame
        dt = track_obs[i].timestamp_sec - track_obs[i - 1].timestamp_sec
        if 1 <= df <= config.max_frame_gap and dt > config.eps:
            dist = math.hypot(
                track_obs[i].center_x - track_obs[i - 1].center_x,
                track_obs[i].center_y - track_obs[i - 1].center_y,
            )
            speeds[i] = dist / dt
            valid_seg[i] = True

    stride_frames = max(1, round(config.angle_stride_sec * fps))
    # Contiguous valid segment start index for each step to avoid crossing occlusion gaps.
    contig_start = [0] * n
    for i in range(1, n):
        contig_start[i] = contig_start[i - 1] if valid_seg[i] else i

    # Precompute forward sustained stopping: for index i, does a contiguous low-speed run
    # of duration >= min_stop_duration_sec start/occur within (t_i, t_i + post_stop_lookahead_sec]?
    stops_ahead = [False] * n
    for i in range(n):
        t_limit = track_obs[i].timestamp_sec + config.post_stop_lookahead_sec
        run_start_time: Optional[float] = None
        run_count = 0
        for j in range(i + 1, n):
            if not valid_seg[j] or track_obs[j].timestamp_sec > t_limit:
                break
            if speeds[j] < config.stationary_speed_px_s:
                if run_start_time is None:
                    run_start_time = track_obs[j - 1].timestamp_sec
                    run_count = 1
                else:
                    run_count += 1
                if (
                    run_count >= 2
                    and (track_obs[j].timestamp_sec - run_start_time) >= config.min_stop_duration_sec
                ):
                    stops_ahead[i] = True
                    break
            else:
                run_start_time = None
                run_count = 0

    results: list[_TrackStepKinematics] = []
    for i in range(n):
        t_now = track_obs[i].timestamp_sec
        start_idx = contig_start[i]

        # Rule A: compare recent speed window vs preceding baseline speed window.
        recent_speeds: list[float] = []
        prev_speeds: list[float] = []
        recent_cutoff = t_now - config.recent_window_sec
        prev_cutoff = recent_cutoff - config.prev_window_sec

        for j in range(i, start_idx, -1):
            if not valid_seg[j]:
                continue
            t_j = track_obs[j].timestamp_sec
            if t_j > recent_cutoff:
                recent_speeds.append(speeds[j])
            elif t_j >= prev_cutoff:
                prev_speeds.append(speeds[j])
            else:
                break

        recent_mean: Optional[float] = None
        prev_mean: Optional[float] = None
        speed_ratio: Optional[float] = None
        speed_drop = False
        if recent_speeds and prev_speeds:
            recent_mean = sum(recent_speeds) / len(recent_speeds)
            prev_mean = sum(prev_speeds) / len(prev_speeds)
            if prev_mean >= config.min_prev_speed_px_s:
                speed_ratio = recent_mean / (prev_mean + config.eps)
                speed_drop = speed_ratio < config.speed_drop_ratio

        # Rule B: sharp direction change between consecutive/strided displacement vectors.
        angle_deg: Optional[float] = None
        dir_change = False
        avail = i - start_idx
        if avail >= 2:
            step = stride_frames if avail >= 2 * stride_frames else min(stride_frames, avail // 2)
            idx1 = _find_strided_index(track_obs, i, step, start_idx)
            idx0 = _find_strided_index(track_obs, idx1, step, start_idx) if idx1 is not None else None
            if idx1 is not None and idx0 is not None:
                dx1 = track_obs[idx1].center_x - track_obs[idx0].center_x
                dy1 = track_obs[idx1].center_y - track_obs[idx0].center_y
                dx2 = track_obs[i].center_x - track_obs[idx1].center_x
                dy2 = track_obs[i].center_y - track_obs[idx1].center_y
                norm1 = math.hypot(dx1, dy1)
                norm2 = math.hypot(dx2, dy2)
                if norm1 >= config.min_vector_norm_px and norm2 >= config.min_vector_norm_px:
                    cos_val = (dx1 * dx2 + dy1 * dy2) / (norm1 * norm2)
                    cos_clamped = max(-1.0, min(1.0, cos_val))
                    angle_deg = math.degrees(math.acos(cos_clamped))
                    dir_change = angle_deg > config.direction_change_deg

        results.append(
            _TrackStepKinematics(
                speed_px_s=speeds[i],
                segment_valid=valid_seg[i],
                recent_speed_px_s=recent_mean,
                prev_speed_px_s=prev_mean,
                speed_ratio=speed_ratio,
                sudden_speed_drop=speed_drop,
                direction_change_deg=angle_deg,
                sharp_direction_change=dir_change,
                stops_ahead=stops_ahead[i],
            )
        )

    return results


def _find_strided_index(
    track_obs: list[TrackObservation],
    end_idx: int,
    target_frame_step: int,
    min_idx: int,
) -> Optional[int]:
    """Find index <= end_idx - 1 closest to target_frame_step frames back, bounded by min_idx."""
    if end_idx <= min_idx:
        return None
    target_frame = track_obs[end_idx].frame - target_frame_step
    best_idx = end_idx - 1
    best_diff = abs(track_obs[best_idx].frame - target_frame)
    for j in range(end_idx - 2, min_idx - 1, -1):
        diff = abs(track_obs[j].frame - target_frame)
        if diff <= best_diff:
            best_diff = diff
            best_idx = j
        else:
            break
    return best_idx


@dataclass
class _PairInteraction:
    partner_id: int
    partner_box: tuple[float, float, float, float]
    norm_dist: float
    iou: float
    approaching: bool
    very_close: bool
    overlap: bool
    partner_abrupt: bool
    partner_stops_ahead: bool


@dataclass
class _MovingContact:
    first: TrackObservation
    second: TrackObservation
    initial_iou: float
    last_seen: float
    speed_drop: bool = False
    direction_change: bool = False
    emitted: bool = False


def _update_moving_contact(
    active: dict[tuple[int, int], _MovingContact], a: TrackObservation, b: TrackObservation,
    kin_a: _TrackStepKinematics, kin_b: _TrackStepKinematics, interaction: _PairInteraction,
    history: list[tuple[float, float, float]], config: AnomalyConfig,
) -> CandidateEvent | None:
    """Preserve the first overlap of a moving pair when the contact persists and motion changes."""
    pair = a.track_id, b.track_id
    timestamp = a.timestamp_sec
    if not interaction.overlap or interaction.iou >= 0.80:
        active.pop(pair, None)
        return None
    contact = active.get(pair)
    if contact is not None and timestamp - contact.last_seen > 0.25:
        active.pop(pair)
        contact = None

    if kin_a.stops_ahead or kin_b.stops_ahead:
        return None
    if contact is None:
        if history and timestamp - history[-1][0] <= 0.25 and history[-1][2] >= config.iou_threshold:
            return None
        scale = (a.diagonal + b.diagonal) / 2.0
        if (not interaction.approaching or not (kin_a.segment_valid and kin_b.segment_valid)
            or min(kin_a.recent_speed_px_s or 0, kin_b.recent_speed_px_s or 0) < 0.35 * scale):
            return None
        distance = math.dist(a.center, b.center)
        if not any(
            0.20 <= timestamp - t <= config.approach_lookback_sec and old_dist - distance >= 0.40 * scale
            for t, old_dist, _ in history
        ):
            return None
        contact = _MovingContact(a, b, interaction.iou, timestamp)
        active[pair] = contact

    contact.last_seen = timestamp
    contact.speed_drop |= kin_a.sudden_speed_drop or kin_b.sudden_speed_drop
    contact.direction_change |= kin_a.sharp_direction_change or kin_b.sharp_direction_change
    initial_time = contact.first.timestamp_sec
    if (contact.emitted or not 0 < timestamp - initial_time <= 0.35
        or interaction.iou < max(0.12, contact.initial_iou + 0.04)
        or not (contact.speed_drop or contact.direction_change)):
        return None

    score = _compute_weighted_score(
        mode="full", speed_drop=contact.speed_drop, dir_change=contact.direction_change,
        very_close=True, overlap=True, post_stopping=False,
    )
    if score < config.effective_candidate_threshold():
        return None
    contact.emitted = True
    box = union_boxes((contact.first.box, contact.second.box))
    anchor = {
        "timestamp_sec": round(initial_time, 3),
        "frame_index": contact.first.frame,
        "candidate_region": [round(v, 2) for v in box],
        "involved_track_ids": list(pair),
        "involved_classes": [contact.first.class_name, contact.second.class_name],
    }
    return CandidateEvent(
        candidate_id=0, **anchor,
        clip_start_sec=round(max(0.0, initial_time - config.clip_before_sec), 3),
        clip_end_sec=round(initial_time + config.clip_after_sec, 3),
        motion_score=score, max_score=config.max_possible_score(),
        evidence={
            "moving_contact": True, "sudden_speed_drop": contact.speed_drop,
            "sharp_direction_change": contact.direction_change, "very_close_interaction": True,
            "bounding_box_overlap": True, "post_interaction_stopping": False,
        },
        metrics={"contact_anchor": anchor, "initial_pair_iou": round(contact.initial_iou, 4),
                 "pair_iou": round(interaction.iou, 4)},
    )


def _evaluate_pair_interaction(
    obs_a: TrackObservation,
    kin_a: _TrackStepKinematics,
    obs_b: TrackObservation,
    kin_b: _TrackStepKinematics,
    pair_history: list[tuple[float, float, float]],
    config: AnomalyConfig,
) -> _PairInteraction:
    """Compute normalized proximity, IoU, and approaching state for a vehicle pair in one frame."""
    dist = math.hypot(obs_a.center_x - obs_b.center_x, obs_a.center_y - obs_b.center_y)
    avg_diag = 0.5 * (obs_a.diagonal + obs_b.diagonal)
    norm_dist = dist / max(avg_diag, config.eps)
    iou = compute_iou(obs_a.box, obs_b.box)

    approaching = False
    lookback_cutoff = obs_a.timestamp_sec - config.approach_lookback_sec
    for prev_t, prev_dist, _ in pair_history:
        if prev_t >= lookback_cutoff:
            if dist < prev_dist - 1.0:
                approaching = True
            break

    overlap = iou >= config.iou_threshold
    within_proximity = norm_dist <= config.proximity_norm_threshold
    # Require approach evidence, bbox overlap, or very tight contact to ignore parallel stopped lanes.
    very_close = within_proximity and (
        approaching or overlap or norm_dist <= 0.75 * config.proximity_norm_threshold
    )
    partner_abrupt = kin_b.sudden_speed_drop or kin_b.sharp_direction_change
    return _PairInteraction(
        partner_id=obs_b.track_id,
        partner_box=obs_b.box,
        norm_dist=norm_dist,
        iou=iou,
        approaching=approaching,
        very_close=very_close,
        overlap=overlap,
        partner_abrupt=partner_abrupt,
        partner_stops_ahead=kin_b.stops_ahead,
    )


def analyze_motion_anomalies(
    observations: list[TrackObservation],
    config: Optional[AnomalyConfig] = None,
) -> tuple[list[FrameTrackScore], list[CandidateEvent]]:
    """Run Stages 4-8 motion & interaction scoring and emit grouped candidate events."""
    cfg = config or AnomalyConfig()
    if not observations:
        return [], []

    fps = cfg.fps if cfg.fps is not None else infer_fps(observations)

    # Deduplicate same (frame, track_id) keeping highest confidence and sort tracks chronologically.
    by_track: dict[int, list[TrackObservation]] = defaultdict(list)
    seen_keys: dict[tuple[int, int], TrackObservation] = {}
    for obs in observations:
        key = (obs.frame, obs.track_id)
        prev = seen_keys.get(key)
        if prev is None or obs.confidence > prev.confidence:
            seen_keys[key] = obs

    for obs in seen_keys.values():
        by_track[obs.track_id].append(obs)
    for track_list in by_track.values():
        track_list.sort(key=lambda o: o.frame)

    kinematics_map: dict[tuple[int, int], _TrackStepKinematics] = {}
    frames_map: dict[int, list[TrackObservation]] = defaultdict(list)
    for tid, track_list in by_track.items():
        kin_list = _compute_track_kinematics(track_list, fps, cfg)
        for obs, kin in zip(track_list, kin_list):
            kinematics_map[(obs.frame, tid)] = kin
            frames_map[obs.frame].append(obs)

    pair_history: dict[tuple[int, int], list[tuple[float, float, float]]] = defaultdict(list)
    active_contacts: dict[tuple[int, int], _MovingContact] = {}
    independent_candidates: list[CandidateEvent] = []
    frame_scores: list[FrameTrackScore] = []
    threshold = cfg.effective_candidate_threshold()

    for frame_idx in sorted(frames_map.keys()):
        frame_obs = sorted(frames_map[frame_idx], key=lambda o: o.track_id)

        # Evaluate all unordered pairs once per frame, recording best partner for each vehicle.
        best_interaction: dict[int, _PairInteraction] = {}
        for idx_a in range(len(frame_obs)):
            obs_a = frame_obs[idx_a]
            kin_a = kinematics_map[(frame_idx, obs_a.track_id)]
            for idx_b in range(idx_a + 1, len(frame_obs)):
                obs_b = frame_obs[idx_b]
                kin_b = kinematics_map[(frame_idx, obs_b.track_id)]
                pair_key = (obs_a.track_id, obs_b.track_id)  # sorted because frame_obs is sorted
                hist = pair_history[pair_key]

                inter_a = _evaluate_pair_interaction(obs_a, kin_a, obs_b, kin_b, hist, cfg)
                inter_b = _evaluate_pair_interaction(obs_b, kin_b, obs_a, kin_a, hist, cfg)
                if cfg.preserve_moving_contacts and cfg.mode == "full":
                    contact = _update_moving_contact(active_contacts, obs_a, obs_b, kin_a, kin_b, inter_a, hist, cfg)
                    if contact is not None:
                        independent_candidates.append(contact)

                dist_now = math.hypot(obs_a.center_x - obs_b.center_x, obs_a.center_y - obs_b.center_y)
                hist.append((obs_a.timestamp_sec, dist_now, inter_a.iou))
                # Prune stale pair history beyond approach lookback window.
                cutoff = obs_a.timestamp_sec - cfg.approach_lookback_sec - 0.5
                while hist and hist[0][0] < cutoff:
                    hist.pop(0)

                prev_a = best_interaction.get(obs_a.track_id)
                if prev_a is None or _pair_rank(inter_a) > _pair_rank(prev_a):
                    best_interaction[obs_a.track_id] = inter_a

                prev_b = best_interaction.get(obs_b.track_id)
                if prev_b is None or _pair_rank(inter_b) > _pair_rank(prev_b):
                    best_interaction[obs_b.track_id] = inter_b

        for obs in frame_obs:
            kin = kinematics_map[(frame_idx, obs.track_id)]
            inter = best_interaction.get(obs.track_id)

            very_close = inter.very_close if inter is not None else False
            overlap = inter.overlap if inter is not None else False
            partner_abrupt = inter.partner_abrupt if inter is not None else False
            partner_stops = inter.partner_stops_ahead if inter is not None else False

            # Post-interaction stopping only counts if preceded/accompanied by an interaction or abrupt change.
            trigger_event = (
                kin.sudden_speed_drop
                or kin.sharp_direction_change
                or very_close
                or overlap
                or ((very_close or overlap) and partner_abrupt)
            )
            post_stopping = trigger_event and (
                kin.stops_ahead or ((very_close or overlap) and partner_stops)
            )

            score = _compute_weighted_score(
                mode=cfg.mode,
                speed_drop=kin.sudden_speed_drop,
                dir_change=kin.sharp_direction_change,
                very_close=very_close,
                overlap=overlap,
                post_stopping=post_stopping,
            )

            if score >= threshold:
                status = "accident_candidate"
            elif score >= 3:
                status = "suspicious"
            else:
                status = "normal"

            frame_scores.append(
                FrameTrackScore(
                    frame=frame_idx,
                    timestamp_sec=obs.timestamp_sec,
                    track_id=obs.track_id,
                    class_name=obs.class_name,
                    box=obs.box,
                    speed_px_s=kin.speed_px_s,
                    recent_speed_px_s=kin.recent_speed_px_s,
                    prev_speed_px_s=kin.prev_speed_px_s,
                    speed_ratio=kin.speed_ratio,
                    direction_change_deg=kin.direction_change_deg,
                    partner_track_id=inter.partner_id if inter is not None else None,
                    partner_box=inter.partner_box if inter is not None else None,
                    normalized_distance=inter.norm_dist if inter is not None else None,
                    pair_iou=inter.iou if inter is not None else 0.0,
                    approaching=inter.approaching if inter is not None else False,
                    sudden_speed_drop=kin.sudden_speed_drop,
                    sharp_direction_change=kin.sharp_direction_change,
                    very_close_interaction=very_close,
                    bounding_box_overlap=overlap,
                    post_interaction_stopping=post_stopping,
                    motion_score=score,
                    status=status,
                )
            )

    candidates = group_candidates(frame_scores, cfg)
    if cfg.preserve_moving_contacts and cfg.mode == "full":
        independent_candidates.extend(_occluded_contact_candidates(by_track, frames_map, cfg))
    if cfg.tracking_size is not None and cfg.mode == "full":
        independent_candidates.extend(_single_vehicle_candidates(by_track, frames_map, cfg.tracking_size, cfg))
    by_contact = {(c.frame_index, tuple(sorted(c.involved_track_ids))): index for index, c in enumerate(candidates)}
    for proposal in independent_candidates:
        key = proposal.frame_index, tuple(proposal.involved_track_ids)
        existing = by_contact.get(key)
        if existing is None:
            proposal.candidate_id = len(candidates) + 1
            by_contact[key] = len(candidates)
            candidates.append(proposal)
        else:
            proposal.candidate_id = candidates[existing].candidate_id
            candidates[existing] = proposal
    return frame_scores, candidates


def _terminal_speed_ratio(track: list[TrackObservation], config: AnomalyConfig) -> float | None:
    last = track[-1]
    offsets = (0.38, 0.18, 0.14)
    lookback = track[bisect_left(track, last.timestamp_sec - 0.44, key=lambda obs: obs.timestamp_sec):]
    first, middle, recent = [min(lookback, key=lambda obs: abs(last.timestamp_sec - offset - obs.timestamp_sec))
                             for offset in offsets]
    if any(abs(last.timestamp_sec - offset - obs.timestamp_sec) > 0.06
           for offset, obs in zip(offsets, (first, middle, recent))):
        return None
    if any(b.frame - a.frame > config.max_frame_gap for a, b in zip(lookback, lookback[1:])):
        return None
    initial_span = middle.timestamp_sec - first.timestamp_sec
    final_span = last.timestamp_sec - recent.timestamp_sec
    if initial_span < 0.1 or final_span < 0.08:
        return None
    initial_speed = math.dist(first.center, middle.center) / initial_span
    if initial_speed < max(config.min_prev_speed_px_s, 0.35 * last.diagonal):
        return None
    return (math.dist(recent.center, last.center) / final_span) / initial_speed


def _occluded_contact_candidates(
    tracks: dict[int, list[TrackObservation]],
    frames: dict[int, list[TrackObservation]],
    config: AnomalyConfig,
) -> list[CandidateEvent]:
    """Keep contacts where braking and a merged detection interrupt one vehicle's track."""
    if not frames:
        return []
    score = _compute_weighted_score(
        mode="full", speed_drop=True, dir_change=False, very_close=True, overlap=True, post_stopping=False,
    )
    if score < config.effective_candidate_threshold():
        return []
    last_timestamp = frames[max(frames)][0].timestamp_sec
    candidates = []
    for track in tracks.values():
        last = track[-1]
        timestamp = last.timestamp_sec
        end_frame = frames[last.frame]
        if len(end_frame) < 3 or last_timestamp - timestamp < 0.4 or timestamp - track[0].timestamp_sec < 0.35:
            continue
        ratio = _terminal_speed_ratio(track, config)
        if ratio is None or ratio > 0.30:
            continue
        partners = [
            obs for obs in end_frame if obs.track_id != last.track_id and obs.class_id == last.class_id
            and 0.01 <= compute_iou(last.box, obs.box) <= 0.45
        ]
        replacements = [
            obs for obs in end_frame if obs.track_id != last.track_id and obs.class_id == last.class_id
            and timestamp - tracks[obs.track_id][0].timestamp_sec <= 0.25
            and tracks[obs.track_id][-1].timestamp_sec - timestamp >= 0.33
            and compute_iou(last.box, obs.box) >= 0.23
        ]
        if not replacements:
            continue
        recent_track = track[bisect_left(track, timestamp - 0.25, key=lambda obs: obs.timestamp_sec):]
        for partner in partners:
            pair_history = [
                (obs, other) for obs in recent_track
                for other in frames[obs.frame] if other.track_id == partner.track_id
            ]
            if sum(0.07 <= compute_iou(obs.box, other.box) <= 0.45 for obs, other in pair_history) < 3:
                continue
            first_partner = pair_history[0][1]
            if (timestamp - first_partner.timestamp_sec < 0.14
                or math.dist(first_partner.center, partner.center) < 0.08 * partner.diagonal):
                continue
            replacement = next(
                (obs for obs in replacements if obs.track_id != partner.track_id
                 and compute_iou(partner.box, obs.box) >= 0.23
                 and obs.width * obs.height >= 1.1 * max(last.width * last.height, partner.width * partner.height)),
                None,
            )
            if replacement is None:
                continue
            pair = sorted((last, partner), key=lambda obs: obs.track_id)
            anchor = {
                "timestamp_sec": round(timestamp, 3),
                "frame_index": last.frame,
                "candidate_region": [round(value, 2) for value in union_boxes((last.box, partner.box))],
                "involved_track_ids": [obs.track_id for obs in pair],
                "involved_classes": [obs.class_name for obs in pair],
            }
            candidates.append(CandidateEvent(
                candidate_id=0, **anchor,
                clip_start_sec=round(max(0.0, timestamp - config.clip_before_sec), 3),
                clip_end_sec=round(timestamp + config.clip_after_sec, 3),
                motion_score=score, max_score=config.max_possible_score(),
                evidence={"occluded_contact": True, "sudden_speed_drop": True,
                          "very_close_interaction": True, "bounding_box_overlap": True},
                metrics={"contact_anchor": anchor, "terminal_speed_ratio": round(ratio, 3),
                         "replacement_track_id": replacement.track_id},
            ))
    return candidates


def _single_vehicle_candidates(
    tracks: dict[int, list[TrackObservation]],
    frames: dict[int, list[TrackObservation]],
    tracking_size: tuple[int, int],
    config: AnomalyConfig,
) -> list[CandidateEvent]:
    """Propose interior track losses coupled with sudden trajectory and box changes."""
    score = 5
    if not frames or score < config.effective_candidate_threshold():
        return []
    width, height = tracking_size
    frame_numbers = sorted(frames)
    last_timestamp = frames[frame_numbers[-1]][0].timestamp_sec
    candidates = []
    for track in tracks.values():
        last = track[-1]
        timestamp = last.timestamp_sec
        if timestamp - track[0].timestamp_sec < 0.6 or last_timestamp - timestamp < 0.45:
            continue
        scale = last.diagonal
        if scale <= config.eps or min(last.x1, last.y1, width - last.x2, height - last.y2) < 0.25 * scale:
            continue
        if last.frame - track[-2].frame > config.max_frame_gap:
            continue
        # Compare short, separate windows. The longer Rule B stride can smooth
        # away the change when only a few frames survive at the end of the track.
        offsets = (0.44, 0.16, 0.12)
        lookback = track[bisect_left(track, timestamp - 0.6, key=lambda obs: obs.timestamp_sec):]
        anchors = [min(lookback, key=lambda obs: abs(timestamp - offset - obs.timestamp_sec))
                   for offset in offsets]
        if any(abs(timestamp - offset - obs.timestamp_sec) > 0.06 for offset, obs in zip(offsets, anchors)):
            continue
        if any(b.frame - a.frame > config.max_frame_gap for a, b in zip(lookback, lookback[1:])):
            continue
        first, previous, recent = anchors
        incoming = (previous.center_x - first.center_x, previous.center_y - first.center_y)
        terminal = (last.center_x - recent.center_x, last.center_y - recent.center_y)
        incoming_length, terminal_length = math.hypot(*incoming), math.hypot(*terminal)
        if incoming_length < 0.13 * scale or terminal_length < 0.09 * scale:
            continue
        cosine = sum(a * b for a, b in zip(incoming, terminal)) / (incoming_length * terminal_length)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        previous_area = previous.width * previous.height
        if previous_area <= config.eps:
            continue
        box_growth = (last.width * last.height) / previous_area
        if angle < 18.0 or box_growth < 1.25:
            continue
        # An overlapping detection after the loss can be a tracker ID swap.
        index = bisect_left(frame_numbers, last.frame + 1)
        replacement = False
        for frame_index in range(index, len(frame_numbers)):
            observations = frames[frame_numbers[frame_index]]
            elapsed = observations[0].timestamp_sec - timestamp
            if elapsed > 0.52:
                break
            if elapsed >= 0.10 and any(compute_iou(last.box, obs.box) >= 0.20 for obs in observations):
                replacement = True
                break
        if replacement:
            continue
        candidates.append(CandidateEvent(
            candidate_id=0,
            timestamp_sec=round(timestamp, 3),
            frame_index=last.frame,
            clip_start_sec=round(max(0.0, timestamp - config.clip_before_sec), 3),
            clip_end_sec=round(timestamp + config.clip_after_sec, 3),
            involved_track_ids=[last.track_id],
            involved_classes=[last.class_name],
            candidate_region=[round(value, 2) for value in last.box],
            motion_score=score,
            max_score=config.max_possible_score(),
            evidence={"single_vehicle_track_loss": True},
            metrics={"terminal_direction_change_deg": round(angle, 2), "terminal_box_area_ratio": round(box_growth, 3)},
        ))
    return candidates


def _pair_rank(inter: _PairInteraction) -> tuple[int, int, float, float]:
    """Rank interaction partners so collision/overlap partners take priority."""
    return (
        int(inter.overlap),
        int(inter.very_close),
        inter.iou,
        -inter.norm_dist,
    )


def _compute_weighted_score(
    *,
    mode: str,
    speed_drop: bool,
    dir_change: bool,
    very_close: bool,
    overlap: bool,
    post_stopping: bool,
) -> int:
    score = 0
    if speed_drop:
        score += SCORE_WEIGHTS["sudden_speed_drop"]
    if dir_change:
        score += SCORE_WEIGHTS["sharp_direction_change"]
    if mode == "full":
        if very_close:
            score += SCORE_WEIGHTS["very_close_interaction"]
        if overlap:
            score += SCORE_WEIGHTS["bounding_box_overlap"]
        if post_stopping:
            score += SCORE_WEIGHTS["post_interaction_stopping"]
    return score


def group_candidates(
    frame_scores: list[FrameTrackScore],
    config: AnomalyConfig,
) -> list[CandidateEvent]:
    """Merge contiguous/nearby candidate triggers into deduplicated peak CandidateEvents."""
    threshold = config.effective_candidate_threshold()
    flagged = [fs for fs in frame_scores if fs.motion_score >= threshold]
    if not flagged:
        return []

    flagged.sort(key=lambda fs: (fs.frame, -fs.motion_score, fs.track_id))
    score_lookup: dict[tuple[int, int], FrameTrackScore] = {
        (fs.frame, fs.track_id): fs for fs in frame_scores
    }

    clusters: list[dict[str, Any]] = []
    for entry in flagged:
        entry_ids = {entry.track_id}
        if entry.partner_track_id is not None and (
            entry.very_close_interaction or entry.bounding_box_overlap
        ):
            entry_ids.add(entry.partner_track_id)

        boxes = [entry.box]
        if entry.partner_box is not None and (
            entry.very_close_interaction or entry.bounding_box_overlap
        ):
            boxes.append(entry.partner_box)
        entry_region = union_boxes(boxes, padding_px=config.region_padding_px)

        matched_cluster = None
        for cluster in reversed(clusters):
            if entry.timestamp_sec - cluster["last_ts"] > config.merge_window_sec:
                continue
            same_vehicles = bool(entry_ids & cluster["track_ids"])
            same_region = compute_iou(entry_region, cluster["peak_region"]) >= config.merge_region_iou
            if same_vehicles or same_region:
                matched_cluster = cluster
                break

        if matched_cluster is None:
            clusters.append(
                {
                    "entries": [entry],
                    "track_ids": set(entry_ids),
                    "last_ts": entry.timestamp_sec,
                    "peak_entry": entry,
                    "peak_region": entry_region,
                }
            )
        else:
            matched_cluster["entries"].append(entry)
            matched_cluster["track_ids"].update(entry_ids)
            matched_cluster["last_ts"] = max(matched_cluster["last_ts"], entry.timestamp_sec)
            if _candidate_rank(entry) > _candidate_rank(matched_cluster["peak_entry"]):
                matched_cluster["peak_entry"] = entry
                matched_cluster["peak_region"] = entry_region

    max_score = config.max_possible_score()
    candidates: list[CandidateEvent] = []
    for idx, cluster in enumerate(clusters, start=1):
        peak: FrameTrackScore = cluster["peak_entry"]
        involved_ids = [peak.track_id]
        involved_classes = [peak.class_name]
        boxes = [peak.box]

        if peak.partner_track_id is not None and (
            peak.very_close_interaction or peak.bounding_box_overlap
        ):
            partner_score = score_lookup.get((peak.frame, peak.partner_track_id))
            involved_ids.append(peak.partner_track_id)
            if partner_score is not None:
                involved_classes.append(partner_score.class_name)
                boxes.append(partner_score.box)
            elif peak.partner_box is not None:
                involved_classes.append("unknown")
                boxes.append(peak.partner_box)

        region = union_boxes(boxes, padding_px=config.region_padding_px)
        t_peak = peak.timestamp_sec
        contact_anchor = _contact_anchor(cluster["entries"], peak, involved_ids, region, score_lookup, config)

        candidates.append(
            CandidateEvent(
                candidate_id=idx,
                timestamp_sec=round(t_peak, 3),
                frame_index=peak.frame,
                clip_start_sec=round(max(0.0, t_peak - config.clip_before_sec), 3),
                clip_end_sec=round(t_peak + config.clip_after_sec, 3),
                involved_track_ids=involved_ids,
                involved_classes=involved_classes,
                candidate_region=[round(v, 2) for v in region],
                motion_score=peak.motion_score,
                max_score=max_score,
                evidence={
                    "sudden_speed_drop": peak.sudden_speed_drop,
                    "sharp_direction_change": peak.sharp_direction_change,
                    "very_close_interaction": peak.very_close_interaction,
                    "bounding_box_overlap": peak.bounding_box_overlap,
                    "post_interaction_stopping": peak.post_interaction_stopping,
                },
                metrics={
                    "speed_px_s": round(peak.speed_px_s, 2),
                    "speed_ratio": round(peak.speed_ratio, 4) if peak.speed_ratio is not None else None,
                    "direction_change_deg": (
                        round(peak.direction_change_deg, 2)
                        if peak.direction_change_deg is not None
                        else None
                    ),
                    "normalized_distance": (
                        round(peak.normalized_distance, 4)
                        if peak.normalized_distance is not None
                        else None
                    ),
                    "pair_iou": round(peak.pair_iou, 4),
                    "trigger_frames_count": len({e.frame for e in cluster["entries"]}),
                    "contact_anchor": contact_anchor,
                },
            )
        )

    return candidates


def _contact_anchor(entries, peak, involved_ids, peak_region, score_lookup, config) -> dict | None:
    """Find an earlier overlap in the same event before the vehicles settle."""
    contacts = []
    for entry in entries:
        if not peak.timestamp_sec - config.clip_before_sec <= entry.timestamp_sec <= peak.timestamp_sec:
            continue
        if not entry.approaching or not 0.10 <= entry.pair_iou < 0.80 or entry.partner_box is None:
            continue
        if entry.track_id not in involved_ids and entry.partner_track_id not in involved_ids:
            continue
        region = union_boxes((entry.box, entry.partner_box))
        if compute_iou(region, peak_region) < config.merge_region_iou:
            continue
        contacts.append((entry, region))
    if not contacts:
        return None
    minimum_overlap = 0.90 * max(entry.pair_iou for entry, _ in contacts)
    entry, box = min(
        ((entry, box) for entry, box in contacts if entry.pair_iou >= minimum_overlap),
        key=lambda item: item[0].frame,
    )
    partner = score_lookup.get((entry.frame, entry.partner_track_id))
    return {
        "timestamp_sec": round(entry.timestamp_sec, 3),
        "frame_index": entry.frame,
        "candidate_region": [round(value, 2) for value in box],
        "involved_track_ids": [entry.track_id, entry.partner_track_id],
        "involved_classes": [entry.class_name, partner.class_name if partner is not None else "unknown"],
    }


def _candidate_rank(entry: FrameTrackScore) -> tuple[int, float, float, int]:
    """Higher tuple wins when picking canonical peak frame inside a candidate cluster."""
    drop_severity = 1.0 - entry.speed_ratio if entry.speed_ratio is not None else 0.0
    return (entry.motion_score, entry.pair_iou, drop_severity, -entry.frame)


def write_motion_scores_csv(frame_scores: list[FrameTrackScore], output_path: Path) -> None:
    """Write per-frame per-track motion anomaly diagnostics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "timestamp_sec",
                "track_id",
                "class_name",
                "speed_px_s",
                "recent_speed_px_s",
                "prev_speed_px_s",
                "speed_ratio",
                "direction_change_deg",
                "partner_track_id",
                "normalized_distance",
                "pair_iou",
                "approaching",
                "sudden_speed_drop",
                "sharp_direction_change",
                "very_close_interaction",
                "bounding_box_overlap",
                "post_interaction_stopping",
                "motion_score",
                "status",
            ]
        )
        for fs in frame_scores:
            writer.writerow(
                [
                    fs.frame,
                    f"{fs.timestamp_sec:.3f}",
                    fs.track_id,
                    fs.class_name,
                    f"{fs.speed_px_s:.2f}",
                    f"{fs.recent_speed_px_s:.2f}" if fs.recent_speed_px_s is not None else "",
                    f"{fs.prev_speed_px_s:.2f}" if fs.prev_speed_px_s is not None else "",
                    f"{fs.speed_ratio:.4f}" if fs.speed_ratio is not None else "",
                    f"{fs.direction_change_deg:.2f}" if fs.direction_change_deg is not None else "",
                    fs.partner_track_id if fs.partner_track_id is not None else "",
                    f"{fs.normalized_distance:.4f}" if fs.normalized_distance is not None else "",
                    f"{fs.pair_iou:.4f}",
                    int(fs.approaching),
                    int(fs.sudden_speed_drop),
                    int(fs.sharp_direction_change),
                    int(fs.very_close_interaction),
                    int(fs.bounding_box_overlap),
                    int(fs.post_interaction_stopping),
                    fs.motion_score,
                    fs.status,
                ]
            )


def write_candidates_json(candidates: list[CandidateEvent], output_path: Path) -> None:
    """Write grouped accident candidate events to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [c.to_dict() for c in candidates]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
