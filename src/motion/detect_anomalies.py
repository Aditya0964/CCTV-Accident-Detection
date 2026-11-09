"""Offline Phase 2 pipeline: Phase 1 tracks.csv -> motion.csv and candidates.json."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .anomaly_scorer import PairEvent
from .interaction import nearby_pairs
from .motion_rules import direction_change, speed_drop, stationary_after_interaction
from .trajectory import Observation, TrackHistory

CSV_COLUMNS = {"frame", "timestamp_sec", "track_id", "class_name", "x1", "y1", "x2", "y2", "center_x", "center_y"}


def load_frames(path: Path):
    """Read Phase 1 CSV, sorted by frame, skipping untracked frames (no CSV rows)."""
    frames = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = CSV_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        for row in reader:
            obs = Observation(frame=int(row["frame"]), timestamp=float(row["timestamp_sec"]),
                              track_id=int(row["track_id"]), class_name=row["class_name"],
                              x1=float(row["x1"]), y1=float(row["y1"]),
                              x2=float(row["x2"]), y2=float(row["y2"]),
                              cx=float(row["center_x"]), cy=float(row["center_y"]))
            frames[obs.frame].append(obs)
    for frame in sorted(frames):
        seen = set()
        for obs in frames[frame]:
            if obs.track_id in seen:
                raise ValueError(f"Duplicate track {obs.track_id} in frame {frame}")
            seen.add(obs.track_id)
        yield frame, frames[frame]


def process(tracks_csv: Path, config: dict, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    history = TrackHistory(config["history_seconds"])
    events: dict[tuple[int,int], PairEvent] = {}
    last_emitted: dict[tuple[int,int], float] = {}
    candidates = []
    with (output_dir / "motion.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame","timestamp_sec","track_id","speed_before_px_s","speed_recent_px_s",
                         "speed_ratio","speed_drop","direction_change_deg","sharp_direction_change"])
        last_time = -float("inf")
        for frame, observations in load_frames(tracks_csv):
            now = observations[0].timestamp
            if now <= last_time:
                raise ValueError("Frame timestamps are not increasing; expected one constant-FPS video")
            last_time = now
            history.expire(now)
            per_vehicle = {}
            for obs in observations:
                history.add(obs)
                trail = history.get(obs.track_id)
                drop, ratio, before, after = speed_drop(trail, now, config)
                turning, angle = direction_change(trail, now, config)
                per_vehicle[obs.track_id] = {"speed_drop":drop,"direction_change":turning}
                writer.writerow([frame, f"{now:.6f}", obs.track_id, before, after,
                                 ratio, int(drop), angle, int(turning)])

            # Encounter evidence is sticky within each pair's short event window.
            for pair, close, overlap, region, distance, overlap_value in nearby_pairs(observations, config):
                event = events.get(pair)
                if event is None or now-event.last_contact > config["interaction_event_seconds"]:
                    event = PairEvent(ids=pair, first_contact=now, last_contact=now, region=region)
                    events[pair] = event
                event.last_contact = now
                event.close |= close
                event.overlap |= overlap
                event.region = [min(event.region[0],region[0]),min(event.region[1],region[1]),
                                max(event.region[2],region[2]),max(event.region[3],region[3])]
                event.speed_drop |= any(per_vehicle[k]["speed_drop"] for k in pair)
                event.direction_change |= any(per_vehicle[k]["direction_change"] for k in pair)

            for pair, event in list(events.items()):
                if now-event.last_contact > config["interaction_event_seconds"]:
                    del events[pair]
                    continue
                # A pair's motion can occur shortly after first contact, even when boxes cease overlapping.
                event.speed_drop |= any(per_vehicle.get(k,{}).get("speed_drop",False) for k in pair)
                event.direction_change |= any(per_vehicle.get(k,{}).get("direction_change",False) for k in pair)
                if not event.post_stop:
                    event.post_stop = any(stationary_after_interaction(history.get(k),event.first_contact,now,config)
                                          for k in pair)
                if event.emitted or event.score() < config["candidate_score_threshold"]:
                    continue
                if now-last_emitted.get(pair,-float("inf")) < config["event_cooldown_seconds"]:
                    event.emitted = True
                    continue
                candidates.append(event.candidate(now))
                event.emitted = True
                last_emitted[pair] = now

    with (output_dir / "candidates.json").open("w", encoding="utf-8") as file:
        json.dump(candidates,file,indent=2)
    return candidates


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, default=Path("outputs/tracks.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/motion.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase2"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.tracks.is_file():
        raise FileNotFoundError(f"Tracking CSV not found: {args.tracks}")
    with args.config.open(encoding="utf-8") as file:
        config = json.load(file)
    candidates = process(args.tracks, config, args.output_dir)
    print(f"Processed tracks: {args.tracks}")
    print(f"Motion measurements: {args.output_dir / 'motion.csv'}")
    print(f"Accident candidates: {args.output_dir / 'candidates.json'} ({len(candidates)} generated)")
    print("Reminder: candidates are unconfirmed motion anomalies, not accident predictions.")


if __name__ == "__main__":
    main()
