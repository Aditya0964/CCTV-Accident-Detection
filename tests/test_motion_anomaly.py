"""Unit tests for Phase 2 motion, interaction, stopping, and candidate scoring."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.motion_anomaly import (
    AnomalyConfig,
    TrackObservation,
    analyze_motion_anomalies,
    compute_iou,
    infer_fps,
    load_tracks_csv,
    write_candidates_json,
    write_motion_scores_csv,
)


def _make_obs(
    frame: int,
    fps: float,
    track_id: int,
    cx: float,
    cy: float,
    w: float = 60.0,
    h: float = 40.0,
    class_id: int = 2,
    class_name: str = "car",
    confidence: float = 0.9,
) -> TrackObservation:
    return TrackObservation(
        frame=frame,
        timestamp_sec=round(frame / fps, 3),
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        x1=cx - w / 2.0,
        y1=cy - h / 2.0,
        x2=cx + w / 2.0,
        y2=cy + h / 2.0,
        center_x=cx,
        center_y=cy,
    )


class TestMotionAnomalyRules(unittest.TestCase):
    def test_occluded_contact_survives_a_merged_detection(self) -> None:
        fps = 28.0
        config = AnomalyConfig(fps=fps, preserve_moving_contacts=True)

        def observations(*, brakes=True, merged=True):
            rows = []
            for frame in range(22, 55):
                offset = frame - 22
                if frame <= 33:
                    x = 70 + 11 * min(offset, 7) + max(0, offset - 7) if brakes else 70 + 10 * offset
                    rows.append(_make_obs(frame, fps, 1, x, 343, w=118 + 7 * offset, h=175,
                                          class_id=5, class_name="bus"))
                if frame <= 40:
                    rows.append(_make_obs(frame, fps, 2, 254 + 14.35 * offset, 400 - 2.8 * offset,
                                          w=505 - 14 * offset, h=200, class_id=5, class_name="bus"))
                if merged and frame >= 33:
                    rows.append(_make_obs(frame, fps, 3, 285 + 9 * (frame - 33), 356,
                                          w=500, h=210, class_id=5, class_name="bus"))
            return rows

        def contacts(rows):
            return [candidate for candidate in analyze_motion_anomalies(rows, config)[1]
                    if candidate.evidence.get("occluded_contact")]

        [contact] = contacts(observations())
        self.assertEqual(contact.frame_index, 33)
        self.assertEqual(contact.involved_track_ids, [1, 2])
        self.assertEqual(contact.metrics["replacement_track_id"], 3)
        self.assertEqual(contacts(observations(merged=False)), [])
        self.assertEqual(contacts(observations(brakes=False)), [])

    def test_single_vehicle_track_loss_requires_interior_motion_and_shape_change(self) -> None:
        fps = 25.0
        config = AnomalyConfig(fps=fps, tracking_size=(960, 540))

        def observations(*, deflect=True, x_offset=0, replacement=False):
            rows = []
            for frame in range(49):
                rows.append(_make_obs(frame, fps, 2, 180, 140))
                if frame <= 29:
                    elapsed = max(0, frame - 25)
                    rows.append(_make_obs(
                        frame, fps, 1, 530 + x_offset - (4 * elapsed if deflect else 0),
                        375 - 2.6 * frame, w=60 + (4 * elapsed if deflect else 0),
                        h=46 + (2 * elapsed if deflect else 0),
                    ))
                if replacement and 32 <= frame <= 40:
                    rows.append(_make_obs(frame, fps, 3, 513, 297, w=76, h=54))
            return rows

        [rollover] = analyze_motion_anomalies(observations(), config)[1]
        self.assertEqual(rollover.involved_track_ids, [1])
        self.assertEqual(rollover.frame_index, 29)
        self.assertTrue(rollover.evidence["single_vehicle_track_loss"])
        for rows in (observations(deflect=False), observations(x_offset=-470), observations(replacement=True)):
            with self.subTest(case=rows[-1]):
                candidates = analyze_motion_anomalies(rows, config)[1]
                self.assertFalse(any(item.evidence.get("single_vehicle_track_loss") for item in candidates))

    def test_first_contact_is_preserved_when_both_cars_continue_moving(self) -> None:
        fps = 15.0
        config = AnomalyConfig(fps=fps, preserve_moving_contacts=True)

        def trajectories(deflect):
            samples = []
            for frame in range(24):
                samples.append(_make_obs(frame, fps, 1, 100 + 24 * frame, 200, w=100, h=45))
                if not deflect:
                    x, y = 420 + 3 * frame, 215
                elif frame < 11:
                    x, y = 420 + 3 * frame, 270 - 5 * frame
                else:
                    x, y = 450 + 8 * (frame - 10), 220 + frame - 10
                samples.append(_make_obs(frame, fps, 2, x, y, w=100, h=45))
            return samples

        _, candidates = analyze_motion_anomalies(trajectories(deflect=True), config)
        [contact] = candidates
        self.assertEqual(contact.frame_index, 12)
        self.assertEqual(contact.involved_track_ids, [1, 2])
        self.assertTrue(contact.evidence["moving_contact"])
        self.assertFalse(contact.evidence["post_interaction_stopping"])
        self.assertGreater(contact.metrics["pair_iou"], contact.metrics["initial_pair_iou"])

        _, ordinary_crossing = analyze_motion_anomalies(trajectories(deflect=False), config)
        self.assertEqual(ordinary_crossing, [])

    def test_iou_geometry(self) -> None:
        self.assertAlmostEqual(compute_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(compute_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(compute_iou((0, 0, 10, 10), (5, 0, 15, 10)), 50.0 / 150.0)

    def test_constant_motion_produces_no_anomalies(self) -> None:
        fps = 10.0
        obs = [_make_obs(f, fps, 1, cx=50.0 + f * 15.0, cy=200.0) for f in range(40)]
        scores, candidates = analyze_motion_anomalies(obs, AnomalyConfig(fps=fps))
        self.assertEqual(len(scores), 40)
        self.assertEqual(candidates, [])
        self.assertTrue(all(s.status == "normal" for s in scores))
        self.assertTrue(all(not s.sudden_speed_drop for s in scores))
        self.assertTrue(all(not s.sharp_direction_change for s in scores))

    def test_stationary_vehicle_with_jitter_does_not_trigger(self) -> None:
        fps = 10.0
        obs = [
            _make_obs(f, fps, 1, cx=300.0 + (1.0 if f % 2 == 0 else -1.0), cy=220.0)
            for f in range(40)
        ]
        scores, candidates = analyze_motion_anomalies(obs, AnomalyConfig(fps=fps))
        self.assertEqual(candidates, [])
        self.assertTrue(all(not s.sudden_speed_drop for s in scores))
        self.assertTrue(all(not s.sharp_direction_change for s in scores))
        self.assertTrue(all(not s.post_interaction_stopping for s in scores))

    def test_sudden_speed_drop_rule_a(self) -> None:
        fps = 10.0
        obs: list[TrackObservation] = []
        # Move at 150 px/s for 2.0s (frames 0..19), then stop abruptly (frames 20..35).
        x = 100.0
        for f in range(20):
            x += 15.0
            obs.append(_make_obs(f, fps, 1, cx=x, cy=200.0))
        for f in range(20, 36):
            obs.append(_make_obs(f, fps, 1, cx=x, cy=200.0))

        scores, candidates = analyze_motion_anomalies(obs, AnomalyConfig(fps=fps))
        drop_frames = [s.frame for s in scores if s.sudden_speed_drop]
        self.assertTrue(any(20 <= f <= 25 for f in drop_frames), f"Expected drop near 20, got {drop_frames}")
        # Single car braking calmly without direction change or collision should not emit >=5 candidate.
        self.assertEqual(candidates, [])

    def test_sharp_direction_change_rule_b(self) -> None:
        fps = 10.0
        obs: list[TrackObservation] = []
        # Move right along +x for frames 0..14, then turn sharply 90 degrees downward (+y).
        x, y = 100.0, 100.0
        for f in range(15):
            x += 15.0
            obs.append(_make_obs(f, fps, 1, cx=x, cy=y))
        for f in range(15, 25):
            y += 15.0
            obs.append(_make_obs(f, fps, 1, cx=x, cy=y))

        scores, _ = analyze_motion_anomalies(obs, AnomalyConfig(fps=fps))
        turn_frames = [s.frame for s in scores if s.sharp_direction_change]
        self.assertTrue(any(15 <= f <= 18 for f in turn_frames), f"Expected turn near 15, got {turn_frames}")
        max_angle = max((s.direction_change_deg or 0.0) for s in scores)
        self.assertGreater(max_angle, 80.0)

    def test_occlusion_gap_does_not_create_fake_speed_spike(self) -> None:
        fps = 10.0
        obs = [_make_obs(f, fps, 1, cx=100.0 + f * 10.0, cy=200.0) for f in range(10)]
        # Skip 12 frames (> max_frame_gap=5) and reappear 200 px ahead, then continue normal speed.
        for f in range(22, 35):
            obs.append(_make_obs(f, fps, 1, cx=300.0 + (f - 22) * 10.0, cy=200.0))

        scores, candidates = analyze_motion_anomalies(obs, AnomalyConfig(fps=fps))
        self.assertEqual(candidates, [])
        frame_22 = next(s for s in scores if s.frame == 22)
        self.assertEqual(frame_22.speed_px_s, 0.0)
        self.assertFalse(frame_22.sudden_speed_drop)

    def test_normalized_proximity_is_scale_invariant(self) -> None:
        fps = 10.0
        # Two approaching vehicles at 1x scale vs 2x scale.
        def run_scale(scale: float) -> list[float]:
            obs = []
            for f in range(15):
                obs.append(_make_obs(f, fps, 1, cx=(100 + f * 10) * scale, cy=200 * scale, w=60 * scale, h=40 * scale))
                obs.append(_make_obs(f, fps, 2, cx=(320 - f * 10) * scale, cy=200 * scale, w=60 * scale, h=40 * scale))
            scores, _ = analyze_motion_anomalies(obs, AnomalyConfig(fps=fps))
            return [s.normalized_distance for s in scores if s.track_id == 1 and s.normalized_distance is not None]

        d1 = run_scale(1.0)
        d2 = run_scale(2.0)
        self.assertEqual(len(d1), len(d2))
        for a, b in zip(d1, d2):
            self.assertAlmostEqual(a, b, places=5)

    def test_parallel_stationary_lanes_do_not_trigger_interaction_or_stopping(self) -> None:
        fps = 10.0
        # Two cars stopped side-by-side at a red light (center distance ~ 70 px, box 60x40 -> diag 72.1).
        obs = []
        for f in range(30):
            obs.append(_make_obs(f, fps, 1, cx=300.0, cy=200.0, w=60.0, h=40.0))
            obs.append(_make_obs(f, fps, 2, cx=300.0, cy=270.0, w=60.0, h=40.0))
        scores, candidates = analyze_motion_anomalies(obs, AnomalyConfig(fps=fps))
        self.assertEqual(candidates, [])
        self.assertTrue(all(not s.very_close_interaction for s in scores))
        self.assertTrue(all(not s.post_interaction_stopping for s in scores))

    def test_full_collision_scenario_emits_single_grouped_candidate(self) -> None:
        fps = 10.0
        obs: list[TrackObservation] = []
        # Frame 0..19: Vehicle 4 drives right toward intersection, Vehicle 9 drives down toward same point.
        # Collision around frame 20 at (300, 200), followed by deflection + stopping for frames 20..44.
        for f in range(20):
            obs.append(_make_obs(f, fps, 4, cx=100.0 + f * 10.0, cy=200.0, w=60.0, h=40.0, class_name="car"))
            obs.append(_make_obs(f, fps, 9, cx=300.0, cy=40.0 + f * 8.0, w=70.0, h=50.0, class_id=7, class_name="truck"))
        # Impact deflection on frame 20..21 then stopped at (305, 212) and (302, 202).
        obs.append(_make_obs(20, fps, 4, cx=300.0, cy=200.0, w=60.0, h=40.0, class_name="car"))
        obs.append(_make_obs(20, fps, 9, cx=302.0, cy=198.0, w=70.0, h=50.0, class_id=7, class_name="truck"))
        obs.append(_make_obs(21, fps, 4, cx=304.0, cy=212.0, w=60.0, h=40.0, class_name="car"))
        obs.append(_make_obs(21, fps, 9, cx=302.0, cy=202.0, w=70.0, h=50.0, class_id=7, class_name="truck"))
        for f in range(22, 45):
            obs.append(_make_obs(f, fps, 4, cx=305.0, cy=212.0, w=60.0, h=40.0, class_name="car"))
            obs.append(_make_obs(f, fps, 9, cx=302.0, cy=202.0, w=70.0, h=50.0, class_id=7, class_name="truck"))

        scores, candidates = analyze_motion_anomalies(obs, AnomalyConfig(fps=fps))
        self.assertEqual(len(candidates), 1)
        cand = candidates[0]
        self.assertGreaterEqual(cand.motion_score, 5)
        self.assertEqual(set(cand.involved_track_ids), {4, 9})
        self.assertTrue(cand.evidence["very_close_interaction"])
        self.assertTrue(cand.evidence["bounding_box_overlap"])
        self.assertTrue(cand.evidence["post_interaction_stopping"])
        self.assertTrue(cand.evidence["sudden_speed_drop"] or cand.evidence["sharp_direction_change"])
        self.assertAlmostEqual(cand.timestamp_sec, 2.0, delta=0.5)
        self.assertTrue(any(s.status == "accident_candidate" for s in scores))

    def test_motion_only_ablation_mode_ignores_interaction_in_score(self) -> None:
        fps = 10.0
        obs: list[TrackObservation] = []
        for f in range(20):
            obs.append(_make_obs(f, fps, 4, cx=100.0 + f * 10.0, cy=200.0))
            obs.append(_make_obs(f, fps, 9, cx=300.0, cy=40.0 + f * 8.0))
        for f in range(20, 40):
            obs.append(_make_obs(f, fps, 4, cx=300.0, cy=200.0))
            obs.append(_make_obs(f, fps, 9, cx=302.0, cy=200.0))

        cfg = AnomalyConfig(fps=fps, mode="motion_only")
        self.assertEqual(cfg.max_possible_score(), 4)
        self.assertEqual(cfg.effective_candidate_threshold(), 4)
        scores, _ = analyze_motion_anomalies(obs, cfg)
        self.assertTrue(all(s.motion_score <= 4 for s in scores))

    def test_tracks_csv_roundtrip_and_outputs(self) -> None:
        fps = 15.0
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            tracks_csv = tmp_dir / "tracks.csv"
            with tracks_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
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
                    ]
                )
                for f in range(25):
                    ts = f / fps
                    cx = 100.0 + f * 12.0 if f < 15 else 268.0
                    cy = 180.0
                    writer.writerow([f, f"{ts:.3f}", 1, 2, "car", "0.9100", cx - 30, cy - 20, cx + 30, cy + 20, cx, cy])

            loaded = load_tracks_csv(tracks_csv)
            self.assertEqual(len(loaded), 25)
            self.assertAlmostEqual(infer_fps(loaded), fps, places=1)

            scores, candidates = analyze_motion_anomalies(loaded)
            scores_out = tmp_dir / "motion_scores.csv"
            cands_out = tmp_dir / "candidates.json"
            write_motion_scores_csv(scores, scores_out)
            write_candidates_json(candidates, cands_out)
            self.assertTrue(scores_out.is_file())
            self.assertEqual(json.loads(cands_out.read_text(encoding="utf-8")), [])


if __name__ == "__main__":
    unittest.main()
