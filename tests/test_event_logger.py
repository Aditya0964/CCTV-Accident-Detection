"""Unit tests for Stage 12 AccidentEventLogger and src.log_events CLI."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.event_manager import AccidentEventLogger, AccidentEventManager
from src.log_events import parse_args as parse_log_args
from src.log_events import run as run_log_events


class TestAccidentEventLoggerAndCLI(unittest.TestCase):
    def test_realtime_logger_writes_log_jsonl_csv_and_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            src_clip = Path(tmp) / "candidate_001.mp4"
            src_img = Path(tmp) / "candidate_001.jpg"
            src_clip.write_bytes(b"fake-mp4-bytes")
            src_img.write_bytes(b"fake-jpg-bytes")

            logger = AccidentEventLogger(output_dir=out_dir, append=False, echo_console=False)
            mgr = AccidentEventManager(cooldown_sec=5.0, region_iou_threshold=0.10, logger=logger)

            c1 = {
                "candidate_id": 1,
                "timestamp_sec": 84.0,
                "frame_index": 2520,
                "involved_track_ids": [4, 9],
                "involved_classes": ["car", "truck"],
                "candidate_region": [100.0, 120.0, 260.0, 240.0],
                "motion_score": 7,
                "clip_path": str(src_clip),
                "keyframe_path": str(src_img),
            }
            c2_merged = {
                "candidate_id": 2,
                "timestamp_sec": 86.5,
                "frame_index": 2595,
                "involved_track_ids": [4, 9],
                "involved_classes": ["car", "truck"],
                "candidate_region": [105.0, 122.0, 265.0, 242.0],
                "motion_score": 8,
            }
            c3_rejected = {
                "candidate_id": 3,
                "timestamp_sec": 95.0,
                "frame_index": 2850,
                "involved_track_ids": [12],
                "involved_classes": ["bus"],
                "candidate_region": [400.0, 200.0, 520.0, 320.0],
                "motion_score": 5,
            }

            mgr.register_confirmed(c1, accident_confidence=0.84)
            mgr.register_confirmed(c2_merged, accident_confidence=0.91)
            mgr.record_rejected(c3_rejected, accident_confidence=0.18, threshold=0.50)

            exported = logger.write_incidents_summary(
                mgr.incidents, export_individual=True, copy_media=True
            )

            self.assertEqual(len(exported), 1)
            self.assertTrue((out_dir / "incidents" / "incident_001.json").is_file())
            self.assertTrue((out_dir / "incidents" / "incident_001.mp4").is_file())
            self.assertTrue((out_dir / "incidents" / "incident_001.jpg").is_file())

            log_text = (out_dir / "events.log").read_text(encoding="utf-8")
            self.assertIn("INCIDENT_CREATED | Incident ID: 001", log_text)
            self.assertIn("INCIDENT_MERGED | Incident ID: 001", log_text)
            self.assertIn("CANDIDATE_REJECTED", log_text)
            self.assertIn("Time: 00:01:24", log_text)
            self.assertIn("ID 4 - car", log_text)
            self.assertIn("ID 9 - truck", log_text)

            jsonl_lines = (out_dir / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(jsonl_lines), 3)
            parsed_events = [json.loads(line) for line in jsonl_lines]
            self.assertEqual(
                [e["event_type"] for e in parsed_events],
                ["INCIDENT_CREATED", "INCIDENT_MERGED", "CANDIDATE_REJECTED"],
            )

            with (out_dir / "incidents.csv").open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["incident_id"], "001")
            self.assertEqual(rows[0]["timestamp_hms"], "00:01:26")
            self.assertEqual(rows[0]["accident_confidence"], "0.9100")

    def test_log_events_cli_replays_confirmation_and_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            conf_json = tmp_dir / "confirmation_results.json"
            cands_json = tmp_dir / "candidates.json"
            out_dir = tmp_dir / "logged"

            cands_json.write_text(
                json.dumps(
                    [
                        {
                            "candidate_id": 1,
                            "timestamp_sec": 10.0,
                            "frame_index": 300,
                            "involved_track_ids": [2, 5],
                            "involved_classes": ["car", "motorcycle"],
                            "candidate_region": [50.0, 60.0, 150.0, 160.0],
                            "motion_score": 7,
                        },
                        {
                            "candidate_id": 2,
                            "timestamp_sec": 25.0,
                            "frame_index": 750,
                            "involved_track_ids": [8],
                            "involved_classes": ["truck"],
                            "candidate_region": [300.0, 100.0, 420.0, 220.0],
                            "motion_score": 5,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            conf_json.write_text(
                json.dumps(
                    [
                        {
                            "candidate_id": 1,
                            "timestamp_sec": 10.0,
                            "frame_index": 300,
                            "involved_track_ids": [2, 5],
                            "candidate_region": [50.0, 60.0, 150.0, 160.0],
                            "motion_score": 7,
                            "p_accident": 0.88,
                            "threshold": 0.5,
                            "confirmed": True,
                        },
                        {
                            "candidate_id": 2,
                            "timestamp_sec": 25.0,
                            "frame_index": 750,
                            "involved_track_ids": [8],
                            "candidate_region": [300.0, 100.0, 420.0, 220.0],
                            "motion_score": 5,
                            "p_accident": 0.21,
                            "threshold": 0.5,
                            "confirmed": False,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            res = run_log_events(
                parse_log_args(
                    [
                        "--input",
                        str(conf_json),
                        "--candidates",
                        str(cands_json),
                        "--output-dir",
                        str(out_dir),
                    ]
                )
            )
            self.assertEqual(res["logged_events_count"], 2)
            self.assertEqual(len(res["incidents"]), 1)
            self.assertEqual(
                res["incidents"][0]["vehicles"],
                [{"track_id": 2, "class_name": "car"}, {"track_id": 5, "class_name": "motorcycle"}],
            )

            # Also verify raw candidates.json mode with --accept-all-candidates + --append.
            res2 = run_log_events(
                parse_log_args(
                    [
                        "--input",
                        str(cands_json),
                        "--accept-all-candidates",
                        "--append",
                        "--output-dir",
                        str(out_dir),
                    ]
                )
            )
            self.assertEqual(len(res2["incidents"]), 2)
            lines = (out_dir / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 4)


if __name__ == "__main__":
    unittest.main()
