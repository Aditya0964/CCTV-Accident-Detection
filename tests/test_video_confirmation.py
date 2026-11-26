"""Unit and integration tests for Phase 3 video confirmation and event management."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from src.confirm_accidents import parse_args as parse_confirm_args
from src.confirm_accidents import run as run_confirm
from src.evaluate_classifier import compute_binary_metrics
from src.evaluate_classifier import parse_args as parse_eval_args
from src.evaluate_classifier import run as run_eval
from src.event_manager import AccidentEventManager
from src.scad_dataset import _infer_video_group, discover_scad_samples, sample_video_frames
from src.train_classifier import parse_args as parse_train_args
from src.train_classifier import run as run_train
from src.video_classifier import (
    ModelConfig,
    build_video_classifier,
    load_checkpoint,
    save_checkpoint,
)


def _write_test_clip(
    path: Path,
    num_frames: int = 20,
    fps: float = 10.0,
    width: int = 64,
    height: int = 64,
    color: tuple[int, int, int] = (40, 120, 200),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create synthetic clip: {path}")
    try:
        for i in range(num_frames):
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            cv2.circle(frame, ((i * 3) % width, height // 2), 6, (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()


def _write_frame_dir(
    folder: Path,
    num_frames: int = 8,
    width: int = 64,
    height: int = 64,
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 20 % 255, 100, 150), dtype=np.uint8)
        cv2.imwrite(str(folder / f"frame_{i:04d}.jpg"), frame)


class TestSCADDatasetAndModels(unittest.TestCase):
    def test_numbered_clips_share_source_group(self) -> None:
        self.assertEqual(_infer_video_group(Path("v51_1.mp4")), "v51")
        self.assertEqual(_infer_video_group(Path("v51_5.mp4")), "v51")
        self.assertEqual(_infer_video_group(Path("000065_2.mp4")), "000065")
        self.assertEqual(_infer_video_group(Path("RoadAccidents113_x264_4 (1).mp4")), "roadaccidents113_x264")

    def test_group_split_prevents_clip_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for g in range(6):
                _write_test_clip(root / "accident" / f"scene{g}_clip_1.mp4", num_frames=6)
                _write_test_clip(root / "accident" / f"scene{g}_clip_2.mp4", num_frames=6)
                _write_test_clip(root / "normal" / f"norm{g}_clip_1.mp4", num_frames=6)
                _write_test_clip(root / "normal" / f"norm{g}_clip_2.mp4", num_frames=6)

            splits = discover_scad_samples(data_root=root, val_ratio=0.2, test_ratio=0.2, seed=42)
            self.assertTrue(splits["train"] and splits["val"] and splits["test"])

            groups_by_split = {
                sp: {s.video_group for s in samples} for sp, samples in splits.items()
            }
            self.assertFalse(groups_by_split["train"] & groups_by_split["val"])
            self.assertFalse(groups_by_split["train"] & groups_by_split["test"])
            self.assertFalse(groups_by_split["val"] & groups_by_split["test"])

    def test_mixed_label_recording_stays_in_one_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for g in range(6):
                _write_test_clip(root / "accident" / f"scene{g}_clip_1.mp4", num_frames=6)
                _write_test_clip(root / "normal" / f"scene{g}_clip_2.mp4", num_frames=6)
            splits = discover_scad_samples(data_root=root, val_ratio=0.2, test_ratio=0.2)
            group_splits = {
                group: {sp for sp, samples in splits.items() if any(s.video_group == group for s in samples)}
                for group in (f"scene{g}" for g in range(6))
            }
            self.assertTrue(all(len(owners) == 1 for owners in group_splits.values()))

    def test_manifest_rejects_group_in_multiple_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["path", "label", "split", "video_group"])
                writer.writerow(["one.mp4", "accident", "train", "same_source"])
                writer.writerow(["two.mp4", "normal", "test", "same_source"])
            with self.assertRaisesRegex(ValueError, "same_source"):
                discover_scad_samples(data_root=root, manifest_csv=manifest)

    def test_manifest_and_frame_dir_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            short_vid = root / "short.mp4"
            _write_test_clip(short_vid, num_frames=5)
            frame_dir = root / "frames_clip"
            _write_frame_dir(frame_dir, num_frames=20)

            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["path", "label", "split", "video_group"])
                writer.writerow(["short.mp4", "accident", "train", "g1"])
                writer.writerow(["frames_clip", "normal", "test", "g2"])

            splits = discover_scad_samples(data_root=root, manifest_csv=manifest)
            self.assertEqual(len(splits["train"]), 1)
            self.assertEqual(len(splits["test"]), 1)

            t_short = sample_video_frames(short_vid, num_frames=16, input_size=64, augment=False)
            t_dir = sample_video_frames(frame_dir, num_frames=16, input_size=64, augment=True)
            self.assertEqual(tuple(t_short.shape), (16, 3, 64, 64))
            self.assertEqual(tuple(t_dir.shape), (16, 3, 64, 64))

    def test_model_architectures_and_checkpoint_roundtrip(self) -> None:
        clips = torch.randn(2, 4, 3, 64, 64)
        for arch in ("mean_pool", "cnn_lstm"):
            for backbone in ("resnet18", "mobilenet_v3_small"):
                cfg = ModelConfig(
                    arch=arch,
                    backbone=backbone,
                    num_frames=4,
                    input_size=64,
                    hidden_dim=32,
                )
                model = build_video_classifier(cfg, pretrained=False)
                model.eval()
                logits = model(clips)
                probs = model.predict_proba(clips)
                self.assertEqual(tuple(logits.shape), (2,))
                self.assertTrue(torch.all((probs >= 0.0) & (probs <= 1.0)))

                with tempfile.TemporaryDirectory() as tmp:
                    ckpt = Path(tmp) / "model.pt"
                    save_checkpoint(ckpt, model, cfg)
                    restored, restored_cfg, _ = load_checkpoint(ckpt, device="cpu")
                    self.assertEqual(restored_cfg, cfg)
                    restored_probs = restored.predict_proba(clips)
                    self.assertTrue(torch.allclose(probs, restored_probs, atol=1e-6))

                model.set_backbone_trainable(False)
                self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))


class TestTrainingEvaluationAndConfirmation(unittest.TestCase):
    def test_training_requires_separate_validation_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["path", "label", "split", "video_group"])
                writer.writerow(["train.mp4", "accident", "train", "train_source"])
                writer.writerow(["test.mp4", "normal", "test", "test_source"])
            with self.assertRaisesRegex(RuntimeError, "No validation samples"):
                run_train(parse_train_args(["--manifest", str(manifest), "--device", "cpu"]))

    def test_compute_binary_metrics_exact(self) -> None:
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.4, 0.8, 0.1, 0.7, 0.2])
        metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["tn"], 2)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["confusion_matrix"], [[2, 1], [1, 2]])
        self.assertAlmostEqual(metrics["precision"], 2 / 3, places=4)
        self.assertAlmostEqual(metrics["recall"], 2 / 3, places=4)

    def test_event_manager_cooldown_deduplication(self) -> None:
        mgr = AccidentEventManager(cooldown_sec=5.0, region_iou_threshold=0.10)
        c1 = {
            "candidate_id": 1,
            "timestamp_sec": 10.0,
            "frame_index": 300,
            "involved_track_ids": [4, 9],
            "involved_classes": ["car", "truck"],
            "candidate_region": [100.0, 100.0, 220.0, 200.0],
            "motion_score": 7,
        }
        c2_dup = {
            "candidate_id": 2,
            "timestamp_sec": 13.5,
            "frame_index": 405,
            "involved_track_ids": [4, 9],
            "involved_classes": ["car", "truck"],
            "candidate_region": [105.0, 102.0, 225.0, 202.0],
            "motion_score": 8,
        }
        c3_later = {
            "candidate_id": 3,
            "timestamp_sec": 20.0,
            "frame_index": 600,
            "involved_track_ids": [4],
            "involved_classes": ["car"],
            "candidate_region": [100.0, 100.0, 220.0, 200.0],
            "motion_score": 6,
        }
        c4_distinct = {
            "candidate_id": 4,
            "timestamp_sec": 12.0,
            "frame_index": 360,
            "involved_track_ids": [15],
            "involved_classes": ["bus"],
            "candidate_region": [500.0, 300.0, 620.0, 400.0],
            "motion_score": 6,
        }

        inc1, new1 = mgr.register_confirmed(c1, 0.78)
        inc2, new2 = mgr.register_confirmed(c2_dup, 0.91)
        inc3, new3 = mgr.register_confirmed(c3_later, 0.85)
        inc4, new4 = mgr.register_confirmed(c4_distinct, 0.88)

        self.assertTrue(new1)
        self.assertFalse(new2)
        self.assertIs(inc1, inc2)
        self.assertAlmostEqual(inc1.accident_confidence, 0.91, places=4)
        self.assertEqual(inc1.candidate_ids, [1, 2])
        self.assertTrue(new3)
        self.assertTrue(new4)
        self.assertEqual(len(mgr.incidents), 3)
        self.assertEqual([i.incident_id for i in mgr.incidents], ["001", "002", "003"])

    def test_train_eval_and_confirm_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            scad_dir = tmp_dir / "scad"
            for sp in ("train", "val", "test"):
                for i in range(2):
                    _write_test_clip(
                        scad_dir / sp / "accident" / f"{sp}_acc_{i}.mp4",
                        num_frames=8,
                        color=(0, 0, 220),
                    )
                    _write_test_clip(
                        scad_dir / sp / "normal" / f"{sp}_norm_{i}.mp4",
                        num_frames=8,
                        color=(0, 200, 0),
                    )

            baseline_dir = tmp_dir / "models" / "baseline"
            lstm_dir = tmp_dir / "models" / "cnn_lstm"

            run_train(
                parse_train_args(
                    [
                        "--data-dir",
                        str(scad_dir),
                        "--arch",
                        "mean_pool",
                        "--backbone",
                        "mobilenet_v3_small",
                        "--num-frames",
                        "4",
                        "--input-size",
                        "64",
                        "--epochs",
                        "2",
                        "--batch-size",
                        "2",
                        "--num-workers",
                        "0",
                        "--no-pretrained",
                        "--device",
                        "cpu",
                        "--save-dir",
                        str(baseline_dir),
                    ]
                )
            )
            run_train(
                parse_train_args(
                    [
                        "--data-dir",
                        str(scad_dir),
                        "--arch",
                        "cnn_lstm",
                        "--backbone",
                        "mobilenet_v3_small",
                        "--num-frames",
                        "4",
                        "--input-size",
                        "64",
                        "--hidden-dim",
                        "32",
                        "--epochs",
                        "2",
                        "--batch-size",
                        "2",
                        "--num-workers",
                        "0",
                        "--no-pretrained",
                        "--device",
                        "cpu",
                        "--save-dir",
                        str(lstm_dir),
                    ]
                )
            )

            eval_dir = tmp_dir / "eval"
            report = run_eval(
                parse_eval_args(
                    [
                        "--data-dir",
                        str(scad_dir),
                        "--split",
                        "test",
                        "--checkpoints",
                        str(baseline_dir / "best.pt"),
                        str(lstm_dir / "best.pt"),
                        "--num-workers",
                        "0",
                        "--device",
                        "cpu",
                        "--sweep-thresholds",
                        "--output-dir",
                        str(eval_dir),
                    ]
                )
            )
            self.assertEqual(len(report["models"]), 2)
            self.assertTrue((eval_dir / "experiment1_metrics.json").is_file())
            for entry in report["models"]:
                self.assertTrue(Path(entry["confusion_matrix_plot"]).is_file())
                with Path(entry["predictions_csv"]).open(newline="", encoding="utf-8") as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 4)

            # Test Phase 3 confirmation + cooldown + Experiment 3 + preview video.
            full_video = tmp_dir / "cctv.mp4"
            _write_test_clip(full_video, num_frames=50, fps=10.0, width=320, height=240)
            cand1_clip = tmp_dir / "cand_1.mp4"
            _write_test_clip(cand1_clip, num_frames=16, fps=10.0, width=320, height=240)

            candidates_json = tmp_dir / "candidates.json"
            candidates_json.write_text(
                json.dumps(
                    [
                        {
                            "candidate_id": 1,
                            "timestamp_sec": 1.5,
                            "frame_index": 15,
                            "clip_start_sec": 0.0,
                            "clip_end_sec": 3.5,
                            "involved_track_ids": [4, 9],
                            "involved_classes": ["car", "truck"],
                            "candidate_region": [60.0, 50.0, 180.0, 140.0],
                            "motion_score": 7,
                            "clip_path": str(cand1_clip),
                        },
                        {
                            "candidate_id": 2,
                            "timestamp_sec": 3.2,
                            "frame_index": 32,
                            "clip_start_sec": 1.2,
                            "clip_end_sec": 4.9,
                            "involved_track_ids": [4, 9],
                            "involved_classes": ["car", "truck"],
                            "candidate_region": [65.0, 52.0, 185.0, 142.0],
                            "motion_score": 6,
                        },
                    ]
                ),
                encoding="utf-8",
            )
            gt_json = tmp_dir / "gt.json"
            gt_json.write_text(json.dumps([{"timestamp_sec": 1.8}]), encoding="utf-8")

            confirm_out = tmp_dir / "confirm_out"
            confirm_res = run_confirm(
                parse_confirm_args(
                    [
                        "--candidates",
                        str(candidates_json),
                        "--checkpoint",
                        str(lstm_dir / "best.pt"),
                        "--threshold",
                        "0.0",
                        "--video",
                        str(full_video),
                        "--width",
                        "320",
                        "--render-video",
                        "--ground-truth",
                        str(gt_json),
                        "--device",
                        "cpu",
                        "--output-dir",
                        str(confirm_out),
                    ]
                )
            )
            self.assertEqual(len(confirm_res["confirmation_results"]), 2)
            # Both share vehicles/region within 5s, so EventManager deduplicates into 1 incident.
            self.assertEqual(len(confirm_res["confirmed_incidents"]), 1)
            self.assertTrue((confirm_out / "confirmed_incidents.json").is_file())
            self.assertTrue((confirm_out / "confirmation_results.json").is_file())
            self.assertTrue((confirm_out / "experiment3_comparison.json").is_file())
            self.assertTrue((confirm_out / "confirmed_preview.mp4").is_file())

            # Threshold > 1.0 rejects all candidates cleanly.
            reject_out = tmp_dir / "reject_out"
            reject_res = run_confirm(
                parse_confirm_args(
                    [
                        "--candidates",
                        str(candidates_json),
                        "--checkpoint",
                        str(lstm_dir / "best.pt"),
                        "--threshold",
                        "1.0",
                        "--video",
                        str(full_video),
                        "--device",
                        "cpu",
                        "--output-dir",
                        str(reject_out),
                    ]
                )
            )
            self.assertEqual(reject_res["confirmed_incidents"], [])
            self.assertTrue(all(not r["confirmed"] for r in reject_res["confirmation_results"]))


if __name__ == "__main__":
    unittest.main()
