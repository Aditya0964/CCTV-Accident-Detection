"""Synthetic, dependency-free tests for Phase 2 motion estimation and scoring."""
import json
import tempfile
import unittest
from pathlib import Path

from src.motion.anomaly_scorer import PairEvent
from src.motion.detect_anomalies import process
from src.motion.interaction import iou, nearby_pairs
from src.motion.motion_rules import direction_change, speed_drop
from src.motion.trajectory import Observation, speed_samples

CONFIG = json.loads((Path(__file__).resolve().parents[1]/"configs/motion.json").read_text())


def obs(time, x, y=0, ident=1, frame=None):
    return Observation(frame=round(time*30) if frame is None else frame, timestamp=time,
                       track_id=ident,class_name="car",x1=x-10,y1=y-10,x2=x+10,y2=y+10,cx=x,cy=y)


class MotionTests(unittest.TestCase):
    def test_fps_normalized_speed(self):
        for fps in (10, 15, 30, 60):
            sequence = [obs(i/fps,300*i/fps) for i in range(fps+1)]
            values = speed_samples(sequence, CONFIG["max_sample_gap_seconds"])
            self.assertTrue(values)
            for _, speed in values:
                self.assertAlmostEqual(speed, 300, places=5)

    def test_speed_drop(self):
        history = []
        for n in range(41):
            t=n*0.05
            x=100*t if t<=1.0 else 100+5*(t-1.0)
            history.append(obs(t,x))
        detected, ratio, previous, recent = speed_drop(history, 1.5, CONFIG)
        self.assertTrue(detected, (ratio, previous, recent))

    def test_direction(self):
        sequence = [obs(0,0,0),obs(0.25,20,0),obs(0.5,20,20)]
        detected, angle = direction_change(sequence,0.5,CONFIG)
        self.assertTrue(detected)
        self.assertAlmostEqual(angle,90)

    def test_overlap(self):
        self.assertAlmostEqual(iou((0,0,10,10),(0,0,10,10)),1)
        self.assertEqual(iou((0,0,10,10),(20,20,30,30)),0)
        self.assertEqual(len(list(nearby_pairs([obs(0,0),obs(0,300,ident=2)],CONFIG))),0)

    def test_score(self):
        event=PairEvent((1,2),1,1,[0,0,10,10],speed_drop=True,close=True,overlap=True)
        self.assertEqual(event.score(),5)

    def test_csv_to_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            csv_path=folder/"tracks.csv"
            lines=["frame,timestamp_sec,track_id,class_name,x1,y1,x2,y2,center_x,center_y"]
            for idx in range(41):
                t=idx*.05
                x=100*t if t<=1 else 100+5*(t-1)
                # stationary second car: close only once moving car reaches its position
                for ident, xx in [(1,x),(2,105)]:
                    lines.append(f"{idx},{t:.6f},{ident},car,{xx-10},-10,{xx+10},10,{xx},0")
            csv_path.write_text("\n".join(lines),encoding="utf-8")
            result=process(csv_path,CONFIG,folder/"out")
            self.assertTrue(result, "Expected a candidate from close encounter + speed drop")
            self.assertGreaterEqual(result[0]["anomaly_score"],5)
            self.assertTrue((folder/"out"/"motion.csv").exists())


if __name__ == "__main__":
    unittest.main()
