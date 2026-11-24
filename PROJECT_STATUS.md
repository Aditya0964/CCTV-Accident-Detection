# Project status

As of 23 November 2026, this repository has **implemented code for vehicle tracking, motion-based accident candidates, and Phase 3 classifier-input windows**. It does not yet confirm accidents or assess fire and severity.

## Implemented

| Stage | Current implementation | Output |
| --- | --- | --- |
| Phase 1: vehicle detection and tracking | `src/track_vehicles.py` reads a video, detects cars, motorcycles, buses and trucks with pretrained YOLO11 nano, and assigns track IDs with ByteTrack. It draws boxes and trails. | `outputs/tracked_vehicles.mp4` and `outputs/tracks.csv` |
| Phase 2: motion anomaly candidates | `src/motion/` reads the track CSV, estimates pixel-space speed and heading changes, checks vehicle proximity/box overlap and post-interaction stopping, and scores vehicle-pair encounters. Thresholds are in `configs/motion.json`. | `outputs/phase2/motion.csv` and `outputs/phase2/candidates.json` |
| Phase 3 preparation: classifier windows | `src/phase3/` exports overlapping full-frame windows, plus extra windows around Phase 2 candidates. Full-frame coverage works even when tracking emits no candidates. | `outputs/phase3/<video>/window_*.mp4` and `manifest.json` |

The presentation described Lucas–Kanade optical-flow tracking. This repository uses ByteTrack for association instead. It also uses YOLO11 nano for vehicle detection rather than the presentation's YOLOv8. These are implementation changes to the detection/tracking stages; they do not add accident confirmation.

## Test material and verification

Three short accident previews from the [ACCIDENT benchmark gallery](https://accidentbench.github.io/index.html) and additional user-provided accident clips are in `tests/clips/`:

| File | Gallery category | Duration | Original preview |
| --- | --- | ---: | --- |
| `accident_tbone.mp4` | T-bone | 5.96 s | [Source](https://accidentbench.github.io/assets/videos/real-world/tbone%20%283%29.mp4) |
| `accident_rearend.mp4` | Rear-end | 5.60 s | [Source](https://accidentbench.github.io/assets/videos/real-world/rearend%20%283%29.mp4) |
| `accident_sideswipe.mp4` | Sideswipe | 8.01 s | [Source](https://accidentbench.github.io/assets/videos/real-world/sideswipe.mp4) |

These are source-labeled examples, not a measured test set. The folder currently has 87 `.mp4` names: 85 decodable videos and two Git LFS pointer files (`54_1.mp4` and `74_2.mp4`) that need their actual video data. No normal-traffic clip or ground-truth accident timestamps are included yet. Check the benchmark's terms before redistributing its footage outside this project.

The nine unit tests pass with `python3 -m unittest discover -s tests -v`. Six cover Phase 2 synthetic trajectories, overlap, scoring and CSV-to-candidate processing; three check Phase 3 window coverage. `accident_tbone.mp4` completed Phases 1 and 2: 149 frames, 772 track observations and nine **unconfirmed** motion candidates. `000028_1.mp4` also completed those stages: 120 frames and three unconfirmed candidates. Phase 3 exported six 100-frame windows covering the T-bone video and one full-frame window for `000028_1.mp4`. These checks do not establish accident-detection accuracy or false-negative rate.

## Still to build

1. **Phase 3: accident confirmation.** Connect a trained accident/normal video classifier to the prepared windows, then group repeated detections into incidents with a cooldown. The windows are implemented; classification is not.
2. **Fire/smoke and severity.** Analyze confirmed incident regions for fire or smoke, then estimate severity from validated features. This depends on labeled data and evaluation.
3. **Video-based evaluation.** Test accident and normal clips, record event recall, false alarms and latency, and tune thresholds. The current motion settings are starting values, not validated parameters.

## Run the current stages

For a fresh Linux CPU environment, install PyTorch's CPU wheels first to avoid downloading the much larger CUDA dependency set, then install this project's requirements:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --timeout 120 --retries 8 torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --timeout 120 --retries 8 -r requirements.txt
```

From the repository root, run the stages (including ByteTrack's `lap` dependency from `requirements.txt`):

```bash
python -m src.track_vehicles --input tests/clips/accident_tbone.mp4
python -m src.motion.detect_anomalies --tracks outputs/tracks.csv --config configs/motion.json
python -m src.phase3.prepare_windows --input tests/clips/accident_tbone.mp4 --candidates outputs/phase2/candidates.json
```

Run each video separately so ByteTrack's state and output files do not mix across clips. The Phase 2 output is a list of **candidates**; the Phase 3 output is a set of classifier inputs, not confirmed accidents. The next-stage design and model choices are in `PIPELINE.md`.
