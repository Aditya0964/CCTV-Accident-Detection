# CCTV Accident Detection — Phase 1

This starter implements input video processing, pretrained YOLO vehicle detection, ByteTrack association, annotated output and track-data export. It does **not** yet detect accidents.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you have an NVIDIA GPU, consult the PyTorch official install selector for the correct CUDA-enabled PyTorch build. Check with:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

## Run

Place a test CCTV video at `data/raw/test_cctv.mp4`, then from the repository root run:

```powershell
python -m src.track_vehicles --input data/raw/test_cctv.mp4
```

The first run may download pretrained YOLO weights. Output:

- `outputs/tracked_vehicles.mp4` — annotated frames with boxes, IDs, trails
- `outputs/tracks.csv` — frame-indexed detections and trajectory center positions

## Notes

- This uses a YOLO11 nano detector (`yolo11n.pt`), with COCO vehicle class IDs 2, 3, 5, and 7. This is a starting model, not a claim that it is the best choice.
- Timestamps are computed from frame index / source FPS. This assumes a fixed-frame-rate source; variable-frame-rate files require timestamp-aware decoding.
- Pixel coordinates are on the **resized** frame. They are not world speeds or distances.
- For now, each video should be run as a separate process to reset tracker state.
- CV tracking can fail due to occlusions, camera motion, small objects or ID switches. Inspect tracks before designing accident rules.
- The initial CSV only includes currently returned tracked detections, not interpolations for lost tracks.

## Next phase

Compute speed drop, heading changes, vehicle proximity/overlap and stopping from `tracks.csv`, then score candidate accident events.

# Phase 2 — Motion anomaly detection

## What it does

Reads `outputs/tracks.csv` from Phase 1 and produces motion measurements and vehicle-pair accident **candidates**. It does not declare an accident; Phase 3/4 will classify candidates using a temporal video model.

Modules:
- `trajectory.py`: per-track timestamp-based history and pixel/second speeds.
- `motion_rules.py`: relative speed drops, direction changes, post-interaction stopping.
- `interaction.py`: normalized vehicle proximity and bounding-box intersection-over-union.
- `anomaly_scorer.py`: weighted evidence and candidate format.
- `detect_anomalies.py`: offline CSV runner and outputs.
- `configs/motion.json`: adjustable thresholds; start values only.

## How to install into existing project

Copy the `src/motion/` directory, `configs/motion.json`, and `tests/test_motion.py` into the matching directories of your existing Phase 1 repository. No additional packages are needed (standard Python library only).

From your project's root with `.venv` active:

```powershell
python -m unittest discover -s tests -p "test_motion.py" -v
python -m src.motion.detect_anomalies --tracks outputs/tracks.csv --config configs/motion.json
```

Outputs (ignored by existing `.gitignore`):
- `outputs/phase2/motion.csv`: per-vehicle speed estimates, ratios and rule indicators.
- `outputs/phase2/candidates.json`: timestamp, candidate IDs, score, region and triggering rules.

## Engineering limitations and validation

- **No candidates is not necessarily a bug.** The video may have no suitable close-vehicle event, and this stage requires sufficient reliable tracking observations.
- `speed_drop_ratio` uses successive 0.5-second windows (with interval-midpoint samples); this is pixel-space motion, not physical km/h.
- Direction changes use three positions approximately 0.25 seconds apart. Small displacements are suppressed to reduce detector jitter.
- `post_stop` requires evidence of motion before first contact and near-stationary motion afterwards; pair events last for up to 2 seconds after the last close contact.
- Proximity/overlap can occur through perspective, without real collision. A threshold score is a **candidate only**.
- The initial implementation targets accidents involving two tracked vehicles, not isolated single-vehicle accidents, rollovers, or pedestrians.
- `tracks.csv` records tracked detections only (no empty-frame rows). Actual observation time differences are used, but video remains assumed constant frame rate as in Phase 1.
- Tracks from different videos must be processed independently; track IDs may be reused across runs.
- Rule scores are heuristic and must be tuned on labeled accident and normal clips. Evaluate event recall and false candidate rate; do not present them as validated accident predictions.
- The history manager, as implemented, removes stale history based on wall-video timestamps; a missing track stays available briefly for event follow-up.

## Practical review

For an event in `candidates.json`, locate `candidate_timestamp_sec` in `outputs/tracked_vehicles.mp4` and visually verify what happened. Use `decision_timestamp_sec` to see when enough evidence accumulated. The surrounding video is not yet extracted automatically; that belongs to later integration.
