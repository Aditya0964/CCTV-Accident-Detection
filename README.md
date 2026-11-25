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

## Next phase (Phase 2 — Motion Anomaly Detection)

Phase 2 reads `outputs/tracks.csv` to compute speed drop (Rule A), sharp direction changes (Rule B), size-normalized vehicle proximity & IoU, post-interaction stopping, and weighted motion anomaly scores (`0–9`), grouping peaks into accident candidates and optionally extracting `[t - 2s, t + 2s]` video clips.

Run offline scoring from `tracks.csv`:

```bash
python -m src.detect_anomalies --tracks outputs/tracks.csv
```

With source video to extract candidate clips (`outputs/candidates/candidate_XXX.mp4` + `.jpg`) and render a debug preview (`outputs/anomaly_preview.mp4`):

```bash
python -m src.detect_anomalies \
  --tracks outputs/tracks.csv \
  --video data/raw/test_cctv.mp4 \
  --extract-clips \
  --render-video
```

Key flags:
- `--speed-drop-ratio` (default `0.25`, testable e.g. `0.15`, `0.25`, `0.40`)
- `--angle-deg` (default `50.0`)
- `--mode` (`full` for all 5 indicators / max score 9; `motion_only` for speed + direction ablation)
- `--candidate-threshold` (default `5` in `full` mode, `4` in `motion_only` mode)

Outputs:
- `outputs/motion_scores.csv` — per-frame, per-track kinematic & interaction metrics and score (`0–9`)
- `outputs/candidates.json` — deduplicated accident candidates (`timestamp_sec`, `involved_track_ids`, `candidate_region`, `motion_score`, clip bounds)
- `outputs/candidates/candidate_XXX.mp4` & `.jpg` — extracted `[t - 2s, t + 2s]` temporal clips and keyframes (when `--extract-clips` is set)
- `outputs/anomaly_preview.mp4` — annotated anomaly debug video (when `--render-video` is set)

## Tests

```bash
python -m unittest discover -s tests -v
```
