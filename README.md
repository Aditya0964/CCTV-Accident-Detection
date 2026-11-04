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
