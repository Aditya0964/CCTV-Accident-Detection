# CCTV Accident Detection

The main demo is a validation-tuned fusion of a video classifier, an adapted image classifier, and an accident-box detector. On the labeled clips in `data/raw/`, it reached **67.5% accuracy on 166 held-out, source-disjoint test videos**. See [BENCHMARK.md](BENCHMARK.md) for the full confusion matrix, comparison to individual models, data audit, and limitations. The repository also contains the original vehicle-tracking and motion-candidate pipeline.

## Windows desktop review

Double-click `launch_demo.cmd`, select a short CCTV clip, and choose **Analyze video**. The default workflow runs vehicle tracking, motion candidate selection, pretrained ViT detection, and the Firewatch scan. The table shows the clip verdict, scan times, scope, fire and smoke findings, and visual severity. Click a scan row to seek to its timestamp and read the severity reason.

For a clean install, install `requirements.txt` and `requirements-phase4.txt`. Put vehicle weights at `outputs/models/yolo11n.pt` and the [Firewatch YOLO11s ONNX weights](https://github.com/leskoont/firewatch/releases/tag/v0.1.0) at `outputs/models/firewatch-yolo11s.onnx`. Run the pretrained ViT evaluation below to generate `outputs/eval/vit/vit_evaluation.json`, or pass an existing calibration with `launch_demo.cmd --calibration PATH`.

For the fusion path, launch `launch_demo.cmd --backend fusion`. This uses the fusion evaluation report, the fitted ViT and R3D-18 classifiers, and the YOLO11x weights referenced by their reports. Reproduce those assets with the benchmark commands below or copy the existing artifacts into their original paths. A different fusion report can be supplied with `--fusion-report PATH`.

Phase 4 scans up to five seconds after each localized ViT incident. For a positive clip-level verdict with no localized incident it uses an approximate scan time and a scene-wide scan. Fusion supplies a source-pixel box when the strongest detector score reaches 0.25. A scene-wide finding describes the visible scene and cannot establish which vehicle produced the hazard. Persistent fire gives **High** visual severity, persistent smoke gives **Elevated**, and strong tracked multi-vehicle motion can also give **Elevated**. The report explains its evidence and marks incomplete assessments **Unknown**.

The final preview shows a red **ACCIDENT** label near a localized contact. The ViT review requires a candidate score above 0.80, approaching track overlap, and abrupt braking or deflection followed by stopping. The 0.80 review cutoff is a conservative screening setting. The clip verdict still uses its evaluation report's cutoff. Fusion requires a positive clip verdict and a detector box score of at least 0.80 for the red label.

Each run is saved under `outputs/desktop_runs` with the Phase 3 results and rejection reasons, hazard JSON and CSV, the final video preview, and a pipeline log. Use **Open saved analysis** to reopen one of those folders.

To run Phase 4 on existing Phase 3 output:

```powershell
python -m src.run_phase4 --video 'C:/videos/example.mp4' --clip-result outputs/vit_pipeline/example/vit/example_result.json --incidents outputs/vit_pipeline/example/confirmed_incidents.json --output-dir outputs/phase4/example
```

## Recommended demo

Put labeled videos in `data/raw/accident/` and `data/raw/normal/`, install the dependencies below, and from the repository root run:

```bash
python -m src.download_models
python -m src.fusion_workflow --video data/raw/accident/000831_3.mp4 --output-dir outputs/demo/accident
python -m src.fusion_workflow --video data/raw/normal/000831_1.mp4 --output-dir outputs/demo/normal
```

Each command writes a JSON clip verdict and a contact sheet of four sampled frames with accident boxes. The decision is based on the validation-selected fusion score; the box score and fusion score are not calibrated accident probabilities. To reproduce the benchmark, run `python -m src.prepare_benchmark`, then `python -m src.vit_probe`, `python -m src.r3d_probe`, `python -m src.yolo_accident_workflow`, and `python -m src.fusion_workflow`. The last command writes the selected rule and held-out metrics to `outputs/eval/fusion/fusion_evaluation.json`.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On a CPU-only Linux machine, install the tested CPU PyTorch wheels before the remaining requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu torchvision==0.20.1+cpu
python -m pip install -r requirements.txt
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

## Phase 3 — Video Accident Confirmation & Event Manager

Phase 3 trains and evaluates binary video classifiers on SCAD clips (`mean_pool` baseline and `cnn_lstm` temporal model), confirms Phase 2 candidates ($P_{\text{acc}} > T_c$), and deduplicates confirmed incidents using a 5-second cooldown event manager.

### Fast pretrained ViT workflow

The [tiya1012 accident-image ViT](https://huggingface.co/tiya1012/vit-accident-image) is a still-image classifier (Apache 2.0). This baseline samples eight frames from each four-second clip and reached 56.0% accuracy on the held-out videos. Use the fusion workflow above for the recommended demo. The model card's reported image accuracy is not this project's video accuracy.

```bash
python -m src.prepare_benchmark --data-dir data/raw
python -m src.vit_workflow evaluate
python -m src.vit_workflow predict --video data/raw/accident/example.mp4
python -m src.run_vit_pipeline --video data/raw/accident/example.mp4 --output-dir outputs/demo_example
```

Put source clips under `data/raw/accident/` and `data/raw/normal/`. `prepare_benchmark` ignores unreadable videos, Git LFS pointer files, and exact duplicates; it keeps clips from one original recording together in `data/benchmark_manifest.csv`. The evaluation writes validation and test predictions plus `outputs/eval/vit/vit_evaluation.json`. The single-video command uses the calibrated rule and writes a JSON verdict and annotated MP4 under `outputs/vit_demo/`. `peak_time_sec` marks the strongest sampled image; it is an approximate cue, not a frame-accurate collision timestamp or bounding box.

`run_vit_pipeline` also runs YOLO vehicle tracking and the motion rules, then uses the calibrated ViT to confirm each motion candidate. It writes `pipeline_summary.json`, tracked and confirmed preview videos, candidate clips, and incident logs under the selected output directory. The video-level ViT verdict and localized motion candidates are reported separately because the motion stage can miss an event even when the clip classifier sees one.

### 1. Train Baseline & CNN + LSTM on SCAD

Supports `{data_dir}/{train,val,test}/{accident,normal}/`, `{data_dir}/{accident,normal}/`, or a `--manifest` CSV (`path,label,split,video_group`):

Training requires a separate validation split. The test split is held out and is never used to select a checkpoint. Clips named like `v51_1.mp4` and `v51_2.mp4` are grouped as source video `v51`. For other naming schemes, assign the same `video_group` in the manifest to clips from one original recording so they cannot be split across train, validation, and test.

```bash
python -m src.train_classifier --data-dir data/scad --arch mean_pool --backbone resnet18 --epochs 15 --save-dir outputs/models/baseline
python -m src.train_classifier --data-dir data/scad --arch cnn_lstm --backbone resnet18 --epochs 15 --save-dir outputs/models/cnn_lstm
```

### 2. Evaluate (Experiment 1)

Compare `CNN + mean pooling` vs. `CNN + LSTM` on `accuracy`, `precision`, `recall`, `F1`, and `confusion matrix`:

```bash
python -m src.evaluate_classifier --data-dir data/scad --checkpoints outputs/models/baseline/best.pt outputs/models/cnn_lstm/best.pt --output-dir outputs/eval/experiment1
```

The evaluation also writes one predictions CSV per checkpoint with each clip's label, accident probability, prediction, and source-video group. Compare accuracy with `majority_class_baseline_accuracy` in the JSON report; a class-imbalanced dataset can make accuracy misleading.
Choose any probability threshold on validation data, then evaluate that fixed threshold on the held-out test set.

### 3. Confirm Candidates & Deduplicate Incidents

```bash
python -m src.confirm_accidents \
  --candidates outputs/candidates.json \
  --checkpoint outputs/models/cnn_lstm/best.pt \
  --video data/raw/test_cctv.mp4 \
  --render-video
```

Outputs:
- `outputs/confirmed_incidents.json` — deduplicated incidents (`incident_id`, `timestamp_hms`, `vehicles`, `accident_confidence`, `motion_score`)
- `outputs/confirmation_results.json` — per-candidate probabilities and confirm/reject decisions
- `outputs/events.log` & `outputs/events.jsonl` — human-readable and JSONL audit event logs (`INCIDENT_CREATED`, `INCIDENT_MERGED`, `CANDIDATE_REJECTED`)
- `outputs/incidents.csv` & `outputs/incidents/incident_XXX.{json,mp4,jpg}` — tabular summary and per-incident bundles
- `outputs/confirmed_preview.mp4` — annotated preview showing confirmed vs. rejected candidates (when `--render-video` is set)
- `outputs/experiment3_comparison.json` — motion-only vs. motion+classifier metrics (when `--ground-truth` is supplied)

### 4. Standalone Event Logger CLI

Replay `confirmation_results.json`, `confirmed_incidents.json`, or raw `candidates.json` through the cooldown event manager and logger:

```bash
python -m src.log_events --input outputs/confirmation_results.json --output-dir outputs --echo
```

## Tests

```bash
python -m unittest discover -s tests -v
```
