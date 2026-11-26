# Accident detection: measured results and demo

## What to show

The current clip classifier combines three signals: an accident-image ViT adapted to the labeled videos, a pretrained R3D-18 video encoder adapted to the videos, and an accident-specific YOLO11x detector. The fusion rule and threshold were selected on the validation split. It reached **67.5% accuracy (112/166)** on a held-out, source-disjoint test split, with **70.0% precision**, **65.1% recall**, and **67.5% F1** for the accident class. The confusion matrix is below; rows are actual labels and columns are predicted labels.

| | Predicted normal | Predicted accident |
| --- | ---: | ---: |
| Actual normal | 56 | 24 |
| Actual accident | 30 | 56 |

The test split's majority-class baseline is 51.8% (86/166). A 95% Wilson interval for the measured fusion accuracy is approximately 60.0%–74.1%; new cameras and longer videos may behave differently.

## Data and evaluation

The `data/raw/accident/` and `data/raw/normal/` folders contained 1,128 `.mp4` files. The audit excluded 9 Git LFS pointer files and 6 exact duplicate videos, leaving **1,113 decodable, uniquely counted clips** from **511 original recording groups**. No identical file had conflicting labels. The classes are balanced overall: 557 accident and 556 normal clips. `data/benchmark_manifest.csv` records the fixed split:

| Split | Accident | Normal | Total |
| --- | ---: | ---: | ---: |
| Train | 385 | 386 | 771 |
| Validation | 86 | 90 | 176 |
| Test | 86 | 80 | 166 |

Clips from the same original recording remain in one split. This avoids scoring a model on an adjacent segment of a recording it saw during training. Model parameters were fitted on train; fusion weights and decision threshold were chosen on validation; the test split was used to report performance.

| Model | Validation accuracy | Test accuracy | Test accident recall |
| --- | ---: | ---: | ---: |
| Pretrained image ViT, frame aggregation | 68.8% | 56.0% | 54.7% |
| Pretrained temporal crash model | 54.6% | 53.6% | 64.0% |
| YOLO11x accident boxes, clip rule | 65.9% | 57.2% | 44.2% |
| R3D-18 video features + trained classifier | 69.9% | 60.2% | 57.0% |
| ViT image features + trained classifier | 71.6% | 63.9% | 60.5% |
| **ViT + R3D-18 + YOLO11x fusion** | **77.8%** | **67.5%** | **65.1%** |

The [image ViT model card](https://huggingface.co/tiya1012/vit-accident-image) reports performance on its own image dataset. The [YOLO11x model card](https://huggingface.co/Enos-123/traffic-accident-detection-yolo11x) reports object-detection mAP on a different dataset. Neither number is the clip-level accuracy on these CCTV videos. The pretrained [temporal crash model](https://huggingface.co/beaunix/car-crash-detector) was evaluated as a comparison and is not part of the final workflow.

## Run the demo

From the repository root, install `requirements.txt` as described in `README.md`, then run:

```bash
python -m src.download_models
python -m src.fusion_workflow --video data/raw/accident/000831_3.mp4 --output-dir outputs/demo/accident
python -m src.fusion_workflow --video data/raw/normal/000831_1.mp4 --output-dir outputs/demo/normal
```

The two test-set examples are from the same camera. The accident clip produced fusion score 0.631 and `ACCIDENT`; the normal clip produced 0.181 and `NORMAL`, against the validation-selected threshold 0.425. Each run writes `<video>_result.json` and `<video>_evidence.jpg`. The image shows the four frames sampled by YOLO, with its accident boxes. The fusion score is a decision score, **not** a calibrated probability that an accident occurred.

To reproduce the benchmark from scratch after the model weights are available:

```bash
python -m src.prepare_benchmark --data-dir data/raw
python -m src.vit_probe
python -m src.r3d_probe
python -m src.yolo_accident_workflow
python -m src.fusion_workflow
```

Results and per-clip predictions are written under `outputs/eval/`. The final report is `outputs/eval/fusion/fusion_evaluation.json`; the per-clip results are `val_predictions.csv` and `test_predictions.csv` beside it.

## What the result means

The dataset is usable and balanced, but it is difficult: cameras, illumination, weather, and viewpoints vary; in some accident-labeled clips the impact is visible only briefly. A still-image classifier can therefore miss the relevant moment. YOLO11x sometimes places accident boxes around vehicles merely crossing an intersection, causing false alarms. The three signals reduce these errors, but the test set still contains 24 false alarms and 30 missed accidents.

Only clip labels are available here. The YOLO boxes and peak detector time are visual evidence for a decision, not independently validated collision localization. This system is a research demo and should not be presented as a safety-critical alerting system without a larger, camera-specific evaluation.
