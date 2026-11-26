# CCTV Accident Detection — Progress Summary

We audited the labeled videos in `data/raw/accident/` and `data/raw/normal/`. Of 1,128 files, 1,113 usable clips remain after excluding 9 Git LFS placeholders and 6 duplicates. We made train, validation, and test splits that keep clips from the same original recording together.

We tested several pretrained models and trained small classifiers on video features. The best workflow combines an accident-image ViT, an R3D-18 video model, and a YOLO11x accident detector. Its fusion rule was chosen on validation clips, then evaluated on 166 held-out test clips:

| Result | Value |
| --- | ---: |
| Validation accuracy | 77.8% |
| Test accuracy | **67.5% (112/166)** |
| Test accident precision / recall / F1 | 70.0% / 65.1% / 67.5% |

This improves on the roughly 54% result we were investigating, but it still makes mistakes: 24 false alarms and 30 missed accidents on the test set. The dataset is balanced; varied camera views and brief collision moments make classification difficult. Published model-page scores are from other datasets and should not be quoted as our accuracy.

To run the prepared demo from the repository root after installing `requirements.txt`:

```bash
python -m src.download_models
python -m src.fusion_workflow --video data/raw/accident/000831_3.mp4 --output-dir outputs/demo/accident
python -m src.fusion_workflow --video data/raw/normal/000831_1.mp4 --output-dir outputs/demo/normal
```

Each run saves a JSON verdict and a JPG of sampled frames with detector boxes. Full methods and per-model results are in `BENCHMARK.md`. All 30 automated tests pass.
