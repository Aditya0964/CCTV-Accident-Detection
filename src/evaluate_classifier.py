"""Experiment 1: Binary classification metrics, threshold sweep, and model comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.scad_dataset import SCADClipDataset, discover_scad_samples
from src.video_classifier import load_checkpoint, resolve_device


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute accuracy, precision, recall, F1, and confusion matrix for binary predictions."""
    y_true_int = np.asarray(y_true, dtype=np.int64).ravel()
    y_prob_arr = np.asarray(y_prob, dtype=np.float64).ravel()
    if y_true_int.shape != y_prob_arr.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true_int.shape}, y_prob={y_prob_arr.shape}")

    y_pred = (y_prob_arr > threshold).astype(np.int64)
    tp = int(np.sum((y_true_int == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true_int == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true_int == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true_int == 1) & (y_pred == 0)))

    total = len(y_true_int)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "support_pos": int(np.sum(y_true_int == 1)),
        "support_neg": int(np.sum(y_true_int == 0)),
    }


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[list[float]] = None,
) -> dict[str, Any]:
    """Find threshold maximizing F1 score across candidate cutoffs."""
    grid = thresholds or [round(0.2 + 0.05 * i, 2) for i in range(13)]
    results = [compute_binary_metrics(y_true, y_prob, threshold=t) for t in grid]
    best = max(results, key=lambda r: (r["f1"], r["recall"], r["accuracy"]))
    return {"best": best, "sweep": results}


@torch.no_grad()
def evaluate_model_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    threshold: float = 0.5,
) -> tuple[dict[str, Any], float, np.ndarray, np.ndarray]:
    """Run evaluation pass returning (metrics_dict, mean_loss, y_true, y_prob)."""
    model.eval()
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    total_loss = 0.0
    total_items = 0

    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(clips)
        if criterion is not None:
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
        probs = torch.sigmoid(logits)
        all_targets.append(labels.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())
        total_items += labels.size(0)

    y_true = np.concatenate(all_targets) if all_targets else np.empty(0, dtype=np.float32)
    y_prob = np.concatenate(all_probs) if all_probs else np.empty(0, dtype=np.float32)
    mean_loss = total_loss / total_items if (criterion is not None and total_items > 0) else 0.0
    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    return metrics, mean_loss, y_true, y_prob


def save_confusion_matrix_plot(
    cm: list[list[int]],
    title: str,
    output_path: Path,
) -> None:
    """Save 2x2 confusion matrix heatmap PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(cm, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(arr, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels = ["Normal (0)", "Accident (1)"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    thresh = arr.max() / 2.0 if arr.size > 0 else 0.0
    for i in range(2):
        for j in range(2):
            color = "white" if arr[i, j] > thresh else "black"
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", color=color, fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None, help="SCAD dataset root directory")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional CSV manifest")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="One or more model checkpoints (.pt) to evaluate/compare",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sweep-thresholds", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval/experiment1"))
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.data_dir is None and args.manifest is None:
        raise ValueError("Specify --data-dir and/or --manifest")

    splits = discover_scad_samples(data_root=args.data_dir, manifest_csv=args.manifest)
    samples = splits[args.split]
    if not samples:
        raise RuntimeError(f"No samples found in split '{args.split}'")

    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"split": args.split, "num_samples": len(samples), "models": []}

    for idx, ckpt_path in enumerate(args.checkpoints):
        model, cfg, _ = load_checkpoint(ckpt_path, device=device)
        dataset = SCADClipDataset(
            samples,
            num_frames=cfg.num_frames,
            input_size=cfg.input_size,
            augment=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        metrics, _, y_true, y_prob = evaluate_model_on_loader(
            model, loader, device=device, threshold=args.threshold
        )

        run_label = f"{ckpt_path.parent.name}_{cfg.arch}" if ckpt_path.parent.name else f"model_{idx}"
        cm_path = args.output_dir / f"cm_{run_label}.png"
        save_confusion_matrix_plot(
            metrics["confusion_matrix"],
            title=f"{cfg.arch} ({cfg.backbone}) - {args.split}",
            output_path=cm_path,
        )

        entry: dict[str, Any] = {
            "checkpoint": str(ckpt_path),
            "arch": cfg.arch,
            "backbone": cfg.backbone,
            "metrics": metrics,
            "confusion_matrix_plot": str(cm_path),
        }
        if args.sweep_thresholds:
            entry["threshold_sweep"] = sweep_thresholds(y_true, y_prob)
        report["models"].append(entry)

        print(
            f"[{cfg.arch} | {cfg.backbone}] "
            f"acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
            f"rec={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
            f"cm={metrics['confusion_matrix']}"
        )

    out_json = args.output_dir / "experiment1_metrics.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Evaluation report saved: {out_json}")
    return report


if __name__ == "__main__":
    run(parse_args())
