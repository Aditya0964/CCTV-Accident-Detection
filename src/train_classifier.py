"""Train Model 1 (CNN + mean pooling) or Model 2 (CNN + LSTM) on SCAD clips."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluate_classifier import evaluate_model_on_loader
from src.scad_dataset import ClipSample, SCADClipDataset, discover_scad_samples
from src.video_classifier import (
    ModelConfig,
    build_video_classifier,
    resolve_device,
    save_checkpoint,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_pos_weight(samples: list[ClipSample], mode: str) -> float:
    if mode != "auto":
        return float(mode)
    pos = sum(1 for s in samples if s.label == 1)
    neg = sum(1 for s in samples if s.label == 0)
    if pos == 0 or neg == 0:
        return 1.0
    return neg / pos


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None, help="SCAD dataset root directory")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional CSV manifest")
    parser.add_argument("--arch", choices=["mean_pool", "cnn_lstm"], default="mean_pool")
    parser.add_argument(
        "--backbone",
        choices=["resnet18", "mobilenet_v3_small", "mobilenet_v3_large"],
        default="resnet18",
    )
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3, help="Classification head learning rate")
    parser.add_argument("--backbone-lr", type=float, default=1e-4, help="CNN backbone learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--pos-weight", default="auto", help="'auto' or positive float")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/models/baseline"))
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.data_dir is None and args.manifest is None:
        raise ValueError("Specify --data-dir and/or --manifest")

    _set_seed(args.seed)
    device = resolve_device(args.device)
    splits = discover_scad_samples(data_root=args.data_dir, manifest_csv=args.manifest, seed=args.seed)

    train_samples = splits["train"]
    val_samples = splits["val"]
    if not train_samples:
        raise RuntimeError("No training samples discovered")
    if not val_samples:
        raise RuntimeError(
            "No validation samples discovered. Provide a separate val split or more "
            "source-video groups for the automatic split; test data must remain held out."
        )

    config = ModelConfig(
        arch=args.arch,
        backbone=args.backbone,
        num_frames=args.num_frames,
        input_size=args.input_size,
        hidden_dim=args.hidden_dim,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    )
    model = build_video_classifier(config, pretrained=not args.no_pretrained)
    if args.freeze_backbone:
        model.set_backbone_trainable(False)
    model.to(device)

    train_ds = SCADClipDataset(
        train_samples,
        num_frames=args.num_frames,
        input_size=args.input_size,
        augment=True,
        seed=args.seed,
    )
    val_ds = SCADClipDataset(
        val_samples,
        num_frames=args.num_frames,
        input_size=args.input_size,
        augment=False,
        seed=args.seed,
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    pos_w = _compute_pos_weight(train_samples, args.pos_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))

    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    param_groups = [{"params": head_params, "lr": args.lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.save_dir / "best.pt"
    last_path = args.save_dir / "last.pt"
    history: list[dict[str, Any]] = []
    best_key = (-1.0, -float("inf"))  # (val_f1, -val_loss)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_backbone:
            # Frozen batch-normalization statistics must also remain frozen.
            model.backbone.eval()
        running_loss = 0.0
        seen = 0
        for clips, labels in train_loader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(clips)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * labels.size(0)
            seen += labels.size(0)

        train_loss = running_loss / max(1, seen)
        val_metrics, val_loss, _, _ = evaluate_model_on_loader(
            model, val_loader, device=device, criterion=criterion, threshold=config.threshold
        )

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        history.append(record)
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        rank_key = (val_metrics["f1"], -val_loss)
        if rank_key > best_key:
            best_key = rank_key
            save_checkpoint(best_path, model, config, extra={"best_epoch": epoch, "val_metrics": val_metrics})

    save_checkpoint(last_path, model, config, extra={"epochs": args.epochs})
    history_path = args.save_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    summary = {
        "save_dir": str(args.save_dir),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "device": str(device),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "history": history,
    }
    print(f"Training complete. Best checkpoint: {best_path}")
    return summary


if __name__ == "__main__":
    run(parse_args())
