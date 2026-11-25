"""Stage 9 & 10: Video-based accident classifiers (CNN + mean pooling, CNN + LSTM)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torchvision import models as tv_models


@dataclass
class ModelConfig:
    """Serializable configuration for video accident classifiers."""

    arch: str = "mean_pool"  # "mean_pool" (Stage 9) or "cnn_lstm" (Stage 10)
    backbone: str = "resnet18"  # "resnet18", "mobilenet_v3_small", "mobilenet_v3_large"
    num_frames: int = 16
    input_size: int = 224
    hidden_dim: int = 256
    lstm_layers: int = 1
    dropout: float = 0.3
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.arch not in ("mean_pool", "cnn_lstm"):
            raise ValueError(f"Unsupported arch '{self.arch}'; expected 'mean_pool' or 'cnn_lstm'")
        if self.backbone not in ("resnet18", "mobilenet_v3_small", "mobilenet_v3_large"):
            raise ValueError(f"Unsupported backbone '{self.backbone}'")
        if self.num_frames < 1 or self.input_size < 32:
            raise ValueError("num_frames must be >= 1 and input_size >= 32")


def resolve_device(device_arg: str = "auto") -> torch.device:
    """Resolve PyTorch compute device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Instantiate a CNN spatial feature extractor returning a 1D embedding per frame."""
    if name == "resnet18":
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        net = tv_models.resnet18(weights=weights)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, feat_dim

    if name == "mobilenet_v3_small":
        weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        net = tv_models.mobilenet_v3_small(weights=weights)
        feat_dim = net.classifier[0].in_features
        backbone = nn.Sequential(net.features, net.avgpool, nn.Flatten(start_dim=1))
        return backbone, feat_dim

    if name == "mobilenet_v3_large":
        weights = tv_models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        net = tv_models.mobilenet_v3_large(weights=weights)
        feat_dim = net.classifier[0].in_features
        backbone = nn.Sequential(net.features, net.avgpool, nn.Flatten(start_dim=1))
        return backbone, feat_dim

    raise ValueError(f"Unknown backbone: {name}")


class _BaseVideoClassifier(nn.Module):
    """Shared frame-feature extraction and probability inference interface."""

    def __init__(self, config: ModelConfig, pretrained: bool = True) -> None:
        super().__init__()
        self.config = config
        self.backbone, self.feature_dim = build_backbone(config.backbone, pretrained=pretrained)

    def extract_frame_features(self, clips: torch.Tensor) -> torch.Tensor:
        """Encode (B, T, C, H, W) clips into per-frame feature sequences (B, T, D)."""
        if clips.ndim != 5:
            raise ValueError(f"Expected 5D clip tensor (B, T, C, H, W), got {tuple(clips.shape)}")
        bsz, steps, channels, height, width = clips.shape
        flat = clips.reshape(bsz * steps, channels, height, width)
        feats = self.backbone(flat)
        return feats.reshape(bsz, steps, self.feature_dim)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = trainable

    @torch.no_grad()
    def predict_proba(self, clips: torch.Tensor) -> torch.Tensor:
        """Return P(accident) in [0, 1] for each clip in batch."""
        logits = self.forward(clips)
        return torch.sigmoid(logits)


class TemporalMeanPoolClassifier(_BaseVideoClassifier):
    """Stage 9 baseline: CNN frame features -> temporal mean pooling -> linear classifier."""

    def __init__(self, config: ModelConfig, pretrained: bool = True) -> None:
        super().__init__(config, pretrained=pretrained)
        self.dropout = nn.Dropout(p=config.dropout)
        self.head = nn.Linear(self.feature_dim, 1)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        seq_feats = self.extract_frame_features(clips)  # (B, T, D)
        pooled = seq_feats.mean(dim=1)  # (B, D)
        return self.head(self.dropout(pooled)).squeeze(-1)


class CNNLSTMClassifier(_BaseVideoClassifier):
    """Stage 10 temporal model: CNN frame features -> LSTM -> binary classifier."""

    def __init__(self, config: ModelConfig, pretrained: bool = True) -> None:
        super().__init__(config, pretrained=pretrained)
        self.norm = nn.LayerNorm(self.feature_dim)
        lstm_dropout = config.dropout if config.lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(p=config.dropout)
        self.head = nn.Linear(config.hidden_dim, 1)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        seq_feats = self.norm(self.extract_frame_features(clips))  # (B, T, D)
        lstm_out, _ = self.lstm(seq_feats)  # (B, T, H)
        last_hidden = lstm_out[:, -1, :]  # h_T: (B, H)
        return self.head(self.dropout(last_hidden)).squeeze(-1)


def build_video_classifier(
    config: ModelConfig,
    pretrained: bool = True,
) -> _BaseVideoClassifier:
    """Factory returning the configured baseline or CNN+LSTM classifier."""
    if config.arch == "mean_pool":
        return TemporalMeanPoolClassifier(config, pretrained=pretrained)
    if config.arch == "cnn_lstm":
        return CNNLSTMClassifier(config, pretrained=pretrained)
    raise ValueError(f"Unsupported arch: {config.arch}")


def save_checkpoint(
    path: Path,
    model: _BaseVideoClassifier,
    config: ModelConfig,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Persist model weights and config for reproducible evaluation and inference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_config": asdict(config),
        "model_state_dict": model.state_dict(),
    }
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    device: torch.device | str = "cpu",
) -> tuple[_BaseVideoClassifier, ModelConfig, dict[str, Any]]:
    """Restore classifier weights and ModelConfig without requiring network access."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=True)
    config = ModelConfig(**payload["model_config"])
    model = build_video_classifier(config, pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config, payload.get("extra", {})
