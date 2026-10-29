from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 22050
    hop_length: int = 512
    fmin: str | float = "C1"
    bins_per_octave: int = 36
    n_bins: int = 216
    log_epsilon: float = 1e-6
    pad_mode: str = "constant"
    time_stride: int = 1
    frame_origin_samples: int = 0
    block_samples: int | None = None
    frames_per_block: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "transformer"
    input_bins: int = 216
    d_model: int = 192
    nhead: int = 6
    num_layers: int = 6
    dim_feedforward: int = 768
    dropout: float = 0.15
    cnn_frontend: bool = False
    boundary_head: bool = False
    num_classes: int = 25
    max_positions: int = 8192


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 1337
    chunk_frames: int = 512
    chunk_hop_frames: int = 256
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    max_epochs: int = 100
    patience: int = 10
    num_workers: int = 0
    boundary_loss_weight: float = 0.2


@dataclass(frozen=True)
class ExperimentConfig:
    feature: FeatureConfig
    model: ModelConfig
    training: TrainingConfig


def _strict_dataclass(cls: type, values: Mapping[str, Any] | None):
    values = dict(values or {})
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {unknown}")
    return cls(**values)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    feature = _strict_dataclass(FeatureConfig, raw.get("feature"))
    model_values = dict(raw.get("model") or {})
    model_values.setdefault("input_bins", feature.n_bins)
    model = _strict_dataclass(ModelConfig, model_values)
    training = _strict_dataclass(TrainingConfig, raw.get("training"))
    if model.input_bins != feature.n_bins:
        raise ValueError("model.input_bins must equal feature.n_bins")
    if model.num_classes != 25:
        raise ValueError("This release implements exactly the 25-class vocabulary; define a versioned label set before expanding it")
    if training.chunk_hop_frames > training.chunk_frames:
        raise ValueError("chunk_hop_frames cannot exceed chunk_frames")
    return ExperimentConfig(feature, model, training)


def config_dict(config: Any) -> dict[str, Any]:
    return asdict(config)


def btc_ismir19_feature_config() -> FeatureConfig:
    return FeatureConfig(
        sample_rate=22050, hop_length=2048, fmin="C1", bins_per_octave=24,
        n_bins=144, block_samples=220500, frames_per_block=108,
    )
