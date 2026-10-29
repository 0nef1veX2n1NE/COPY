from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import FeatureConfig, ModelConfig, btc_ismir19_feature_config
from .labels import LABELS_25
from .models import ChordTransformer, LegacyBTCModel

FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    pass


def save_checkpoint(
    path: str | Path, model: ChordTransformer, feature_config: FeatureConfig,
    normalization: dict[str, float], epoch: int, split_version: str,
    provenance: dict[str, Any], optimizer: torch.optim.Optimizer | None = None,
) -> None:
    payload = {
        "format": "mir-chord-estimator", "format_version": FORMAT_VERSION,
        "model_state": model.state_dict(), "model_config": asdict(model.config),
        "feature_config": asdict(feature_config), "labels": list(LABELS_25),
        "normalization": normalization, "epoch": int(epoch),
        "split_version": split_version, "provenance": provenance,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_custom_checkpoint(path: str | Path, device: torch.device) -> tuple[ChordTransformer, dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise CheckpointError(
            f"Checkpoint not found: {source}. Inference is disabled; train a model or supply a verified checkpoint."
        )
    payload = torch.load(source, map_location=device, weights_only=False)
    required = {"format", "format_version", "model_state", "model_config", "feature_config", "labels", "normalization"}
    missing = required - set(payload)
    if missing or payload.get("format") != "mir-chord-estimator":
        raise CheckpointError(f"Not a compatible custom checkpoint; missing={sorted(missing)}")
    if payload["format_version"] != FORMAT_VERSION:
        raise CheckpointError(f"Unsupported checkpoint format version: {payload['format_version']}")
    if tuple(payload["labels"]) != LABELS_25:
        raise CheckpointError("Checkpoint label ordering does not match the 25-class vocabulary")
    model = ChordTransformer(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    payload["feature_config_object"] = FeatureConfig(**payload["feature_config"])
    return model, payload


def load_btc_ismir19_checkpoint(path: str | Path, device: torch.device) -> tuple[LegacyBTCModel, dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise CheckpointError(f"BTC checkpoint not found: {source}. No fallback prediction will be produced.")
    payload = torch.load(source, map_location=device, weights_only=False)
    required = {"model", "mean", "std"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise CheckpointError(f"BTC checkpoint must contain keys {sorted(required)}")
    model = LegacyBTCModel()
    if "labels" in payload and tuple(payload["labels"]) != LABELS_25:
        raise CheckpointError("BTC-derived checkpoint label ordering mismatch")
    try:
        model.load_state_dict(payload["model"], strict=True)
    except RuntimeError as error:
        raise CheckpointError(f"BTC checkpoint architecture/state ordering mismatch: {error}") from error
    model.to(device).eval()
    return model, {
        "format": "btc-ismir19-upstream", "format_version": 1,
        "feature_config_object": btc_ismir19_feature_config(),
        "normalization": {"mean": float(payload["mean"]), "std": float(payload["std"])},
        "epoch": payload.get("epoch"), "split_version": payload.get("split_version", "upstream-5-fold-unspecified-checkpoint-fold"),
        "provenance": payload.get("provenance", {"repository": "https://github.com/jayg996/BTC-ISMIR19", "commit": "2682317be668032e6e4b269ded36adaa2ad57df0", "checkpoint_license": "not separately stated upstream"}),
    }


def load_inference_checkpoint(path: str | Path, device: torch.device, profile: str):
    if profile == "custom":
        return load_custom_checkpoint(path, device)
    if profile == "btc-ismir19-25":
        return load_btc_ismir19_checkpoint(path, device)
    raise CheckpointError(f"Unknown checkpoint profile: {profile}")


def save_btc_finetuned_checkpoint(
    path: str | Path, model: LegacyBTCModel, normalization: dict[str, float],
    epoch: int, split_version: str, optimizer: torch.optim.Optimizer,
    provenance: dict[str, Any],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "mean": float(normalization["mean"]),
        "std": float(normalization["std"]), "optimizer": optimizer.state_dict(),
        "epoch": int(epoch), "labels": list(LABELS_25),
        "feature_config": asdict(btc_ismir19_feature_config()),
        "split_version": split_version, "provenance": provenance,
    }, path)
