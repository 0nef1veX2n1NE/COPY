from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .audio_io import decode_audio
from .checkpoint import load_btc_ismir19_checkpoint, save_btc_finetuned_checkpoint, save_checkpoint
from .config import ExperimentConfig, btc_ismir19_feature_config
from .dataset import ChunkDataset, PreparedTrack, align_labels, load_manifest, read_annotations, IGNORE_INDEX
from .features import compute_btc_legacy_log_cqt, compute_log_cqt, normalize_features, training_statistics
from .models import ChordTransformer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _prepare_tracks(tracks, feature_config, ffmpeg, feature_function=compute_log_cqt):
    raw = []
    for track in tracks:
        audio = decode_audio(track.audio, feature_config.sample_rate, ffmpeg)
        features = feature_function(audio.samples, feature_config)
        labels, counts = align_labels(read_annotations(track.annotation), len(features), feature_config, audio.offset_sec)
        raw.append((track, features, labels, counts))
    return raw


def train_model(
    config: ExperimentConfig, manifest_path: str, output_dir: str,
    device: torch.device, ffmpeg: str | None = None,
    init_btc_checkpoint: str | None = None,
) -> dict:
    seed_everything(config.training.seed)
    tracks, split_version = load_manifest(manifest_path)
    train_rows = [track for track in tracks if track.split == "train"]
    validation_rows = [track for track in tracks if track.split == "validation"]
    if not train_rows or not validation_rows:
        raise ValueError("Manifest must contain non-empty train and validation splits")
    if init_btc_checkpoint:
        feature_config = btc_ismir19_feature_config()
        model, upstream = load_btc_ismir19_checkpoint(init_btc_checkpoint, device)
        mean, std = upstream["normalization"]["mean"], upstream["normalization"]["std"]
        feature_function = compute_btc_legacy_log_cqt
        chunk_frames, chunk_hop_frames = 108, 54
        model_kind = "btc-ismir19-25-finetune"
    else:
        feature_config = config.feature
        model = ChordTransformer(config.model).to(device)
        feature_function = compute_log_cqt
        chunk_frames, chunk_hop_frames = config.training.chunk_frames, config.training.chunk_hop_frames
        model_kind = config.model.architecture
    raw_train = _prepare_tracks(train_rows, feature_config, ffmpeg, feature_function)
    if not init_btc_checkpoint:
        mean, std = training_statistics([row[1] for row in raw_train])
    raw_validation = _prepare_tracks(validation_rows, feature_config, ffmpeg, feature_function)

    def normalize(rows):
        return [PreparedTrack(normalize_features(features, mean, std), labels, track.recording_id) for track, features, labels, _ in rows]

    train_dataset = ChunkDataset(normalize(raw_train), chunk_frames, chunk_hop_frames)
    validation_dataset = ChunkDataset(normalize(raw_validation), chunk_frames, chunk_frames)
    generator = torch.Generator().manual_seed(config.training.seed)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=config.training.num_workers, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    chord_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    boundary_loss = nn.BCEWithLogitsLoss(reduction="none")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment_config.json").write_text(json.dumps({"feature": asdict(feature_config), "model": model_kind, "training": asdict(config.training), "init_btc_checkpoint": init_btc_checkpoint}, indent=2), encoding="utf-8")
    best_loss = float("inf")
    stale = 0
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, config.training.max_epochs + 1):
        model.train()
        train_total = 0.0
        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            valid = batch["valid"].to(device)
            result = model(features, valid)
            loss = chord_loss(result["chord_logits"].reshape(-1, 25), labels.reshape(-1))
            if "boundary_logits" in result:
                mask = (labels != IGNORE_INDEX).float()
                bce = boundary_loss(result["boundary_logits"], batch["boundary"].to(device))
                loss = loss + config.training.boundary_loss_weight * (bce * mask).sum() / mask.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_total += float(loss.detach())
        model.eval()
        validation_total = 0.0
        with torch.inference_mode():
            for batch in validation_loader:
                labels = batch["labels"].to(device)
                result = model(batch["features"].to(device), batch["valid"].to(device))
                validation_total += float(chord_loss(result["chord_logits"].reshape(-1, 25), labels.reshape(-1)))
        row = {"epoch": epoch, "train_loss": train_total / max(len(train_loader), 1), "validation_loss": validation_total / max(len(validation_loader), 1)}
        history.append(row)
        if row["validation_loss"] < best_loss:
            best_loss = row["validation_loss"]
            stale = 0
            provenance = {"kind": model_kind, "manifest": str(Path(manifest_path).resolve())}
            if init_btc_checkpoint:
                provenance["initialized_from"] = str(Path(init_btc_checkpoint).resolve())
                save_btc_finetuned_checkpoint(output / "best.pt", model, {"mean": mean, "std": std}, epoch, split_version, optimizer, provenance)
            else:
                save_checkpoint(output / "best.pt", model, feature_config, {"mean": mean, "std": std}, epoch, split_version, provenance, optimizer)
        else:
            stale += 1
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stale >= config.training.patience:
            break
    return {"best_checkpoint": str(output / "best.pt"), "best_validation_loss": best_loss, "epochs": len(history), "elapsed_sec": time.perf_counter() - start_time}
