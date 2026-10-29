from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset import IGNORE_INDEX, align_labels, load_manifest, read_annotations
from .labels import LABELS_25
from .pipeline import analyze_audio
from .timing import frame_edges


def boundary_f1(reference: np.ndarray, estimate: np.ndarray, tolerance_frames: int) -> dict[str, float]:
    ref = np.flatnonzero((reference[1:] != reference[:-1]) & (reference[1:] != IGNORE_INDEX) & (reference[:-1] != IGNORE_INDEX)) + 1
    est = np.flatnonzero(estimate[1:] != estimate[:-1]) + 1
    used: set[int] = set()
    true_positive = 0
    for boundary in est:
        candidates = [(abs(int(boundary) - int(value)), i) for i, value in enumerate(ref) if i not in used and abs(int(boundary) - int(value)) <= tolerance_frames]
        if candidates:
            _, index = min(candidates)
            used.add(index)
            true_positive += 1
    precision = true_positive / len(est) if len(est) else float(len(ref) == 0)
    recall = true_positive / len(ref) if len(ref) else float(len(est) == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "reference_count": int(len(ref)), "estimate_count": int(len(est))}


def evaluate_manifest(
    manifest_path: str, checkpoint_path: str, device: torch.device,
    boundary_tolerance_sec: float = 0.5, include_splits: tuple[str, ...] = ("test",),
    ffmpeg: str | None = None, cache_dir: str | None = None, profile: str = "custom",
) -> dict[str, Any]:
    tracks, split_version = load_manifest(manifest_path)
    chosen = [track for track in tracks if track.split in include_splits]
    if not chosen:
        raise ValueError(f"No tracks in requested splits: {include_splits}")
    totals = {"correct_all": 0.0, "duration_all": 0.0, "correct_no_n": 0.0, "duration_no_n": 0.0, "unsupported_duration": 0.0, "annotated_duration": 0.0}
    boundary_counts = {"tp_weighted": 0.0, "estimate_count": 0, "reference_count": 0}
    per_track = []
    strata: dict[str, list[float]] = {}
    for track in chosen:
        result = analyze_audio(track.audio, checkpoint_path, device, cache_dir=cache_dir, ffmpeg=ffmpeg, profile=profile)
        config = result["feature_config"]
        reference, counts = align_labels(read_annotations(track.annotation), len(result["logits"]), config)
        estimate = result["probabilities"].argmax(axis=1)
        edges = frame_edges(len(reference), config, result["metadata"]["audio"]["duration_sec"])
        weights = np.diff(edges)
        supported = reference != IGNORE_INDEX
        no_n = supported & (reference != len(LABELS_25) - 1)
        correct = reference == estimate
        totals["correct_all"] += float(weights[supported & correct].sum())
        totals["duration_all"] += float(weights[supported].sum())
        totals["correct_no_n"] += float(weights[no_n & correct].sum())
        totals["duration_no_n"] += float(weights[no_n].sum())
        totals["unsupported_duration"] += counts["unsupported_frames"] * config.hop_length / config.sample_rate
        totals["annotated_duration"] += (counts["unsupported_frames"] + counts["supported_frames"]) * config.hop_length / config.sample_rate
        tolerance_frames = max(0, round(boundary_tolerance_sec * config.sample_rate / config.hop_length / config.time_stride))
        boundary = boundary_f1(reference, estimate, tolerance_frames)
        tp = boundary["recall"] * boundary["reference_count"]
        boundary_counts["tp_weighted"] += tp
        boundary_counts["estimate_count"] += boundary["estimate_count"]
        boundary_counts["reference_count"] += boundary["reference_count"]
        accuracy = float(weights[supported & correct].sum() / weights[supported].sum()) if weights[supported].sum() else float("nan")
        strata.setdefault(track.source_type, []).append(accuracy)
        per_track.append({"recording_id": track.recording_id, "source_type": track.source_type, "weighted_accuracy": accuracy, "boundary": boundary, "mapping_counts": counts, "runtime": result["metadata"]["timing"]})
    precision = boundary_counts["tp_weighted"] / boundary_counts["estimate_count"] if boundary_counts["estimate_count"] else 0.0
    recall = boundary_counts["tp_weighted"] / boundary_counts["reference_count"] if boundary_counts["reference_count"] else 0.0
    report = {
        "evaluation_vocabulary": list(LABELS_25), "splits": list(include_splits), "split_version": split_version,
        "boundary_tolerance_sec": boundary_tolerance_sec,
        "weighted_chord_accuracy_including_n": totals["correct_all"] / totals["duration_all"] if totals["duration_all"] else None,
        "weighted_chord_accuracy_excluding_n": totals["correct_no_n"] / totals["duration_no_n"] if totals["duration_no_n"] else None,
        "unsupported_annotation_ratio": totals["unsupported_duration"] / totals["annotated_duration"] if totals["annotated_duration"] else None,
        "boundary": {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, **boundary_counts},
        "strata_mean_track_accuracy": {key: float(np.nanmean(values)) for key, values in strata.items()},
        "per_track": per_track,
        "notes": ["Scores cover only supported 25-class reference frames.", "Frame spacing is not claimed as chord-boundary accuracy.", "No accuracy or runtime is reported for unexecuted data."],
    }
    return report


def write_evaluation(report: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
