from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import FeatureConfig
from .labels import internal_to_display
from .timing import frame_edges


@dataclass(frozen=True)
class Candidate:
    label: str
    display: str
    score: float


@dataclass(frozen=True)
class Segment:
    start_sec: float
    end_sec: float
    chord: str
    display: str
    candidates: list[Candidate]
    score: float
    uncertainty: float
    needs_review: bool


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def viterbi_decode(logits: np.ndarray, stay_bias: float) -> np.ndarray:
    """Optional symmetric persistence prior; tune stay_bias on validation data."""
    frames, classes = logits.shape
    transition = np.zeros((classes, classes), np.float32)
    np.fill_diagonal(transition, float(stay_bias))
    score = logits[0].astype(np.float64)
    back = np.zeros((frames, classes), np.int32)
    for t in range(1, frames):
        candidates = score[:, None] + transition
        back[t] = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + logits[t]
    path = np.empty(frames, np.int64)
    path[-1] = score.argmax()
    for t in range(frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_segments(
    logits: np.ndarray, labels: Sequence[str], config: FeatureConfig,
    duration_sec: float, offset_sec: float = 0.0, top_k: int = 3,
    review_threshold: float = 0.45, viterbi_stay_bias: float | None = None,
) -> tuple[list[Segment], np.ndarray]:
    """Convert frame scores to segments. Scores are explicitly uncalibrated softmax."""
    if logits.ndim != 2 or logits.shape[1] != len(labels):
        raise ValueError("logits/labels shape mismatch")
    probabilities = softmax(logits.astype(np.float64)).astype(np.float32)
    path = viterbi_decode(np.log(probabilities + 1e-12), viterbi_stay_bias) if viterbi_stay_bias is not None else probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    effective = np.array(["UNKNOWN" if value < review_threshold else labels[index] for index, value in zip(path, confidence)], dtype=object)
    edges = frame_edges(len(logits), config, duration_sec, offset_sec)
    changes = np.flatnonzero(effective[1:] != effective[:-1]) + 1
    bounds = np.concatenate(([0], changes, [len(effective)]))
    segments: list[Segment] = []
    for left, right in zip(bounds[:-1], bounds[1:]):
        average = probabilities[left:right].mean(axis=0)
        order = np.argsort(average)[::-1][:max(1, top_k)]
        candidates = [Candidate(labels[i], internal_to_display(labels[i]), float(average[i])) for i in order]
        chord = str(effective[left])
        normalized_entropy = float(-(average * np.log(average + 1e-12)).sum() / np.log(len(labels)))
        segments.append(Segment(
            start_sec=float(edges[left]), end_sec=float(edges[right]), chord=chord,
            display=internal_to_display(chord), candidates=candidates,
            score=float(average[order[0]]), uncertainty=normalized_entropy,
            needs_review=chord == "UNKNOWN",
        ))
    return segments, probabilities


def write_outputs(
    output_prefix: str | Path, segments: list[Segment], logits: np.ndarray,
    probabilities: np.ndarray, metadata: dict[str, Any], labels: Sequence[str],
) -> dict[str, str]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": str(prefix.with_suffix(".json")), "csv": str(prefix.with_suffix(".csv")),
        "lab": str(prefix.with_suffix(".lab")), "frames": str(prefix.with_suffix(".frames.npz")),
    }
    document = {
        "schema_version": 1,
        "score_semantics": "uncalibrated_softmax_score; not a calibrated correctness probability",
        "task": "time-varying chord-symbol recognition; not guitar fret/voicing/capo recovery",
        "metadata": metadata,
        "segments": [asdict(segment) for segment in segments],
        "raw_frames": {"path": Path(paths["frames"]).name, "labels": list(labels)},
    }
    Path(paths["json"]).write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(paths["csv"]).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["start_sec", "end_sec", "chord", "display", "score", "uncertainty", "needs_review", "candidates_json"])
        writer.writeheader()
        for segment in segments:
            row = asdict(segment)
            row["candidates_json"] = json.dumps(row.pop("candidates"), ensure_ascii=False)
            writer.writerow(row)
    with Path(paths["lab"]).open("w", encoding="utf-8", newline="\n") as stream:
        for segment in segments:
            stream.write(f"{segment.start_sec:.6f}\t{segment.end_sec:.6f}\t{segment.chord}\n")
    np.savez_compressed(paths["frames"], logits=logits, softmax_scores=probabilities, labels=np.asarray(labels))
    return paths

