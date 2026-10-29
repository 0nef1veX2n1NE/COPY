from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

ROOTS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
ROOT_TO_PC = {name: i for i, name in enumerate(ROOTS)}
ENHARMONIC = {"CB": "B", "DB": "C#", "EB": "D#", "FB": "E", "GB": "F#", "AB": "G#", "BB": "A#", "B#": "C", "E#": "F"}
LABELS_25 = tuple(label for root in ROOTS for label in (f"{root}:maj", f"{root}:min")) + ("N",)
LABEL_TO_INDEX_25 = {label: i for i, label in enumerate(LABELS_25)}
DISPLAY_QUALITY = {"maj": "", "min": "m", "7": "7", "maj7": "maj7", "min7": "m7", "sus2": "sus2", "sus4": "sus4"}


@dataclass(frozen=True)
class LabelMapping:
    source: str
    internal: str | None
    supported: bool
    reason: str | None = None


def _canonical_root(root: str) -> str | None:
    root = root.strip().upper().replace("♭", "B").replace("♯", "#")
    root = ENHARMONIC.get(root, root)
    return root if root in ROOT_TO_PC else None


def map_to_majmin(label: str) -> LabelMapping:
    """Map labels without converting unsupported harmony to N."""
    source = label.strip()
    if source.upper() in {"N", "NO_CHORD", "NO CHORD"}:
        return LabelMapping(source, "N", True)
    if source.upper() in {"X", "UNKNOWN", "UNK", "?"}:
        return LabelMapping(source, None, False, "unknown label")
    match = re.match(r"^([A-Ga-g](?:#|b|♯|♭)?)(?::([^/()]+))?(?:\([^)]*\))?(?:/.*)?$", source)
    if not match:
        return LabelMapping(source, None, False, "unparseable label")
    root = _canonical_root(match.group(1))
    if root is None:
        return LabelMapping(source, None, False, "unsupported root")
    quality = (match.group(2) or "maj").lower()
    quality = {"": "maj", "major": "maj", "maj": "maj", "min": "min", "minor": "min", "m": "min"}.get(quality, quality)
    if quality not in {"maj", "min"}:
        return LabelMapping(source, None, False, f"unsupported quality: {quality}")
    return LabelMapping(source, f"{root}:{quality}", True)


def internal_to_display(label: str) -> str:
    if label in {"N", "UNKNOWN"}:
        return label
    root, quality = label.split(":", 1)
    return root + DISPLAY_QUALITY.get(quality, f":{quality}")


def transpose_internal(label: str, semitones: int) -> str:
    if label in {"N", "UNKNOWN"}:
        return label
    root, quality = label.split(":", 1)
    return f"{ROOTS[(ROOT_TO_PC[root] + semitones) % 12]}:{quality}"


def labels_to_indices(labels: Iterable[str | None], ignore_index: int = -100) -> list[int]:
    return [LABEL_TO_INDEX_25.get(label, ignore_index) if label is not None else ignore_index for label in labels]

