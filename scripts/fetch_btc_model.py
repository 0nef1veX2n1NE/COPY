"""Fetch the one audited BTC checkpoint and fail closed on any byte change."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen

PINNED_URL = (
    "https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/"
    "2682317be668032e6e4b269ded36adaa2ad57df0/test/btc_model.pt"
)
PINNED_SHA256 = "71c2c5db17e8c43b8a9a9da5db36ef2d667158c07a214eba16344c154c00bf54"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--url", default=PINNED_URL)
    parser.add_argument("--sha256", default=PINNED_SHA256)
    parser.add_argument(
        "--accept-unverified-weight-terms",
        action="store_true",
        help="Acknowledge that upstream states no separate checkpoint license.",
    )
    args = parser.parse_args()
    if not args.accept_unverified_weight_terms:
        parser.error("review docs/PRETRAINED_MODELS.md, then pass --accept-unverified-weight-terms")
    expected = args.sha256.lower()
    if args.destination.exists() and file_sha256(args.destination) == expected:
        print(f"Verified existing checkpoint: {args.destination}")
        return

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        request = Request(args.url, headers={"User-Agent": "mir-chord-estimator/0.2"})
        with urlopen(request, timeout=120) as response:
            shutil.copyfileobj(response, temporary)
    try:
        actual = file_sha256(temporary_path)
        if actual != expected:
            raise RuntimeError(f"checkpoint SHA-256 mismatch: expected {expected}, got {actual}")
        temporary_path.replace(args.destination)
        print(f"Downloaded and verified {args.destination} ({actual})")
    finally:
        temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
