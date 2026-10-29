from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_experiment_config
from .evaluation import evaluate_manifest, write_evaluation
from .pipeline import analyze_to_files, choose_device
from .training import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chord-estimator", description="Checkpoint-gated time-varying chord estimation")
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="Analyze one MP3/WAV and write JSON, CSV, LAB and raw frame scores")
    analyze.add_argument("input")
    analyze.add_argument("--checkpoint", required=True, help="A trained compatible checkpoint; random inference is prohibited")
    analyze.add_argument("--profile", default="custom", choices=["custom", "btc-ismir19-25"])
    analyze.add_argument("--output", required=True, help="Output prefix, without extension")
    analyze.add_argument("--device", default="auto")
    analyze.add_argument("--ffmpeg")
    analyze.add_argument("--cache-dir")
    analyze.add_argument("--chunk-frames", type=int)
    analyze.add_argument("--chunk-hop-frames", type=int)
    analyze.add_argument("--batch-size", type=int, default=8)
    analyze.add_argument("--top-k", type=int, default=3)
    analyze.add_argument("--review-threshold", type=float, default=0.45)
    analyze.add_argument("--viterbi-stay-bias", type=float, help="Optional; tune on validation data before use")

    train = commands.add_parser("train", help="Train from a recording-level split manifest")
    train.add_argument("--config", required=True)
    train.add_argument("--manifest", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--device", default="auto")
    train.add_argument("--ffmpeg")
    train.add_argument("--init-btc-checkpoint", help="Fine-tune the exact BTC-ISMIR19 25-class architecture and preprocessing")

    evaluate = commands.add_parser("evaluate", help="Evaluate only a held-out manifest split")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--profile", default="custom", choices=["custom", "btc-ismir19-25"])
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--ffmpeg")
    evaluate.add_argument("--cache-dir")
    evaluate.add_argument("--split", action="append", default=None, choices=["validation", "test"])
    evaluate.add_argument("--boundary-tolerance-sec", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = choose_device(args.device)
    if args.command == "analyze":
        result = analyze_to_files(
            audio_path=args.input, checkpoint_path=args.checkpoint, output_prefix=args.output,
            device=device, ffmpeg=args.ffmpeg, cache_dir=args.cache_dir,
            chunk_frames=args.chunk_frames, chunk_hop_frames=args.chunk_hop_frames,
            batch_size=args.batch_size, top_k=args.top_k, profile=args.profile,
            review_threshold=args.review_threshold, viterbi_stay_bias=args.viterbi_stay_bias,
        )
    elif args.command == "train":
        result = train_model(load_experiment_config(args.config), args.manifest, args.output_dir, device, args.ffmpeg, args.init_btc_checkpoint)
    else:
        result = evaluate_manifest(
            args.manifest, args.checkpoint, device,
            boundary_tolerance_sec=args.boundary_tolerance_sec,
            include_splits=tuple(args.split or ["test"]), ffmpeg=args.ffmpeg, cache_dir=args.cache_dir, profile=args.profile,
        )
        write_evaluation(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
