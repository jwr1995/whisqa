"""Command-line interface for WhiSQA."""

import argparse
import sys

import whisqa


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="whisqa",
        description="Predict speech quality scores for a WAV file.",
    )
    parser.add_argument("audio_file", help="Path to a mono WAV file.")
    parser.add_argument(
        "--model",
        choices=["single", "multi"],
        default="single",
        help=(
            "'single' predicts MOS only (default). "
            "'multi' predicts MOS + Noisiness, Coloration, Discontinuity, Loudness."
        ),
    )
    parser.add_argument(
        "--no-warn-resample",
        action="store_true",
        default=False,
        help="Suppress the warning when the input is resampled to 16 kHz.",
    )
    args = parser.parse_args()

    try:
        scores = whisqa.predict(
            args.audio_file,
            model_type=args.model,
            warn_resample=not args.no_warn_resample,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    for key, value in scores.items():
        print(f"{key.capitalize()}: {value:.2f}")


if __name__ == "__main__":
    main()
