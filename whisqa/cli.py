"""Command-line interface for WhiSQA."""

import argparse
import csv
import sys
from pathlib import Path

import whisqa


def _print_scores(scores: dict, filename: str = None) -> None:
    if filename:
        print(filename)
    for key, value in scores.items():
        print(f"  {key.capitalize()}: {value:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="whisqa",
        description="Predict speech quality scores for a WAV file or directory.",
    )
    parser.add_argument(
        "audio_path",
        help="Path to a mono audio file or a directory of audio files.",
    )
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
        help="Suppress the warning when input is resampled to 16 kHz.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        default=False,
        help="When scoring a directory, do not search sub-directories.",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Write results to a CSV file (directory mode only).",
    )
    args = parser.parse_args()

    path = Path(args.audio_path)
    warn_resample = not args.no_warn_resample

    # ------------------------------------------------------------------ #
    # Directory mode                                                      #
    # ------------------------------------------------------------------ #
    if path.is_dir():
        scorer = whisqa.WhiSQA(args.model)
        try:
            results = scorer.predict_dir(
                path,
                recursive=not args.no_recursive,
                warn_resample=warn_resample,
            )
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        if not results:
            print("No audio files found.", file=sys.stderr)
            sys.exit(1)

        if args.output:
            out_path = Path(args.output)
            keys = list(results[0].keys())
            with out_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
            print(f"Results written to {out_path} ({len(results)} files).")
        else:
            for row in results:
                fname = Path(row["file"]).name
                scores = {k: v for k, v in row.items() if k != "file"}
                _print_scores(scores, filename=fname)

    # ------------------------------------------------------------------ #
    # Single file mode                                                    #
    # ------------------------------------------------------------------ #
    elif path.exists():
        try:
            scores = whisqa.predict(
                str(path),
                model_type=args.model,
                warn_resample=warn_resample,
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        _print_scores(scores)

    else:
        print(f"Error: {path} is not a file or directory.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
