#!/usr/bin/env python3
"""
Strip frozen Whisper encoder weights from a full WhiSQA checkpoint.

The checkpoints produced during training include the full Whisper encoder
(~240 MB frozen) alongside the small trainable prediction head. For
distribution the encoder is unnecessary — it is downloaded from
``openai/whisper-small`` on HuggingFace Hub at model-load time anyway.

This script keeps only the head weights, shrinking each checkpoint from
~350 MB to a few MB.  The output files live in ``whisqa/_models/`` and are
bundled directly into the package wheel — no external model hosting required.

Usage
-----
After pulling the full checkpoints via git-lfs::

    git lfs pull

    python scripts/strip_whisper_weights.py \\
        --input checkpoints/single_head_model.pt

    python scripts/strip_whisper_weights.py \\
        --input checkpoints/multi_head_model.pt

Outputs are written to ``whisqa/_models/<filename>`` by default.
Commit the resulting files; they are small enough for regular git.

Loading
-------
The stripped checkpoints are loaded with ``strict=False`` so that missing
Whisper encoder keys are silently ignored — the encoder is initialised
separately from ``openai/whisper-small`` on HuggingFace Hub.
"""
import argparse
from pathlib import Path

import torch

# All keys under this prefix belong to the frozen Whisper encoder.
_WHISPER_PREFIX = "feat_extract.model."


def strip(input_path: Path, output_path: Path) -> None:
    print(f"Loading {input_path} …")
    state_dict = torch.load(input_path, map_location="cpu", weights_only=True)

    head_only = {k: v for k, v in state_dict.items() if not k.startswith(_WHISPER_PREFIX)}

    n_total = len(state_dict)
    n_kept = len(head_only)
    n_removed = n_total - n_kept

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head_only, output_path)

    in_mb = input_path.stat().st_size / 1024**2
    out_mb = output_path.stat().st_size / 1024**2
    print(f"  Tensors : {n_total} total → removed {n_removed} (Whisper encoder), kept {n_kept}")
    print(f"  Size    : {in_mb:.1f} MB → {out_mb:.1f} MB")
    print(f"  Saved   : {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Full checkpoint .pt file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination for stripped .pt file (default: whisqa/_models/<input filename>)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.input}")

    output = args.output or (Path(__file__).parent.parent / "whisqa" / "_models" / args.input.name)
    strip(args.input, output)


if __name__ == "__main__":
    main()
