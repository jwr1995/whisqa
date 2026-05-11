import io
from importlib.resources import files

_FILENAMES = {
    "single": "single_head_model.pt",
    "multi": "multi_head_model.pt",
}


def get_checkpoint_stream(model_type: str) -> io.BytesIO:
    """
    Return the head-only checkpoint for *model_type* as a BytesIO stream.

    Checkpoints live in ``whisqa/_models/`` and are bundled with the package,
    so no network access is required at inference time.
    """
    if model_type not in _FILENAMES:
        raise ValueError(
            f"Unknown model type {model_type!r}. Choose from: {list(_FILENAMES)}"
        )
    ref = files("whisqa._models").joinpath(_FILENAMES[model_type])
    try:
        return io.BytesIO(ref.read_bytes())
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Checkpoint '{_FILENAMES[model_type]}' not found inside the package. "
            "If you are working from source, run the strip script first:\n"
            "  python scripts/strip_whisper_weights.py "
            "--input checkpoints/single_head_model.pt "
            "--output whisqa/_models/single_head_model.pt"
        )
