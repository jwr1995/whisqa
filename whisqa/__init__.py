"""WhiSQA: Whisper-based Speech Quality Assessment."""

import warnings
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional

from whisqa._checkpoints import get_checkpoint_stream
from whisqa._models.predictors import MultiHeadPredictor, SingleHeadPredictor

__version__ = "0.1.0"
__all__ = ["predict", "load_model"]

_DIMENSIONS = ["mos", "noisiness", "coloration", "discontinuity", "loudness"]


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_type: str = "single") -> torch.nn.Module:
    """
    Load a WhiSQA model with pre-trained head weights.

    The WhiSQA head weights are bundled with the package (no download needed).
    The Whisper encoder (``openai/whisper-small``, ~240 MB) is downloaded from
    HuggingFace Hub on the first call and cached in ``~/.cache/huggingface/``.

    Args:
        model_type: ``'single'`` for MOS-only, ``'multi'`` for MOS + four
                    P.835 speech quality dimensions.

    Returns:
        An ``eval()``-mode model on the best available device.
    """
    if model_type == "single":
        model = SingleHeadPredictor()
    elif model_type == "multi":
        model = MultiHeadPredictor()
    else:
        raise ValueError(
            f"Unknown model type {model_type!r}. Choose 'single' or 'multi'."
        )

    device = _get_device()
    stream = get_checkpoint_stream(model_type)
    state_dict = torch.load(stream, map_location=device, weights_only=True)
    # strict=False: Whisper encoder keys are absent in head-only checkpoints
    # and are already initialised from HuggingFace Hub above.
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


def predict(
    audio_file: str,
    model_type: str = "single",
    warn_resample: bool = True,
    model: Optional[torch.nn.Module] = None,
) -> dict:
    """
    Predict speech quality scores for an audio file.

    Args:
        audio_file:   Path to a WAV file. Must be mono; any sample rate is
                      accepted (audio is resampled to 16 kHz if needed).
        model_type:   ``'single'`` (MOS only) or ``'multi'`` (MOS + Noisiness,
                      Coloration, Discontinuity, Loudness). Ignored when
                      ``model`` is supplied — the supplied model's type takes
                      precedence. A :class:`UserWarning` is emitted if the
                      two conflict.
        warn_resample: Emit a :class:`UserWarning` when the file is not
                       16 kHz and will be resampled. Pass ``False`` to
                       suppress. Respects Python's :mod:`warnings` filters.
        model:        Pre-loaded model returned by :func:`load_model`. Pass
                      this for efficient repeated inference to avoid
                      re-loading weights on every call.

    Returns:
        ``dict`` with ``'mos'`` and (for ``model_type='multi'``) the four
        P.835 dimensions ``'noisiness'``, ``'coloration'``, ``'discontinuity'``,
        and ``'loudness'``. All values are on the MOS scale 1–5.

    Example::

        >>> import whisqa
        >>> whisqa.predict("speech.wav")
        {'mos': 3.82}
        >>> m = whisqa.load_model("multi")
        >>> whisqa.predict("speech.wav", model_type="multi", model=m)
        {'mos': 3.82, 'noisiness': 4.1, 'coloration': 3.6, ...}
    """
    data, sample_rate = sf.read(audio_file, dtype="float32", always_2d=True)
    # soundfile returns (samples, channels); convert to (channels, samples)
    waveform = torch.from_numpy(data.T)

    if waveform.shape[0] != 1:
        raise ValueError(
            f"Audio must be mono (1 channel), got {waveform.shape[0]} channels."
        )

    if sample_rate != 16000:
        if warn_resample:
            warnings.warn(
                f"Input sample rate is {sample_rate} Hz; resampling to 16000 Hz. "
                "Pass warn_resample=False to suppress this warning.",
                UserWarning,
                stacklevel=2,
            )
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

    if model is None:
        model = load_model(model_type)
    else:
        # Infer actual type from the object; warn if caller also passed model_type
        # with a value that contradicts the supplied model.
        _actual = "multi" if isinstance(model, MultiHeadPredictor) else "single"
        if model_type != "single" and model_type != _actual:
            warnings.warn(
                f"model_type={model_type!r} was supplied but the provided model is a "
                f"{type(model).__name__} ({_actual!r}). model_type is ignored when "
                "model= is given; the supplied model's type takes precedence.",
                UserWarning,
                stacklevel=2,
            )

    # Infer model_type from the loaded model when one is supplied directly.
    _is_multi = isinstance(model, MultiHeadPredictor)

    device = next(model.parameters()).device
    waveform = waveform.to(device)

    with torch.no_grad():
        score = model(waveform)

    if _is_multi:
        score = score.squeeze(0)
        return {dim: round(score[i].item() * 5, 4) for i, dim in enumerate(_DIMENSIONS)}
    else:
        return {"mos": round(score.item() * 5, 4)}
