"""WhiSQA: Whisper-based Speech Quality Assessment."""

import os
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional

from whisqa._checkpoints import get_checkpoint_stream
from whisqa._models.predictors import MultiHeadPredictor, SingleHeadPredictor

__version__ = "0.1.4"
__all__ = ["predict", "load_model"]

_DIMENSIONS = ["mos", "noisiness", "coloration", "discontinuity", "loudness"]

AudioInput = Union[str, os.PathLike, np.ndarray, torch.Tensor]


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_audio(
    audio: AudioInput,
    sample_rate: Optional[int],
) -> tuple:
    """Return ``(waveform, sample_rate)`` with waveform shape ``(1, samples)``."""
    if isinstance(audio, (str, os.PathLike)):
        data, sr = sf.read(audio, dtype="float32", always_2d=True)
        # soundfile returns (samples, channels); transpose to (channels, samples)
        return torch.from_numpy(data.T), sr

    if sample_rate is None:
        raise ValueError(
            "sample_rate must be provided when audio is a numpy array or torch tensor."
        )

    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(np.asarray(audio, dtype=np.float32))

    if not isinstance(audio, torch.Tensor):
        raise TypeError(
            f"audio must be a file path, numpy array, or torch tensor; got {type(audio).__name__}."
        )

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)  # (samples,) → (1, samples)
    elif audio.ndim != 2:
        raise ValueError(
            f"Audio tensor must be 1-D (samples,) or 2-D (1, samples); got shape {tuple(audio.shape)}."
        )

    return audio, sample_rate


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
    audio: AudioInput,
    model_type: str = "single",
    sample_rate: Optional[int] = None,
    warn_resample: bool = True,
    model: Optional[torch.nn.Module] = None,
) -> dict:
    """
    Predict speech quality scores for an audio signal.

    Args:
        audio:        Audio to score. Accepts:

                      * **File path** (``str`` or :class:`pathlib.Path`) — WAV or
                        any format supported by libsndfile. ``sample_rate`` is
                        read from the file and the ``sample_rate`` argument is
                        ignored.
                      * **NumPy array** — shape ``(samples,)`` or ``(1, samples)``,
                        float32. ``sample_rate`` must be provided.
                      * **Torch tensor** — shape ``(samples,)`` or ``(1, samples)``.
                        ``sample_rate`` must be provided.

                      All inputs must be mono. Any sample rate is accepted;
                      audio is resampled to 16 kHz automatically if needed.

        model_type:   ``'single'`` (MOS only) or ``'multi'`` (MOS + Noisiness,
                      Coloration, Discontinuity, Loudness). Ignored when
                      ``model`` is supplied — the supplied model's type takes
                      precedence. A :class:`UserWarning` is emitted if the
                      two conflict.
        sample_rate:  Sample rate of the audio in Hz. Required when ``audio``
                      is a numpy array or torch tensor; ignored for file paths.
        warn_resample: Emit a :class:`UserWarning` when the audio is not 16 kHz
                       and will be resampled. Pass ``False`` to suppress.
                       Respects Python's :mod:`warnings` filters.
        model:        Pre-loaded model returned by :func:`load_model`. Pass
                      this for efficient repeated inference to avoid
                      re-loading weights on every call.

    Returns:
        ``dict`` with ``'mos'`` and (for ``model_type='multi'``) the four
        P.835 dimensions ``'noisiness'``, ``'coloration'``, ``'discontinuity'``,
        and ``'loudness'``. All values are on the MOS scale 1–5.

    Examples::

        >>> import whisqa
        >>> whisqa.predict("speech.wav")
        {'mos': 3.82}

        >>> import numpy as np
        >>> whisqa.predict(np.zeros(16000), sample_rate=16000)
        {'mos': ...}

        >>> import torch
        >>> whisqa.predict(torch.zeros(1, 16000), sample_rate=16000)
        {'mos': ...}
    """
    waveform, sr = _load_audio(audio, sample_rate)

    if waveform.shape[0] != 1:
        raise ValueError(
            f"Audio must be mono (1 channel), got {waveform.shape[0]} channels."
        )

    if sr != 16000:
        if warn_resample:
            warnings.warn(
                f"Input sample rate is {sr} Hz; resampling to 16000 Hz. "
                "Pass warn_resample=False to suppress this warning.",
                UserWarning,
                stacklevel=2,
            )
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    if model is None:
        model = load_model(model_type)
    else:
        _actual = "multi" if isinstance(model, MultiHeadPredictor) else "single"
        if model_type != "single" and model_type != _actual:
            warnings.warn(
                f"model_type={model_type!r} was supplied but the provided model is a "
                f"{type(model).__name__} ({_actual!r}). model_type is ignored when "
                "model= is given; the supplied model's type takes precedence.",
                UserWarning,
                stacklevel=2,
            )

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
