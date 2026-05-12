"""WhiSQA: Whisper-based Speech Quality Assessment."""

import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional

from whisqa._checkpoints import get_checkpoint_stream
from whisqa._models.predictors import MultiHeadPredictor, SingleHeadPredictor

__version__ = "0.1.5"
__all__ = ["predict", "load_model", "WhiSQA"]

_DIMENSIONS = ["mos", "noisiness", "coloration", "discontinuity", "loudness"]

_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".au"}

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
        return torch.from_numpy(data.T), sr

    if sample_rate is None:
        warnings.warn(
            "sample_rate was not provided for array/tensor input — assuming 16000 Hz. "
            "Pass sample_rate explicitly to suppress this warning.",
            UserWarning,
            stacklevel=3,
        )
        sample_rate = 16000

    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(np.asarray(audio, dtype=np.float32))

    if not isinstance(audio, torch.Tensor):
        raise TypeError(
            f"audio must be a file path, numpy array, or torch tensor; got {type(audio).__name__}."
        )

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    elif audio.ndim != 2:
        raise ValueError(
            f"Audio tensor must be 1-D (samples,) or 2-D (1, samples); got shape {tuple(audio.shape)}."
        )

    return audio, sample_rate


def load_model(
    model_type: str = "single",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    """
    Load a WhiSQA model with pre-trained head weights.

    The WhiSQA head weights are bundled with the package (no download needed).
    The Whisper encoder (``openai/whisper-small``, ~240 MB) is downloaded from
    HuggingFace Hub on the first call and cached in ``~/.cache/huggingface/``.

    Args:
        model_type: ``'single'`` for MOS-only, ``'multi'`` for MOS + four
                    P.835 speech quality dimensions.
        dtype:      Floating-point dtype for the Whisper encoder and head
                    weights. Defaults to ``torch.bfloat16``, which halves
                    memory usage with good numerical stability on modern
                    hardware. Use ``torch.float32`` for maximum compatibility.

    Returns:
        An ``eval()``-mode model on the best available device.
    """
    if model_type == "single":
        model = SingleHeadPredictor(dtype=dtype)
    elif model_type == "multi":
        model = MultiHeadPredictor(dtype=dtype)
    else:
        raise ValueError(
            f"Unknown model type {model_type!r}. Choose 'single' or 'multi'."
        )

    device = _get_device()
    stream = get_checkpoint_stream(model_type)
    state_dict = torch.load(stream, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    # Apply dtype only to the encoder — it dominates memory and compute.
    # The head stays float32 to avoid BatchNorm dtype issues on CPU.
    model.feat_extract.to(dtype)
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

                      * **File path** (``str`` or :class:`pathlib.Path`) — any
                        format supported by libsndfile. ``sample_rate`` is read
                        from the file and the ``sample_rate`` argument is ignored.
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
        sample_rate:  Sample rate in Hz. Required for array/tensor inputs;
                      ignored for file paths.
        warn_resample: Emit a :class:`UserWarning` when audio is not 16 kHz.
                       Pass ``False`` to suppress.
        model:        Pre-loaded model from :func:`load_model`. Use
                      :class:`WhiSQA` for repeated inference instead of
                      managing the model object manually.

    Returns:
        ``dict`` with ``'mos'`` and (for ``model_type='multi'``) the four
        P.835 dimensions ``'noisiness'``, ``'coloration'``, ``'discontinuity'``,
        and ``'loudness'``. All values are on the MOS scale 1–5.
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


class WhiSQA:
    """
    Stateful WhiSQA scorer that keeps the model in memory between calls.

    Prefer this over calling :func:`predict` repeatedly when scoring multiple
    files — the Whisper encoder is loaded once in ``__init__`` and reused
    for every subsequent call.

    Args:
        model_type: ``'single'`` (MOS only) or ``'multi'`` (MOS + four
                    P.835 dimensions). Passed to :func:`load_model`.

    Example::

        scorer = whisqa.WhiSQA("single")

        # file path
        scorer.predict("speech.wav")

        # numpy / tensor
        scorer.predict(waveform, sample_rate=16000)

        # whole directory
        results = scorer.predict_dir("/path/to/wavs")
    """

    def __init__(self, model_type: str = "single", dtype: torch.dtype = torch.bfloat16):
        self.model_type = model_type
        self._model = load_model(model_type, dtype=dtype)

    def predict(
        self,
        audio: AudioInput,
        sample_rate: Optional[int] = None,
        warn_resample: bool = True,
    ) -> dict:
        """Score a single audio signal. See :func:`predict` for argument details."""
        return predict(
            audio,
            sample_rate=sample_rate,
            warn_resample=warn_resample,
            model=self._model,
        )

    def predict_dir(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        warn_resample: bool = True,
    ) -> List[dict]:
        """
        Score every audio file in *directory*.

        Supported extensions: ``wav``, ``flac``, ``ogg``, ``aiff``, ``au``.
        Files that cannot be scored are skipped with a warning to stderr.

        Args:
            directory:    Path to a directory.
            recursive:    If ``True`` (default), search sub-directories too.
            warn_resample: Passed through to :meth:`predict` for each file.

        Returns:
            List of result dicts, each with an extra ``'file'`` key containing
            the absolute path string, in the order files were discovered.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"{directory} is not a directory.")

        pattern = "**/*" if recursive else "*"
        files = sorted(
            p for p in directory.glob(pattern)
            if p.is_file() and p.suffix.lower() in _AUDIO_EXTENSIONS
        )

        if not files:
            warnings.warn(
                f"No audio files found in {directory}.",
                UserWarning,
                stacklevel=2,
            )
            return []

        results = []
        for f in files:
            try:
                scores = self.predict(f, warn_resample=warn_resample)
                results.append({"file": str(f), **scores})
            except Exception as exc:
                print(f"Warning: skipping {f.name} — {exc}", file=sys.stderr)

        return results
