"""
Shared fixtures for the WhiSQA test suite.

Unit tests (no marker) mock model loading so they run without downloading
Whisper or head checkpoints.  Integration tests (marked ``integration``)
use the real stack and require network access on first run.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Paths to bundled test audio (from ludlows/PESQ, MIT licence — see ATTRIBUTION.md)
# ---------------------------------------------------------------------------
AUDIO_DIR = Path(__file__).parent / "audio"
CLEAN_WAV = AUDIO_DIR / "speech.wav"           # 16 kHz, mono, ~3.1 s, clean speech
NOISY_WAV = AUDIO_DIR / "speech_bab_0dB.wav"  # 16 kHz, mono, ~3.1 s, babble noise


# ---------------------------------------------------------------------------
# Synthetic audio helpers
# ---------------------------------------------------------------------------

def make_wav(path: Path, sample_rate: int = 16000, duration: float = 1.0, channels: int = 1) -> Path:
    """Write a sine-wave WAV to *path* and return it."""
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, dtype=np.float32)
    wave = np.sin(2 * np.pi * 440 * t)
    if channels > 1:
        wave = np.stack([wave] * channels, axis=-1)
    sf.write(str(path), wave, sample_rate)
    return path


@pytest.fixture
def clean_wav() -> Path:
    return CLEAN_WAV


@pytest.fixture
def noisy_wav() -> Path:
    return NOISY_WAV


@pytest.fixture
def wav_16k(tmp_path) -> Path:
    return make_wav(tmp_path / "sine_16k.wav", sample_rate=16000)


@pytest.fixture
def wav_8k(tmp_path) -> Path:
    """8 kHz file — used to test the auto-resample path."""
    return make_wav(tmp_path / "sine_8k.wav", sample_rate=8000)


@pytest.fixture
def wav_stereo(tmp_path) -> Path:
    """Stereo file — used to test the channel-count error path."""
    return make_wav(tmp_path / "stereo.wav", channels=2)


# ---------------------------------------------------------------------------
# Mock model fixtures
# ---------------------------------------------------------------------------

def _make_mock_single_model():
    """Returns a mock that mimics SingleHeadPredictor output."""
    from whisqa._models.predictors import SingleHeadPredictor

    _param = torch.tensor([0.0])

    model = MagicMock(spec=SingleHeadPredictor)
    model.return_value = torch.tensor([[0.6]])   # → MOS 3.0
    model.parameters.side_effect = lambda: iter([_param])
    return model


def _make_mock_multi_model():
    """Returns a mock that mimics MultiHeadPredictor output."""
    from whisqa._models.predictors import MultiHeadPredictor

    _param = torch.tensor([0.0])

    model = MagicMock(spec=MultiHeadPredictor)
    # shape (1, 5, 1) — same as real forward output before squeeze
    model.return_value = torch.ones(1, 5, 1) * 0.7
    model.parameters.side_effect = lambda: iter([_param])
    return model


@pytest.fixture
def mock_single_model():
    return _make_mock_single_model()


@pytest.fixture
def mock_multi_model():
    return _make_mock_multi_model()


@pytest.fixture
def patch_load_model_single(mock_single_model):
    """Patch whisqa.load_model to return the mock single-head model."""
    with patch("whisqa.load_model", return_value=mock_single_model):
        yield mock_single_model


@pytest.fixture
def patch_load_model_multi(mock_multi_model):
    """Patch whisqa.load_model to return the mock multi-head model."""
    with patch("whisqa.load_model", return_value=mock_multi_model):
        yield mock_multi_model
