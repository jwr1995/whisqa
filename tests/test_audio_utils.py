"""
Tests for audio utility functions that require no model weights.
"""
import numpy as np
import pytest
import torch

from whisqa._models.whisper_wrapper import (
    N_SAMPLES,
    log_mel_spectrogram,
    mel_filters,
    pad_or_trim,
)


class TestPadOrTrim:
    def test_trims_long_tensor(self):
        x = torch.zeros(1, N_SAMPLES * 2)
        result = pad_or_trim(x, N_SAMPLES)
        assert result.shape[-1] == N_SAMPLES

    def test_pads_short_tensor(self):
        x = torch.zeros(1, 8000)
        result = pad_or_trim(x, N_SAMPLES)
        assert result.shape[-1] == N_SAMPLES

    def test_exact_length_unchanged(self):
        x = torch.zeros(1, N_SAMPLES)
        result = pad_or_trim(x, N_SAMPLES)
        assert result.shape[-1] == N_SAMPLES

    def test_padding_is_zero(self):
        x = torch.ones(1, 1000)
        result = pad_or_trim(x, 2000)
        assert result[0, 1000:].sum().item() == 0.0

    def test_trim_preserves_values(self):
        x = torch.arange(100, dtype=torch.float32).unsqueeze(0)
        result = pad_or_trim(x, 50)
        assert torch.equal(result, x[:, :50])


class TestMelFilters:
    def test_returns_tensor(self):
        filters = mel_filters("cpu", 80)
        assert isinstance(filters, torch.Tensor)

    def test_shape(self):
        filters = mel_filters("cpu", 80)
        assert filters.shape[0] == 80

    def test_unsupported_n_mels_raises(self):
        with pytest.raises(AssertionError):
            mel_filters("cpu", 40)

    def test_cached(self):
        # Calling twice with same args returns same object (lru_cache hit)
        f1 = mel_filters("cpu", 80)
        f2 = mel_filters("cpu", 80)
        assert f1 is f2


class TestLogMelSpectrogram:
    @pytest.fixture
    def mono_audio(self):
        return torch.randn(1, N_SAMPLES)

    def test_output_shape(self, mono_audio):
        spec = log_mel_spectrogram(mono_audio)
        # (channels, n_mels, frames) — batched input
        assert spec.shape[-2] == 80

    def test_output_is_finite(self, mono_audio):
        spec = log_mel_spectrogram(mono_audio)
        assert torch.isfinite(spec).all()

    def test_silence_does_not_nan(self):
        silence = torch.zeros(1, N_SAMPLES)
        spec = log_mel_spectrogram(silence)
        assert torch.isfinite(spec).all()
