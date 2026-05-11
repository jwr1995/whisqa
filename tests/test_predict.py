"""
Tests for the public whisqa.predict() and whisqa.load_model() API.

Unit tests use mocked models and do not require network access or checkpoints.
Integration tests (``pytest -m integration``) use real weights.
"""
import warnings

import pytest
import torch

import whisqa
from whisqa._models.predictors import MultiHeadPredictor, SingleHeadPredictor


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------

class TestPredictOutputFormat:
    def test_single_returns_mos_key(self, wav_16k, patch_load_model_single):
        result = whisqa.predict(str(wav_16k))
        assert set(result.keys()) == {"mos"}

    def test_single_mos_in_range(self, wav_16k, patch_load_model_single):
        result = whisqa.predict(str(wav_16k))
        assert 1.0 <= result["mos"] <= 5.0

    def test_multi_returns_all_keys(self, wav_16k, patch_load_model_multi):
        result = whisqa.predict(str(wav_16k), model_type="multi")
        assert set(result.keys()) == {"mos", "noisiness", "coloration", "discontinuity", "loudness"}

    def test_multi_all_values_in_range(self, wav_16k, patch_load_model_multi):
        result = whisqa.predict(str(wav_16k), model_type="multi")
        for key, value in result.items():
            assert 1.0 <= value <= 5.0, f"{key}={value} is outside [1, 5]"

    def test_values_are_rounded(self, wav_16k, patch_load_model_single):
        result = whisqa.predict(str(wav_16k))
        # 4 decimal places
        assert result["mos"] == round(result["mos"], 4)


# ---------------------------------------------------------------------------
# Pre-loaded model passthrough
# ---------------------------------------------------------------------------

class TestPredictWithPreloadedModel:
    def test_accepts_single_model_directly(self, wav_16k, mock_single_model):
        result = whisqa.predict(str(wav_16k), model=mock_single_model)
        assert "mos" in result

    def test_accepts_multi_model_directly(self, wav_16k, mock_multi_model):
        result = whisqa.predict(str(wav_16k), model=mock_multi_model)
        assert "mos" in result
        assert "noisiness" in result

    def test_skips_load_model_when_model_supplied(self, wav_16k, mock_single_model, mocker):
        spy = mocker.patch("whisqa.load_model")
        whisqa.predict(str(wav_16k), model=mock_single_model)
        spy.assert_not_called()


# ---------------------------------------------------------------------------
# Resampling behaviour
# ---------------------------------------------------------------------------

class TestResampling:
    def test_8k_input_emits_warning_by_default(self, wav_8k, patch_load_model_single):
        with pytest.warns(UserWarning, match="resampling to 16000 Hz"):
            whisqa.predict(str(wav_8k))

    def test_8k_input_no_warning_when_suppressed(self, wav_8k, patch_load_model_single):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Should not raise despite 8 kHz input
            whisqa.predict(str(wav_8k), warn_resample=False)

    def test_16k_input_no_warning(self, wav_16k, patch_load_model_single):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            whisqa.predict(str(wav_16k))   # must not raise

    def test_warning_is_userwarning_subclass(self, wav_8k, patch_load_model_single):
        with pytest.warns(UserWarning):
            whisqa.predict(str(wav_8k))

    def test_8k_still_produces_valid_output(self, wav_8k, patch_load_model_single):
        result = whisqa.predict(str(wav_8k), warn_resample=False)
        assert "mos" in result


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_stereo_raises_value_error(self, wav_stereo, patch_load_model_single):
        with pytest.raises(ValueError, match="mono"):
            whisqa.predict(str(wav_stereo))

    def test_missing_file_raises(self, patch_load_model_single):
        with pytest.raises(Exception):
            whisqa.predict("/nonexistent/path/audio.wav")

    def test_unknown_model_type_raises(self, wav_16k):
        with pytest.raises(ValueError, match="Unknown model type"):
            whisqa.predict(str(wav_16k), model_type="banana")


# ---------------------------------------------------------------------------
# Integration tests — require real checkpoints from HuggingFace Hub
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegration:
    def test_single_predict_clean_speech(self, clean_wav):
        result = whisqa.predict(str(clean_wav))
        assert 1.0 <= result["mos"] <= 5.0

    def test_multi_predict_clean_speech(self, clean_wav):
        result = whisqa.predict(str(clean_wav), model_type="multi")
        assert set(result.keys()) == {"mos", "noisiness", "coloration", "discontinuity", "loudness"}
        for v in result.values():
            assert 1.0 <= v <= 5.0

    def test_noisy_scores_lower_than_clean(self, clean_wav, noisy_wav):
        clean = whisqa.predict(str(clean_wav))
        noisy = whisqa.predict(str(noisy_wav))
        assert noisy["mos"] < clean["mos"], "Expected lower MOS for noisy speech"

    def test_load_model_single(self):
        model = whisqa.load_model("single")
        assert isinstance(model, SingleHeadPredictor)

    def test_load_model_multi(self):
        model = whisqa.load_model("multi")
        assert isinstance(model, MultiHeadPredictor)

    def test_preloaded_model_matches_predict(self, clean_wav):
        model = whisqa.load_model("single")
        r1 = whisqa.predict(str(clean_wav), model=model)
        r2 = whisqa.predict(str(clean_wav), model=model)
        assert r1 == r2  # deterministic
