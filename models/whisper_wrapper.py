from functools import lru_cache
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import WhisperFeatureExtractor, WhisperModel

SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE


def pad_or_trim(array: torch.Tensor, length: int = N_SAMPLES, *, axis: int = -1) -> torch.Tensor:
    """Pad or trim a tensor to exactly `length` samples along `axis`."""
    if array.shape[axis] > length:
        array = array.index_select(axis, torch.arange(length, device=array.device))
    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = F.pad(array, [p for sizes in pad_widths[::-1] for p in sizes])
    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load("models/mel_filters.npz", allow_pickle=True) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[np.ndarray, torch.Tensor],
    n_mels: int = N_MELS,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


class WhisperWrapper_encoder(nn.Module):
    """Whisper-small encoder, used as a frozen feature extractor."""

    def __init__(
        self,
        layer=None,
        use_feat_extractor: bool = False,
        pretrained_model: Optional[str] = None,
    ):
        super().__init__()
        self.use_feat_extractor = use_feat_extractor
        self.layer = layer

        if not use_feat_extractor:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                "openai/whisper-small"
            )
        model_name = pretrained_model or "openai/whisper-small"
        self.model = WhisperModel.from_pretrained(model_name).encoder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if self.use_feat_extractor:
            data_padded = pad_or_trim(data, length=N_SAMPLES).to(self.device)
            data_feats = log_mel_spectrogram(data_padded)
        else:
            d_list = [d.to("cpu").tolist() for d in data]
            data = self.feature_extractor(d_list, sampling_rate=16000, return_tensors="pt")
            data_feats = data.input_features.to(self.device)

        if self.layer is None:
            return self.model(input_features=data_feats, return_dict=True)[0]
        elif self.layer == -1:
            out = self.model(
                input_features=data_feats,
                return_dict=True,
                output_hidden_states=True,
            )
            return torch.stack(list(out.hidden_states), dim=-1)
        else:
            out = self.model(
                input_features=data_feats,
                return_dict=True,
                output_hidden_states=True,
            )
            return out.hidden_states[self.layer]
