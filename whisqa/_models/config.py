from enum import Enum

import torch
from torch.nn.functional import pad


class Input(Enum):
    MFCC = 0
    XLSR = 1


class CenterCrop(torch.nn.Module):
    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        unsqueezed = False
        if x.dim() == 2:
            unsqueezed = True
            x = x.unsqueeze(0)
        assert x.dim() == 3  # N, L, C

        if x.size(1) > self.seq_len:
            center_start = int(x.size(1) / 2 - self.seq_len / 2)
            x = x[:, center_start : center_start + self.seq_len, :]
        if x.size(1) < self.seq_len:
            to_pad = self.seq_len - x.size(1)
            x = pad(x, (0, 0, 0, to_pad, 0, 0), mode="constant", value=0.0)

        if unsqueezed:
            x = x.squeeze(0)
        return x


class Config:
    name: str = None
    input: Input = None
    feat_seq_len: int = None
    dim_input: int = None
    dim_transformer: int = None
    dim_head_in: int = None
    dim_head_out: int = None

    def __init__(
        self,
        name: str,
        input: Input,
        feat_seq_len: int,
        dim_transformer: int = None,
        xlsr_name: str = None,
        nhead_transformer: int = 4,
        nlayers_transformer: int = 2,
    ):
        if input == Input.MFCC:
            xlsr_name = None

        assert feat_seq_len > 0, "feat_seq_len must be positive."

        self.name = name
        self.input = input
        self.feat_seq_len = feat_seq_len
        self.dim_transformer = dim_transformer
        self.xlsr_name = xlsr_name
        self.nhead_transformer = nhead_transformer
        self.nlayers_transformer = nlayers_transformer

        _dim_by_name = {
            "whisper_encoder": 768,
            "whisper_encoder_ref": 768 * 2,
            "whisper_encoder_t": 1500,
            "whisper_full": 768,
            "whisper_full_t": 384,
            "hubert_encoder": 512,
            "hubert_encoder_t": 384,
            "hubert_full": 768,
            "hubert_full_t": 384,
            "wav2vec2-xls-r-300m": 1024,
            "wav2vec2-xls-r-1b": 1280,
            "wav2vec2-xls-r-2b": 1920,
        }

        if xlsr_name is not None:
            self.dim_input = _dim_by_name[xlsr_name]
            self.xlsr_layers = -1  # unused in package models
        else:
            # MFCC / mel input
            self.xlsr_layers = None
            self.dim_input = 80 if feat_seq_len != 80 else 3000

        self.dim_head_in = self.dim_transformer
        self.dim_head_out = 1
        self.dropout = 0.1
