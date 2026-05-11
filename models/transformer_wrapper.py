import math

import torch
from torch import Tensor, nn

try:
    from transformer_config import Config
except ImportError:
    from models.transformer_config import Config


class TransformerWrapper(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.norm = nn.BatchNorm1d(config.dim_input)
        self.position_encoding = PositionalEncoding(config)
        self.linear_proj = nn.Linear(config.dim_input, config.dim_transformer)
        self.linear_proj_drop = nn.Dropout(config.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_transformer,
            dim_feedforward=config.dim_transformer * 2,
            nhead=config.nhead_transformer,
            batch_first=True,
            dropout=config.dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.nlayers_transformer
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        x = self.norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.linear_proj(x)
        x = self.linear_proj_drop(x)
        x = self.position_encoding(x)
        x = self.transformer_encoder(x, mask)
        return self.dropout(x)


class PositionalEncoding(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        d_model = config.dim_transformer
        seq_len = config.feat_seq_len
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(2 * seq_len) / d_model)
        )
        pe = torch.zeros(1, seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe.expand(x.shape)
