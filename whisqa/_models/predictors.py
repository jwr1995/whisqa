"""
WhiSQA model architectures.

Only the two shipped models are included here. The full research archive
(including reference-based, mel-input, and full Whisper variants) lives in
the original models/ directory.
"""
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from whisqa._models.config import Config, Input
from whisqa._models.transformer import TransformerWrapper
from whisqa._models.whisper_wrapper import WhisperWrapper_encoder


class _PoolAttFF(nn.Module):
    """Attention-pooling module with a feed-forward network."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, 2 * dim)
        self.linear2 = nn.Linear(2 * dim, 1)
        self.linear3 = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor) -> Tensor:
        att = self.linear2(self.dropout(F.relu(self.linear1(x))))
        att = F.softmax(att.transpose(2, 1), dim=2)
        x = torch.bmm(att, x).squeeze(1)
        return self.linear3(x)


def _encoder_config() -> Config:
    return Config(
        "WHISPER_ENCODER_CONFIG",
        Input.XLSR,
        feat_seq_len=1500,
        dim_transformer=256,
        xlsr_name="whisper_encoder",
        nhead_transformer=4,
        nlayers_transformer=4,
    )


class SingleHeadPredictor(nn.Module):
    """
    Single-head MOS predictor.

    Uses a weighted sum over all 13 Whisper-small encoder hidden states,
    followed by a small Transformer and attention pooling, to predict a
    scalar MOS score in [0, 1] (multiply by 5 for the MOS scale).
    """

    def __init__(self, feat_seq: int = 1500, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.norm_input = nn.BatchNorm1d(768)
        self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True, layer=-1, dtype=dtype)
        self.feat_extract.requires_grad_(False)
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.softmax = nn.Softmax(dim=0)
        self.transformer = TransformerWrapper(_encoder_config())
        self.attenPool = _PoolAttFF(256)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        # Encoder runs in its dtype (default bfloat16); cast to float32
        # immediately so the float32 head (BatchNorm, transformer) is consistent.
        feats = self.feat_extract(x).float()
        feats = feats @ self.softmax(self.layer_weights)
        feats = self.norm_input(feats.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.transformer(feats)
        return self.sigmoid(self.attenPool(out))


class MultiHeadPredictor(nn.Module):
    """
    Multi-head predictor for MOS + 4 P.835 speech quality dimensions.

    Outputs 5 scores in [0, 1] along a new dimension:
    [MOS, Noisiness, Coloration, Discontinuity, Loudness].
    Multiply by 5 for the MOS scale.
    """

    def __init__(self, feat_seq: int = 1500, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.norm_input = nn.BatchNorm1d(768)
        self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True, layer=-1, dtype=dtype)
        self.feat_extract.requires_grad_(False)
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.softmax = nn.Softmax(dim=0)
        self.transformer = TransformerWrapper(_encoder_config())
        self.attenPool1 = _PoolAttFF(256)
        self.attenPool2 = _PoolAttFF(256)
        self.attenPool3 = _PoolAttFF(256)
        self.attenPool4 = _PoolAttFF(256)
        self.attenPool5 = _PoolAttFF(256)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        feats = self.feat_extract(x).float()
        feats = feats @ self.softmax(self.layer_weights)
        feats = self.norm_input(feats.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.transformer(feats)
        scores = [self.sigmoid(pool(out))
                  for pool in [self.attenPool1, self.attenPool2,
                               self.attenPool3, self.attenPool4, self.attenPool5]]
        return torch.stack(scores, dim=1)


# Keep original class names as aliases so that old checkpoints (saved with
# the original class names) can still be loaded with strict=False.
whisperMetricPredictorEncoderLayersTransformerSmall = SingleHeadPredictor
whisperMetricPredictorEncoderLayersTransformerSmalldim = MultiHeadPredictor
