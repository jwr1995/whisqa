import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from whisper_wrapper import WhisperWrapper_encoder
    from transformer_wrapper import TransformerWrapper
    from transformer_config import Config, Input
except ImportError:
    from models.whisper_wrapper import WhisperWrapper_encoder
    from models.transformer_wrapper import TransformerWrapper
    from models.transformer_config import Config, Input


class PoolAttFF(nn.Module):
    """Attention-pooling with a feed-forward network."""

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
    Non-intrusive MOS predictor (single output).

    Uses a weighted sum of all 13 Whisper-small encoder hidden states,
    a small Transformer encoder, and attention pooling to predict a scalar
    MOS score in [0, 1].  Multiply by 5 to get the standard MOS scale.

    This is the primary model described in the paper.
    """

    def __init__(self, feat_seq: int = 1500):
        super().__init__()
        self.norm_input = nn.BatchNorm1d(768)
        self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True, layer=-1)
        self.feat_extract.requires_grad_(False)
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.softmax = nn.Softmax(dim=0)
        self.transformer = TransformerWrapper(_encoder_config())
        self.attn_pool = PoolAttFF(256)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        feats = self.feat_extract(x)                                      # (B, 1500, 512, 13)
        feats = feats @ self.softmax(self.layer_weights)                  # (B, 1500, 768)
        feats = self.norm_input(feats.permute(0, 2, 1)).permute(0, 2, 1) # normalise
        out = self.transformer(feats)                                      # (B, 1500, 256)
        return self.sigmoid(self.attn_pool(out))                          # (B, 1)


class MultiHeadPredictor(nn.Module):
    """
    Non-intrusive predictor for MOS + 4 P.835 quality dimensions.

    Shares the same encoder stack as SingleHeadPredictor but uses five
    independent attention-pooling heads to produce scores for:
    MOS, Noisiness, Coloration, Discontinuity, Loudness.

    Output shape: (B, 5, 1), values in [0, 1].  Multiply by 5 for MOS scale.
    """

    def __init__(self, feat_seq: int = 1500):
        super().__init__()
        self.norm_input = nn.BatchNorm1d(768)
        self.feat_extract = WhisperWrapper_encoder(use_feat_extractor=True, layer=-1)
        self.feat_extract.requires_grad_(False)
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.softmax = nn.Softmax(dim=0)
        self.transformer = TransformerWrapper(_encoder_config())
        self.attn_pools = nn.ModuleList([PoolAttFF(256) for _ in range(5)])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        feats = self.feat_extract(x)
        feats = feats @ self.softmax(self.layer_weights)
        feats = self.norm_input(feats.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.transformer(feats)
        scores = [self.sigmoid(pool(out)) for pool in self.attn_pools]
        return torch.stack(scores, dim=1)                                  # (B, 5, 1)


# Legacy aliases so that old checkpoints (saved under the original long names)
# can still be loaded with model.load_state_dict(...).
whisperMetricPredictorEncoderLayersTransformerSmall = SingleHeadPredictor
whisperMetricPredictorEncoderLayersTransformerSmalldim = MultiHeadPredictor
