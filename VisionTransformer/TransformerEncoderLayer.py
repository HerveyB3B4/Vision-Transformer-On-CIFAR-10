import torch
import torch.nn as nn

from MultiHeadSelfAttention import MultiHeadSelfAttention
from FeedForward import FeedForward

class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            embedding_dimensions: int = 512,
            heads_num: int = 8,
            mlp_dimensions: int = 1024,
            dropout: float = 0.1,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadSelfAttention(
            embedding_dimensions = embedding_dimensions,
            heads_num = heads_num,
        )
        self.normalization_layer_1 = nn.LayerNorm(embedding_dimensions)
        self.normalization_layer_2 = nn.LayerNorm(embedding_dimensions)

        self.mlp = FeedForward(
            in_features=embedding_dimensions,
            hidden_features=mlp_dimensions,
            out_features=embedding_dimensions,
            activation_layer=nn.GELU,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.attention(self.normalization_layer_1(x))
        x = x + attention
        mlp = self.mlp(self.normalization_layer_2(x))
        x = x + mlp
        return x