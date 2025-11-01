import torch
import torch.nn as nn

class FeedForward(nn.Sequential):
    def __init__(
        self,
        in_features: int = 8,
        hidden_features: int = 1024,
        out_features: int = 8,
        activation_layer: nn.Module = nn.GELU,
        dropout: float = 0.1,
    ):
        super(FeedForward, self).__init__(
            nn.Linear(in_features, hidden_features),
            activation_layer(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(dropout)
        )