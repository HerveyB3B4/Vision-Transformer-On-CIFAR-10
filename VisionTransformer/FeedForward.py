import torch
import torch.nn as nn

class FeedForward(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        activation_layer: nn.Module,
        dropout: float,
    ):
        super(FeedForward, self).__init__(
            nn.Linear(in_features, hidden_features),
            activation_layer(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(dropout)
        )