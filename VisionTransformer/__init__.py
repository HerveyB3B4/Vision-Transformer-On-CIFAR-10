import torch
import torch.nn as nn

from PatchEmbedding import PatchEmbedding
from TransformerEncoderLayer import TransformerEncoderLayer

class VisionTransformer(nn.Module):
    def __init__(
            self,
            image_size: int = 32,
            in_channels: int = 3,
            patch_size: int = 4,
            embedding_dimensions: int = 512,
            heads_num: int = 8,
            mlp_dimensions: int = 1024,
            dropout: float = 0.1,
            transformer_layers_num: int = 6,
            classes_num: int = 10,
            embedding_layer: PatchEmbedding = PatchEmbedding
        ):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = embedding_layer(
            image_size=image_size,
            in_channels=in_channels,
            patch_size=patch_size,
            embedding_dimensions=embedding_dimensions
        )
        self.patches_num = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dimensions))
        self.position_embedding = nn.Parameter(torch.randn(1, 1 + self.patches_num, embedding_dimensions))
        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    embedding_dimensions=embedding_dimensions,
                    heads_num=heads_num,
                    mlp_dimensions=mlp_dimensions,
                    dropout=dropout,
                )
                for _ in range(transformer_layers_num)
            ]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedding_dimensions),
            nn.Linear(embedding_dimensions, classes_num),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)

        batch_size = x.shape[0]
        cls_token = self.cls_token.repeat(batch_size, 1, 1)
        x = torch.cat((cls_token, x), dim=1)

        x += self.position_embedding

        x = self.transformer_encoder(x)

        logits = self.mlp_head(x[:, 0])

        return logits