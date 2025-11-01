import torch
import torch.nn as nn

from PatchEmbedding import PatchEmbedding

class VisionTransformer(nn.Module):
    def __init__(
            self,
            image_size: int = 32,
            in_channels: int = 3,
            patch_size: int = 4,
            embedding_dimensions: int = 512,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)

        batch_size = x.shape[0]
        cls_token = self.cls_token.repeat(batch_size, 1, 1)
        x = torch.cat((cls_token, x), dim=1)

        x += self.position_embedding

        return x