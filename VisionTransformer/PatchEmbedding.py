import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        in_channels: int = 3,
        patch_size: int = 4,
        embedding_dimensions: int = 512,
    ):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.patches_num = self.grid_size ** 2
        self.projection = nn.Conv2d(
            in_channels,
            out_channels=embedding_dimensions,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        origin:  (batch, image_size, image_size, in_channels)
        proj: -> (batch, embedding_dimensions, grid_size, grid_size)
        flat: -> (batch, embedding_dimensions, grid_size * grid_size)
        tran: -> (batch, grid_size * grid_size, embedding_dimensions)
        """
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x