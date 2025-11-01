import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            embedding_dimensions: int = 512,
            heads_num: int = 8,
        ):
        super(MultiHeadSelfAttention, self).__init__()
        self.embedding_dimensions = embedding_dimensions
        self.heads_num = heads_num
        self.head_dimensions = embedding_dimensions // heads_num

        self.query = nn.Linear(embedding_dimensions, embedding_dimensions)
        self.key = nn.Linear(embedding_dimensions, embedding_dimensions)
        self.value = nn.Linear(embedding_dimensions, embedding_dimensions)

        self.projection = nn.Linear(embedding_dimensions, embedding_dimensions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input:     (batch_size, sequence_length, embedding_dimensions)
        q, k, v:   (batch_size, heads_num, sequence_length, head_dimensions)
        attention: (batch_size, heads_num, sequence_length, sequence_length)
        context:   (batch_size, heads_num, sequence_length, head_dimensions)
        output:    (batch_size, heads_num, embedding_dimensions)
        """
        B, N, C = x.shape

        assert C == self.embedding_dimensions

        q = self.query(x).view(B, N, self.heads_num, self.head_dimensions).permute(0, 2, 1, 3)
        k = self.key(x).view(B, N, self.heads_num, self.head_dimensions).permute(0, 2, 1, 3)
        v = self.value(x).view(B, N, self.heads_num, self.head_dimensions).permute(0, 2, 1, 3)
        
        # attention = softmax(q @ k^T / sqrt(head_dimensions))
        attention = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dimensions ** 0.5)
        attention = torch.softmax(attention, dim=-1)

        # context = attention @ v
        context = torch.matmul(attention, v)
        
        out = context.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        out = self.projection(out)
        return out
