import torch.nn as nn
from torch.nn import functional as F


class DilatedTransformer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=num_heads)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        x = x.view(batch_size, C, -1).permute(2, 0, 1)  # (W*H, B, C)

        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output  # 残差连接
        x = self.norm1(x)

        x = self.linear2(F.gelu(self.linear1(x)))
        x = x + x  # 残差连接
        x = self.norm2(x)

        return x.permute(1, 2, 0).view(batch_size, C, width, height)  # 恢复维度
