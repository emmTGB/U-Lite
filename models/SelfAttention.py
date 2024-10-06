import torch
import torch.nn as nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()

        query = self.query_conv(x).view(batch_size, -1, width * height)  # (B, C/8, W*H)
        key = self.key_conv(x).view(batch_size, -1, width * height)  # (B, C/8, W*H)
        value = self.value_conv(x).view(batch_size, -1, width * height)  # (B, C, W*H)

        attention = torch.bmm(query.permute(0, 2, 1), key)  # (B, W*H, W*H)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, W*H)
        out = out.view(batch_size, C, width, height)  # 恢复维度
        out = self.gamma * out + x  # 残差连接
        return out
