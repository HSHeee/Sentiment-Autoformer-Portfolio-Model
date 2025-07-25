# modeling/nsautoformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SeriesDecomp(nn.Module):
    """Moving average로 시계열 분해 (Trend + Seasonal)"""
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x):
        moving_mean = F.avg_pool1d(x.permute(0,2,1), self.kernel_size, stride=1, padding=self.padding)
        moving_mean = moving_mean.permute(0,2,1)
        seasonal = x - moving_mean
        return seasonal, moving_mean


class DeStationaryAttention(nn.Module):
    """정규화 + 비정상성 보정 + Attention"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, L, self.n_heads, -1).transpose(1, 2)
        k = k.view(B, L, self.n_heads, -1).transpose(1, 2)
        v = v.view(B, L, self.n_heads, -1).transpose(1, 2)

        # 기본 attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(attn_output)


class NSAutoformerBlock(nn.Module):
    def __init__(self, d_model, kernel_size, n_heads):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size)
        self.attn = DeStationaryAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        seasonal, trend = self.decomp(x)
        x = self.attn(seasonal) + seasonal
        x = self.ff(x) + x
        return x + trend


class NSAutoformer(nn.Module):
    def __init__(self, input_dim, d_model=64, kernel_size=25, n_heads=4, n_blocks=2, out_len=1):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            NSAutoformerBlock(d_model, kernel_size, n_heads) for _ in range(n_blocks)
        ])
        self.projection = nn.Linear(d_model, out_len)

    def forward(self, x):
        """
        Parameters:
        - x: (B, T, F) → F는 feature 개수, T는 시계열 길이

        Returns:
        - forecast: (B, out_len)
        """
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.projection(x[:, -1, :])  # 마지막 시점만 예측
        return x
