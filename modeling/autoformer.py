# modeling/autoformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 시계열 분해 모듈
class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size=25):
        super(SeriesDecomp, self).__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        trend = self.avg_pool(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend

# 2. Auto-Correlation Mechanism
class AutoCorrelation(nn.Module):
    def __init__(self, dropout=0.1):
        super(AutoCorrelation, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, qk, v):
        freq_x = torch.fft.fft(qk, dim=2)
        power_x = freq_x * torch.conj(freq_x)
        auto_corr = torch.fft.ifft(power_x, dim=2).real
        attn_weights = F.softmax(auto_corr, dim=2)
        attn_weights = self.dropout(attn_weights)
        print("attn_weights:", attn_weights.shape)
        print("v:", v.shape)
        output = torch.matmul(attn_weights, v)
        return output

# 3. Multi-Head Auto-Correlation Layer
class AutoCorrelationLayer(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super(AutoCorrelationLayer, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.auto_corr = AutoCorrelation(dropout)

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = self.auto_corr(q + k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(attn_out)

# 4. Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.auto_corr = AutoCorrelationLayer(d_model)
        self.series_decomp = SeriesDecomp()
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.auto_corr(x)
        x = self.norm1(x + residual)

        seasonal, trend = self.series_decomp(x)
        x = seasonal

        residual = x
        x = self.feedforward(x)
        x = self.norm2(x + residual)
        return x

# 5. Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.auto_corr = AutoCorrelationLayer(d_model)
        self.series_decomp = SeriesDecomp()
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output):
        residual = x
        x = self.auto_corr(x)
        x = self.norm1(x + residual)

        seasonal, trend = self.series_decomp(x)
        x = seasonal

        residual = x
        x = self.feedforward(x)
        x = self.norm2(x + residual)
        return x

# 6. 전체 Autoformer 모델
class Autoformer(nn.Module):
    def __init__(self, input_dim, d_model, output_dim, num_enc_layers=2, num_dec_layers=1, dropout=0.1):
        super(Autoformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, dropout) for _ in range(num_enc_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, dropout) for _ in range(num_dec_layers)
        ])

    def forward(self, x_enc, x_dec):
        x_enc = self.input_proj(x_enc)
        x_dec = self.input_proj(x_dec)

        for layer in self.encoder_layers:
            x_enc = layer(x_enc)

        x = x_dec
        for layer in self.decoder_layers:
            x = layer(x, x_enc)

        output = self.output_proj(x)
        return output
