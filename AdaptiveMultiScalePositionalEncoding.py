import torch
import torch.nn as nn
import numpy as np

class AdaptiveMultiScalePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(AdaptiveMultiScalePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        # Learnable parameter to balance coarse and detailed encodings
        self.alpha = nn.Parameter(torch.zeros(1))

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        
        pe_coarse = torch.zeros(max_len, d_model)
        pe_coarse[:, 0::2] = torch.sin(position * div_term * 10)
        pe_coarse[:, 1::2] = torch.cos(position * div_term * 10)
        self.register_buffer('pe_coarse', pe_coarse.unsqueeze(0))

        pe_detail = torch.zeros(max_len, d_model)
        pe_detail[:, 0::2] = torch.sin(position * div_term)
        pe_detail[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe_detail', pe_detail.unsqueeze(0))

    def forward(self, x, detail_level=0.5):
        """
        x: Input embeddings (batch_size, seq_len, d_model)
        detail_level: Threshold to determine the level of detail needed (0 to 1)
        """
        seq_len = x.size(1)
        alpha = torch.sigmoid(self.alpha)

        adaptive_pe = alpha * self.pe_coarse[:, :seq_len] + (1 - alpha) * self.pe_detail[:, :seq_len] * detail_level

        x = x + adaptive_pe
        return x
