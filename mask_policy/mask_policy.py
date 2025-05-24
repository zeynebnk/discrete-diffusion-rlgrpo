import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MaskPolicy(nn.Module):
    def __init__(self, in_size=4096, hidden_size=256, n_layers=2, n_heads=4):
        super().__init__()

        ## simple tf architecture
        self.proj = nn.Linear(in_size, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
         
    def forward(self, inputs):
            
        X = self.proj(inputs)  # bs, gen_len, hidden_size
        X = self.transformer(X)  # bs, gen_len, hidden_size
        
        out = self.output_layer(X).squeeze(-1)  # bs, gen_len
        
        return out



from modeling_llada import (
    ModelConfig,
    BufferCache,
    RotaryEmbedding,
    LLaDABlock,
    _non_meta_init_device
)

class MaskPolicy_LLaDA(nn.Module):
    def __init__(self, in_size=4096, hidden_size=256, n_layers=2, n_heads=4):
        super().__init__()
        
        self.config = ModelConfig(
            d_model=hidden_size,
            n_heads=n_heads,
            rope=True, 
            rope_theta=10000.0,
            max_sequence_length=2048
        )
        self.buffer_cache = BufferCache()
        
        self.proj = nn.Linear(in_size, hidden_size)
        
        # transformer 
        self.blocks = nn.ModuleList([
            LLaDABlock.build(i, self.config, self.buffer_cache)
            for i in range(n_layers)
        ])
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, inputs):
        x = self.proj(inputs)  # bs, gen_len, hidden_size
        
        for block in self.blocks:
            x, _ = block(x, attention_bias=None, layer_past=None, use_cache=False)
        
        out = self.output_layer(x).squeeze(-1)  # bs, gen_len
        
        return out

