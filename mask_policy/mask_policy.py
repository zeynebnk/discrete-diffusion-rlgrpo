import torch
import torch.nn as nn
import torch.nn.functional as F

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

