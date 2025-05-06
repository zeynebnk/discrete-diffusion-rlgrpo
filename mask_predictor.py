import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskPredictor(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, n_layers=2, n_heads=4):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Project vocabulary probabilities to hidden dimension
        self.proj = nn.Linear(vocab_size, hidden_size)
        
        # Transformer encoder to process sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Final prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, logits, gen_len):
        """
        Args:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
            gen_len: Integer indicating how many tokens to generate
        Returns:
            mask_probs: Tensor of shape (batch_size, gen_len) with probabilities of masking each token
        """
        batch_size, seq_len, _ = logits.shape
        
        # Get probability distribution over vocabulary
        probs = F.softmax(logits, dim=-1)
        
        # Only consider the last gen_len tokens
        probs = probs[:, -gen_len:, :]
        
        # Project to hidden dimension
        hidden = self.proj(probs)  # [batch_size, gen_len, hidden_size]
        
        # Process sequence with transformer
        hidden = self.transformer(hidden)  # [batch_size, gen_len, hidden_size]
        
        # Predict masking probabilities
        mask_probs = self.predictor(hidden).squeeze(-1)  # [batch_size, gen_len]
        
        return mask_probs
    
    def predict_masks(self, logits, gen_len, threshold=0.5):
        """
        Predict binary mask for each token based on logits
        Args:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
            gen_len: Integer indicating how many tokens to generate
            threshold: Probability threshold for masking (default: 0.5)
        Returns:
            masks: Boolean tensor of shape (batch_size, gen_len) indicating which tokens should be masked
        """
        mask_probs = self.forward(logits, gen_len)
        return mask_probs > threshold 