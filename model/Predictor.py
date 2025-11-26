import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPPredictor(nn.Module):
    """
    2-layer MLP Predictor
    From context embedding to target embedding
    Structure: Linear -> BN -> GELU -> Dropout -> Linear -> (Residual Add)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        super(MLPPredictor, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.layer2 = nn.Linear(hidden_channels, out_channels)
        
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        """
        Args:
            x: input embeddings [batch_size, in_channels]
        Returns:
            output: predicted embeddings [batch_size, out_channels]
        """
        
        x = self.layer1(x)
        x = self.layer2(x)
            
        return x