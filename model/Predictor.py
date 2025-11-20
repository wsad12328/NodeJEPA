import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPPredictor(nn.Module):
    """
    2層 MLP 預測器
    從中心節點的 embedding 預測 2-hop ring 的子圖 embedding
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        super(MLPPredictor, self).__init__()
        
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """
        Args:
            x: context embeddings, shape [batch_size, in_channels]
        
        Returns:
            predictions: shape [batch_size, out_channels]
        """
        # with residual connection
        x_residual = x
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.leaky_relu(x)
        x += x_residual  # 殘差連接
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        
        return x