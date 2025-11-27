import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F

class GNNEncoder(nn.Module):
    """GNN Encoder (Supports GCN and GAT)"""
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.1, gnn_type='GCN'):
        super(GNNEncoder, self).__init__()
        
        self.convs = nn.ModuleList()
        if gnn_type == 'GCN':
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
        elif gnn_type == 'GAT':
            self.convs.append(GATConv(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(GATConv(hidden_channels, hidden_channels))
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        """
        前向傳播，返回節點級別的 embedding
        
        Args:
            x: 節點特徵 [num_nodes, in_channels]
            edge_index: 邊索引 [2, num_edges]
        
        Returns:
            node_embeddings: [num_nodes, hidden_channels]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.gelu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
