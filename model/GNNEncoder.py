import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class GNNEncoder(nn.Module):
    """GNN Encoder (使用 GCN)"""
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.1):
        super(GNNEncoder, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
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
            if i < len(self.convs) - 1:
                x = F.leaky_relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SubgraphEncoder(nn.Module):
    """
    子圖編碼器：使用 GNN + Global Pooling
    將子圖編碼為單一向量表徵
    """
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.1):
        super(SubgraphEncoder, self).__init__()
        
        # GNN 層
        self.gnn = GNNEncoder(in_channels, hidden_channels, num_layers, dropout)
    
    def forward(self, x, edge_index, batch=None):
        """
        將子圖編碼為單一向量
        
        Args:
            x: 節點特徵 [num_nodes, in_channels]
            edge_index: 邊索引 [2, num_edges]
            batch: 批次索引 [num_nodes]，用於區分不同子圖（可選）
        
        Returns:
            subgraph_embedding: [batch_size, hidden_channels] 或 [1, hidden_channels]
        """
        # GNN 編碼
        node_embeddings = self.gnn(x, edge_index)  # [num_nodes, hidden_channels]
        
        # Global Mean Pooling
        if batch is None:
            # 單一子圖：對所有節點取平均
            subgraph_embedding = node_embeddings.mean(dim=0, keepdim=True)  # [1, hidden_channels]
        else:
            # 多個子圖：使用 batch 索引進行分組池化
            subgraph_embedding = global_mean_pool(node_embeddings, batch)  # [batch_size, hidden_channels]
        
        return subgraph_embedding
