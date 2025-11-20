import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GNNEncoder import GNNEncoder, SubgraphEncoder
from model.Predictor import MLPPredictor

class GraphJEPA(nn.Module):
    """
    Graph Joint-Embedding Predictive Architecture
    
    架構：
    - Context: 1-hop 鄰域子圖 S_x = N_1(v)
    - Target: 2-hop ring 子圖 S_y = N_2(v) \ N_1(v)
    - Context Encoder: 編碼 S_x 為 z_x (節點級別)
    - Target Encoder: 編碼 S_y 為 z_y (子圖級別，使用 global mean pooling)
    - Predictor: 從 z_x 預測 z_y
    """
    def __init__(self, in_channels, hidden_channels, num_layers=2, 
                 ema_decay=0.996, dropout=0.1):
        super(GraphJEPA, self).__init__()
        
        # Context Encoder: 編碼 1-hop 鄰域，輸出中心節點的 embedding
        self.context_encoder = GNNEncoder(in_channels, hidden_channels, num_layers, dropout)
        
        # Target Encoder (EMA): 編碼 2-hop ring 子圖，輸出整個子圖的 embedding
        self.target_encoder = SubgraphEncoder(in_channels, hidden_channels, num_layers, dropout)
        
        # 初始化 target_encoder 為 context_encoder 的複製
        for param_c, param_t in zip(self.context_encoder.parameters(), 
                                     self.target_encoder.gnn.parameters()):
            param_t.data.copy_(param_c.data)
            param_t.requires_grad = False
        
        # Predictor: 從中心節點 embedding 預測 2-hop ring 的子圖 embedding
        self.predictor = MLPPredictor(hidden_channels, hidden_channels*2, hidden_channels, dropout=dropout)
        
        self.ema_decay = ema_decay
    
    @torch.no_grad()
    def update_target_encoder(self):
        """使用 EMA 更新 target encoder"""
        for param_c, param_t in zip(self.context_encoder.parameters(), 
                                     self.target_encoder.gnn.parameters()):
            param_t.data = self.ema_decay * param_t.data + (1 - self.ema_decay) * param_c.data
    
    def forward(self, batch_context, batch_target):
        """
        前向傳播 (Batch Processing)
        
        Args:
            batch_context: context 子圖的 Batch 對象 (包含 center_mask)
            batch_target: target 子圖的 Batch 對象
            
        Returns:
            z_pred: 預測的 target embedding [batch_size, hidden_channels]
            z_target: 實際的 target embedding [batch_size, hidden_channels]
        """
        # Context encoding: 編碼 1-hop 鄰域
        z_context_all = self.context_encoder(batch_context.x, batch_context.edge_index)  # [total_nodes_context, hidden]
        
        # 取出每個子圖的中心節點 embedding
        # batch_context.center_mask 是一個 bool mask，標記了每個子圖的中心節點
        z_context = z_context_all[batch_context.center_mask]  # [batch_size, hidden]
        
        # Prediction: 從中心節點預測 2-hop ring 的子圖 embedding
        z_pred = self.predictor(z_context)  # [batch_size, hidden]
        
        # Target encoding: 編碼 2-hop ring 子圖 (no grad)
        with torch.no_grad():
            # 使用 batch 參數進行 global mean pooling
            z_target = self.target_encoder(batch_target.x, batch_target.edge_index, batch_target.batch)  # [batch_size, hidden]
            z_target = z_target.detach()  # 確保從計算圖中分離
        
        return z_pred, z_target
    
    def compute_loss(self, z_pred, z_target):
        """
        計算損失：Cosine Similarity Loss
        
        Args:
            z_pred: 預測的 embedding [batch_size, hidden_channels]
            z_target: 目標 embedding [batch_size, hidden_channels]
        
        Returns:
            loss: scalar (mean over batch)
        """
        z_pred_norm = F.normalize(z_pred, p=2, dim=-1)
        z_target_norm = F.normalize(z_target, p=2, dim=-1)
        
        # 計算每個樣本的 cosine similarity
        cosine_sim = (z_pred_norm * z_target_norm).sum(dim=-1)  # [batch_size]
        
        # Loss = 1 - cosine_sim
        loss = (1 - cosine_sim).mean()
        return loss
