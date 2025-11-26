import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GNNEncoder import GNNEncoder
from model.Predictor import MLPPredictor

class PositionProjectionNetwork(nn.Module):
    """
    Position Projection Network
    Input: Relative position delta_p (k-dim)
    Output: Position embedding e_pos (d-dim)
    Structure: Linear -> LayerNorm -> GELU -> Linear
    """
    def __init__(self, in_channels, out_channels):
        super(PositionProjectionNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, x):
        return self.net(x)

class GraphJEPA(nn.Module):
    """
    Graph Joint-Embedding Predictive Architecture with Geometry Awareness
    """
    def __init__(self, in_channels, hidden_channels, pos_dim, num_layers=2, 
                 ema_decay=0.996, dropout=0.1, gnn_type='GCN', pe_type='RWSE'):
        super(GraphJEPA, self).__init__()
        
        self.pe_type = pe_type
        
        # Context Encoder
        self.context_encoder = GNNEncoder(in_channels, hidden_channels, num_layers, dropout, gnn_type=gnn_type)
        
        # Target Encoder (EMA)
        self.target_encoder = GNNEncoder(in_channels, hidden_channels, num_layers, dropout, gnn_type=gnn_type)
        
        # Initialize target encoder
        for param_c, param_t in zip(self.context_encoder.parameters(), 
                                     self.target_encoder.parameters()):
            param_t.data.copy_(param_c.data)
            param_t.requires_grad = False
        
        # Position Projection Network
        self.pos_proj = PositionProjectionNetwork(pos_dim, hidden_channels)
        
        # Predictor
        self.predictor = MLPPredictor(hidden_channels, hidden_channels*2, hidden_channels, dropout=dropout)
        
        self.ema_decay = ema_decay
    
    @torch.no_grad()
    def update_target_encoder(self, decay=None):
        """Update target encoder with EMA"""
        tau = decay if decay is not None else self.ema_decay
        for param_c, param_t in zip(self.context_encoder.parameters(), 
                                     self.target_encoder.parameters()):
            param_t.data = tau * param_t.data + (1 - tau) * param_c.data
    
    def encode_context(self, x, edge_index):
        return self.context_encoder(x, edge_index)
        
    @torch.no_grad()
    def encode_target(self, x, edge_index):
        return self.target_encoder(x, edge_index)

    def predict(self, h_u, delta_p):
        """
        Args:
            h_u: Context embeddings [B, M, d] (already expanded)
            delta_p: Relative positions [B, M, k]
        Returns:
            h_v_pred: Predicted target embeddings [B, M, d]
        """
        # Project position
        e_pos = self.pos_proj(delta_p) # [B, M, d]
        
        # Combine
        z = h_u + e_pos
        
        # Predict
        # Flatten for MLP with BatchNorm: [B*M, d]
        B, M, D = z.shape
        z_flat = z.view(B * M, D)
        h_v_pred_flat = self.predictor(z_flat)
        h_v_pred = h_v_pred_flat.view(B, M, D)
        
        return h_v_pred
    
    def compute_loss(self, h_v_pred, h_v_target):
        """
        Prediction Loss: 1 - cosine similarity between predicted and target embeddings
        """
        cos_sim = F.cosine_similarity(h_v_pred, h_v_target, dim=-1)
        loss = 1 - cos_sim.mean()
        return loss

    def forward(self, x, edge_index, u_idx, v_idx, pos_u, pos_v):
        """
        Forward pass for training
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            u_idx: Source node indices [B]
            v_idx: Target node indices [B, M]
            pos_u: Source node PE [B, K]
            pos_v: Target node PE [B, M, K]
        """
        batch_size = u_idx.size(0)
        num_targets = v_idx.size(1)
        
        # Step 1: Sign Flipping
        if self.pe_type == 'Laplacian':
            # Generate random sign vector s in {-1, 1}^k
            s = torch.randint(0, 2, (1, 1, pos_u.size(-1)), device=x.device).float() * 2 - 1
            pos_u = pos_u.unsqueeze(1) * s # [B, 1, k]
            pos_v = pos_v * s # [B, M, k]
        else:
            # RWSE values are probabilities [0, 1], sign flipping is not appropriate.
            # Just unsqueeze pos_u to match dimensions
            pos_u = pos_u.unsqueeze(1) # [B, 1, k]
        
        # Step 2: Get Embeddings
        h_context_all = self.encode_context(x, edge_index)
        
        with torch.no_grad():
            h_target_all = self.encode_target(x, edge_index)
        
        h_u = h_context_all[u_idx] # [B, d]
        
        # Flatten v_idx to index
        v_idx_flat = v_idx.view(-1)
        h_v = h_target_all[v_idx_flat] # [B*M, d]
        h_v = h_v.view(batch_size, num_targets, -1) # [B, M, d]
        
        # Step 3: Construct Queries
        # Relative position
        delta_p = pos_v - pos_u # [B, M, k]
        
        # Step 4: Predict
        # Expand h_u to [B, M, d]
        h_u_expanded = h_u.unsqueeze(1).expand(-1, num_targets, -1)
        
        h_v_pred = self.predict(h_u_expanded, delta_p) # [B, M, d]
        
        # Step 5: Loss
        loss = self.compute_loss(h_v_pred, h_v.detach())
        
        return loss
