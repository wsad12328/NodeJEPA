import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import k_hop_subgraph
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm

class SimpleGNN(nn.Module):
    """Simple GNN encoder for node embeddings"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class ContextEncoder(nn.Module):
    """Encode 1-hop context graph"""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gnn = SimpleGNN(in_channels, hidden_channels, out_channels, num_layers=3)
        
    def forward(self, x, edge_index):
        return self.gnn(x, edge_index)

class TargetEncoder(nn.Module):
    """Encode 2-hop target graph (EMA updated)"""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gnn = SimpleGNN(in_channels, hidden_channels, out_channels, num_layers=3)
        
    def forward(self, x, edge_index):
        return self.gnn(x, edge_index)

class Predictor(nn.Module):
    """Predict target embeddings from context embeddings"""
    def __init__(self, embed_dim, hidden_dim, num_predict=5):
        super().__init__()
        self.num_predict = num_predict
        # 預測多個節點的 embeddings
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embed_dim * num_predict)
        )
        
    def forward(self, x):
        out = self.mlp(x)  # [batch, embed_dim * num_predict]
        # Reshape to [batch, num_predict, embed_dim]
        return out.view(-1, self.num_predict, x.size(-1))

class GraphJEPA(nn.Module):
    """
    Graph JEPA for node-level representation learning
    核心思想：用 1-hop 的信息預測 2-hop 但不在 1-hop 的節點
    """
    def __init__(self, in_channels, hidden_channels, embed_dim, momentum=0.99, num_predict=3):
        super().__init__()
        self.momentum = momentum
        self.embed_dim = embed_dim
        self.num_predict = num_predict
        
        # Context encoder (trainable) - 看 1-hop subgraph
        self.context_encoder = ContextEncoder(in_channels, hidden_channels, embed_dim)
        
        # Target encoder (EMA) - 看 2-hop subgraph
        self.target_encoder = TargetEncoder(in_channels, hidden_channels, embed_dim)
        
        # Predictor - 從 1-hop context 預測 2-hop target nodes
        self.predictor = Predictor(embed_dim, hidden_dim=hidden_channels*2, num_predict=num_predict)
        
        # Initialize target encoder
        self._initialize_target_encoder()
        
    def _initialize_target_encoder(self):
        """Initialize target encoder with context encoder weights"""
        for param_c, param_t in zip(self.context_encoder.parameters(), 
                                     self.target_encoder.parameters()):
            param_t.data.copy_(param_c.data)
            param_t.requires_grad = False
            
    @torch.no_grad()
    def _update_target_encoder(self):
        """EMA update of target encoder"""
        for param_c, param_t in zip(self.context_encoder.parameters(), 
                                     self.target_encoder.parameters()):
            param_t.data = self.momentum * param_t.data + (1 - self.momentum) * param_c.data
    
    def sample_target_nodes(self, center_node, edge_index, num_nodes, num_samples=3):
        """
        Sample target nodes from 2-hop neighborhood but not in 1-hop
        """
        # Get 1-hop subgraph (context)
        subset_1hop, edge_index_1hop, _, _ = k_hop_subgraph(
            center_node, num_hops=1, edge_index=edge_index, 
            num_nodes=num_nodes, relabel_nodes=False
        )
        
        # Get 2-hop subgraph (target)
        subset_2hop, edge_index_2hop, _, _ = k_hop_subgraph(
            center_node, num_hops=2, edge_index=edge_index, 
            num_nodes=num_nodes, relabel_nodes=False
        )
        
        # Find nodes in 2-hop but not in 1-hop (這些是我們要預測的 target)
        set_1hop = set(subset_1hop.tolist())
        set_2hop = set(subset_2hop.tolist())
        candidate_nodes = list(set_2hop - set_1hop)
        
        if len(candidate_nodes) == 0:
            return None, None, None, None
        
        # Random sample target nodes
        num_samples = min(num_samples, len(candidate_nodes))
        target_indices = np.random.choice(len(candidate_nodes), num_samples, replace=False)
        target_nodes = torch.tensor([candidate_nodes[i] for i in target_indices], 
                                    dtype=torch.long, device=edge_index.device)
        
        return subset_1hop, edge_index_1hop, edge_index_2hop, target_nodes
    
    def forward_batch(self, x, edge_index, node_indices):
        """
        關鍵改進：
        1. Context encoder 看完整的 1-hop subgraph 並聚合信息
        2. Target encoder 看完整的 2-hop subgraph
        3. 從 1-hop 的聚合信息預測 2-hop target nodes 的 embeddings
        """
        batch_results = []
        num_nodes = x.size(0)
        
        for center_node in node_indices:
            # Sample target nodes and get subgraphs
            result = self.sample_target_nodes(
                center_node, edge_index, num_nodes, self.num_predict
            )
            
            if result[0] is None:
                continue
            
            subset_1hop, edge_index_1hop, edge_index_2hop, target_nodes = result
            
            # === Context: 用 1-hop subgraph 獲取局部信息 ===
            context_embed_all = self.context_encoder(x, edge_index_1hop)
            
            # 聚合 1-hop 所有節點的信息（包括中心節點和鄰居）
            context_embeds = context_embed_all[subset_1hop]  # [num_1hop_nodes, embed_dim]
            
            # 使用 attention-based aggregation 或 mean pooling
            # 這裡用 mean pooling，你也可以改成更複雜的聚合方式
            context_aggregated = context_embeds.mean(dim=0, keepdim=True)  # [1, embed_dim]
            
            # === Target: 用 2-hop subgraph 獲取 target nodes 的真實 embeddings ===
            with torch.no_grad():
                target_embed_all = self.target_encoder(x, edge_index_2hop)
                target_embeds = target_embed_all[target_nodes]  # [num_predict, embed_dim]
            
            batch_results.append({
                'context': context_aggregated,
                'target': target_embeds,
                'num_targets': len(target_nodes),
                'center_node': center_node
            })
        
        if len(batch_results) == 0:
            return None
        
        # Stack for batch processing
        contexts = torch.cat([r['context'] for r in batch_results], dim=0)  # [batch, embed_dim]
        
        # Predict target embeddings from context
        predicted_embeds = self.predictor(contexts)  # [batch, num_predict, embed_dim]
        
        # Split back
        for i, result in enumerate(batch_results):
            result['predicted'] = predicted_embeds[i]  # [num_predict, embed_dim]
        
        return batch_results
    
    def compute_loss_batch(self, batch_results):
        """
        改進的 loss 計算：
        1. 使用 cosine similarity + smooth L1
        2. 加入 variance regularization 防止模式崩潰
        """
        total_loss = 0
        total_var_loss = 0
        count = 0
        
        for result in batch_results:
            pred = result['predicted']  # [num_predict, embed_dim]
            tgt = result['target']  # [num_targets, embed_dim]
            
            # 確保維度匹配
            min_size = min(pred.size(0), tgt.size(0))
            pred = pred[:min_size]
            tgt = tgt[:min_size]
            
            # Smooth L1 Loss
            l1_loss = F.smooth_l1_loss(pred, tgt, beta=1.0)
            
            # Variance regularization to prevent collapse
            pred_var = pred.var(dim=0).mean()
            var_loss = torch.clamp(1.0 - pred_var, min=0.0)

            total_loss += l1_loss
            total_var_loss += var_loss
            count += 1
        
        avg_loss = total_loss / max(count, 1)
        avg_var_loss = total_var_loss / max(count, 1)
        
        return avg_loss + 0.1 * avg_var_loss
    
    def get_node_embeddings(self, x, edge_index):
        """
        Get node embeddings for downstream tasks
        使用 context encoder（在完整圖上）來獲取最終的 node embeddings
        """
        with torch.no_grad():
            embeddings = self.context_encoder(x, edge_index)
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

def train_jepa(model, data, optimizer, scheduler, num_epochs=200, batch_size=32, device='cpu'):
    """Train Graph JEPA model with batching"""
    model.train()
    num_nodes = data.num_nodes
    
    epoch_pbar = tqdm(range(num_epochs), desc='Training', unit='epoch')
    best_loss = float('inf')
    
    for epoch in epoch_pbar:
        total_loss = 0
        num_batches = 0
        
        perm = torch.randperm(num_nodes)
        num_total_batches = (num_nodes + batch_size - 1) // batch_size
        batch_pbar = tqdm(range(0, num_nodes, batch_size), 
                          desc=f'Epoch {epoch+1}', 
                          leave=False,
                          total=num_total_batches)
        
        for start_idx in batch_pbar:
            end_idx = min(start_idx + batch_size, num_nodes)
            batch_nodes = perm[start_idx:end_idx].tolist()
            
            optimizer.zero_grad()
            
            batch_results = model.forward_batch(data.x, data.edge_index, batch_nodes)
            
            if batch_results is None:
                continue
            
            loss = model.compute_loss_batch(batch_results)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            model._update_target_encoder()
            
            total_loss += loss.item()
            num_batches += 1
            
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        avg_loss = total_loss / max(num_batches, 1)
        epoch_pbar.set_postfix({
            'avg_loss': f'{avg_loss:.4f}', 
            'batches': num_batches,
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    return model

def evaluate_embeddings(model, data):
    """Evaluate learned embeddings using linear probe"""
    model.eval()
    
    print("Extracting embeddings...")
    embeddings = model.get_node_embeddings(data.x, data.edge_index)
    
    X = embeddings.cpu().numpy()
    y = data.y.cpu().numpy()
    
    train_mask = data.train_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()
    
    print("Training linear classifier...")
    clf = LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs')
    clf.fit(X[train_mask], y[train_mask])
    
    # Train accuracy
    y_train_pred = clf.predict(X[train_mask])
    train_acc = accuracy_score(y[train_mask], y_train_pred)
    
    # Test accuracy
    y_pred = clf.predict(X[test_mask])
    accuracy = accuracy_score(y[test_mask], y_pred)
    f1 = f1_score(y[test_mask], y_pred, average='macro')
    
    return train_acc, accuracy, f1

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    for dataset_name in ['Cora', 'CiteSeer']:
        print(f'\n{"="*60}')
        print(f'Dataset: {dataset_name}')
        print(f'{"="*60}')
        
        dataset = Planetoid(root='./data/Planetoid', name=dataset_name)
        data = dataset[0].to(device)
        
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        
        model = GraphJEPA(
            in_channels=dataset.num_features,
            hidden_channels=256,
            embed_dim=128,
            momentum=0.99,
            num_predict=3  # 預測 3 個 target nodes
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
        
        print('\nTraining Graph JEPA...')
        model = train_jepa(
            model, data, optimizer, scheduler,
            num_epochs=200, batch_size=64, device=device
        )
        
        print('\nEvaluating with linear probe...')
        train_acc, test_acc, f1 = evaluate_embeddings(model, data)
        print(f'\n{"="*60}')
        print(f'Results for {dataset_name}:')
        print(f'Train Accuracy: {train_acc:.4f}')
        print(f'Test Accuracy: {test_acc:.4f}')
        print(f'Test F1-Score (Macro): {f1:.4f}')
        print(f'{"="*60}')

if __name__ == '__main__':
    main()