import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, PPI, Reddit
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_cluster import random_walk
from torch_geometric.loader import NeighborLoader
from model.GraphJEPA import GraphJEPA
from tqdm import tqdm
import os
import argparse
from config import Config
import numpy as np
from utils.pe import compute_pe
from utils.seed import set_seed
import random

def train_epoch(model, loader, optimizer, device, pe, current_tau):
    model.train()
    total_loss = 0
    num_samples = 0
    
    for batch in tqdm(loader, desc="Training"):
        # Perform random walks on CPU for determinism
        # batch is on CPU here
        
        # Seed nodes (u) are the first batch_size nodes
        u_idx = torch.arange(batch.batch_size)
        
        # Perform random walks on the SUBGRAPH to find targets (v)
        # We use the subgraph structure (batch.edge_index)
        row, col = batch.edge_index
        
        targets = []
        for _ in range(Config.num_targets):
            length = torch.randint(1, Config.walk_length + 1, (1,)).item()
            # random_walk returns [start, step1, ..., end]
            walk = random_walk(row, col, u_idx, walk_length=length, p=1, q=0.5)
            v = walk[:, -1]
            targets.append(v)

        v_idx = torch.stack(targets, dim=1) # [B, M]
        
        # Move data to device
        batch = batch.to(device)
        u_idx = u_idx.to(device)
        v_idx = v_idx.to(device)
        
        # Get PE for the nodes in the batch
        # pe is global [N, k], batch.n_id maps subgraph nodes to global nodes
        batch_pe = pe[batch.n_id]
        
        pos_u = batch_pe[u_idx] # [B, k]
        pos_v = batch_pe[v_idx] # [B, M, k]
        
        loss = model(batch.x, batch.edge_index, u_idx, v_idx, pos_u, pos_v)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.update_target_encoder(decay=current_tau)
        
        total_loss += loss.item() * batch.batch_size
        num_samples += batch.batch_size
    
    return total_loss / num_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PubMed', help='Dataset name')
    args = parser.parse_args()
    
    # Load Config
    Config.load(args.dataset)
    
    # Set Seed
    set_seed(Config.seed)
    
    # Load Dataset
    if Config.dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root=Config.dataset_root, name=Config.dataset_name)
    elif Config.dataset_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name=Config.dataset_name, root=Config.dataset_root)
        dataset.transform = T.ToUndirected()
    elif Config.dataset_name == 'PPI':
        dataset = PPI(root=Config.dataset_root)
    elif Config.dataset_name == 'Reddit':
        dataset = Reddit(root=Config.dataset_root)
    else:
        raise ValueError(f"Unknown dataset: {Config.dataset_name}")

    data = dataset[0]
    
    print(f"Dataset: {Config.dataset_name}")
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.num_features}")
    
    # Phase 1: Geometry Pre-processing (Load PE)
    pe_path = os.path.join(Config.pe_dir, f'{Config.pe_type.lower()}_pe_{Config.pos_dim}.pt')
    if os.path.exists(pe_path):
        print(f"Loading PE from {pe_path}")
        pe = torch.load(pe_path)
    else:
        print(f"PE not found at {pe_path}. Computing...")
        pe = compute_pe(data, Config.pe_type, Config.pos_dim)
        os.makedirs(Config.pe_dir, exist_ok=True)
        torch.save(pe, pe_path)
        print(f"PE saved to {pe_path}")
    
    # Phase 2: Data Pipeline
    g = torch.Generator()
    g.manual_seed(Config.seed)
    
    train_loader = NeighborLoader(
        data,
        num_neighbors=[-1]*6,
        batch_size=Config.batch_size,
        shuffle=True,
        generator=g
    )
    
    # Phase 3: Model
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    pe = pe.to(device)
    model = GraphJEPA(
        in_channels=dataset.num_features,
        hidden_channels=Config.hidden_channels,
        pos_dim=Config.pos_dim,
        num_layers=Config.num_layers,
        ema_decay=Config.ema_decay,
        dropout=Config.dropout,
        gnn_type=Config.gnn_type,
        pe_type=Config.pe_type
    ).to(device)
    
    optimizer = torch.optim.AdamW([
        {'params': model.context_encoder.parameters()},
        {'params': model.predictor.parameters()},
        {'params': model.pos_proj.parameters()}
    ], lr=Config.lr, weight_decay=Config.weight_decay)
    
    checkpoint_dir = os.path.join(Config.checkpoint_dir, Config.dataset_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Phase 4: Training Loop
    print("Starting training...")
    for epoch in range(Config.num_epochs):
        # Update tau
        tau = Config.tau_start + (Config.tau_end - Config.tau_start) * epoch / Config.num_epochs
        
        loss = train_epoch(model, train_loader, optimizer, device, pe, tau)
        
        print(f"Epoch {epoch+1}/{Config.num_epochs} | Loss: {loss:.4f} | Tau: {tau:.4f}")
        
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(checkpoint_dir, f'graph_jepa_epoch_{epoch+1}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': loss
            }, save_path)
            
    final_path = os.path.join(checkpoint_dir, 'graph_jepa_pretrained.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': Config.num_epochs,
        'loss': loss
    }, final_path)
    print(f"Training complete. Model saved to {final_path}")


