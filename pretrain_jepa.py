import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data, Batch
from model.GraphJEPA import GraphJEPA
from tqdm import tqdm
import os
from config import Config


def train_epoch(model, loader, optimizer, device):
    """
    訓練一個 epoch (使用 NeighborLoader 加速)
    
    Args:
        model: GraphJEPA 模型
        loader: NeighborLoader
        optimizer: 優化器
        device: 設備
    
    Returns:
        avg_loss: 平均損失
    """
    model.train()
    total_loss = 0
    num_samples = 0
    
    # 遍歷 Batch
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        batch_size = batch.batch_size
        
        # Center Mask: NeighborLoader 保證前 batch_size 個節點是種子節點 (中心節點)
        center_mask = torch.zeros(batch.num_nodes, dtype=torch.bool, device=device)
        center_mask[:batch_size] = True
        
        # Target View (完整)
        # batch 已經包含了完整的 2-hop 子圖信息
        batch_target = batch
        batch_target.center_mask = center_mask
        
        # Context View (Masked)
        batch_context = batch.clone()
        batch_context.x = batch.x.clone()
        batch_context.x[center_mask] = 0  # Mask 中心節點特徵
        batch_context.center_mask = center_mask
        
        # 前向傳播
        z_pred, z_target = model(batch_context, batch_target)
        
        # 計算損失
        loss = model.compute_loss(z_pred, z_target)
        
        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 動量更新 target_encoder
        model.update_target_encoder()
        
        total_loss += loss.item() * batch_size
        num_samples += batch_size
    
    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    return avg_loss


# 使用範例
if __name__ == "__main__":
    
    # 載入資料集
    dataset = Planetoid(root=Config.dataset_root, name=Config.dataset_name)
    data = dataset[0]
    
    print(f"Dataset: {dataset.name}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {data.num_features}")
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphJEPA(
        in_channels=dataset.num_features,
        hidden_channels=Config.hidden_channels,
        num_layers=Config.num_layers,
        ema_decay=Config.ema_decay,
        dropout=Config.dropout
    ).to(device)
    
    # 優化器 (只訓練 context_encoder 和 predictor)
    optimizer = torch.optim.AdamW([
        {'params': model.context_encoder.parameters()},
        {'params': model.predictor.parameters()}
    ], lr=Config.lr, weight_decay=Config.weight_decay)
    
    # 建立 NeighborLoader
    train_loader = NeighborLoader(
        data,
        num_neighbors=[-1] * 2,  # 2-hop, -1 表示取所有鄰居
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=4  # 可以根據 CPU 核心數調整
    )
    
    # 訓練
    num_epochs = Config.num_epochs
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
        
        # 每 10 個 epoch 保存一次
        if (epoch + 1) % 10 == 0:
            os.makedirs(Config.checkpoint_dir, exist_ok=True)
            model_save_path = os.path.join(Config.checkpoint_dir, f'graph_jepa_epoch_{epoch+1}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
            }, model_save_path)
            print(f"Model saved to {model_save_path}")
    
    # 保存最終模型
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
        'loss': avg_loss,
    }, Config.model_save_path)
    print(f"Final model saved to {Config.model_save_path}")
    print("Training complete.")