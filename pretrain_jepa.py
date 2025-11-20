import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import k_hop_subgraph, subgraph
from torch_geometric.data import Data, Batch
from model.GraphJEPA import GraphJEPA
from tqdm import tqdm
import os
from config import Config


def precompute_subgraphs(data):
    """
    預計算所有節點的 1-hop 和 2-hop ring 子圖
    
    Args:
        data: PyG Data 對象
    
    Returns:
        subgraph_data: dict，包含每個節點的子圖信息
            {
                node_id: {
                    'context_nodes': tensor,      # 1-hop 節點索引
                    'target_nodes': tensor,       # 2-hop ring 節點索引
                    'context_edge_index': tensor, # 1-hop 內部邊
                    'target_edge_index': tensor,  # 2-hop ring 內部邊
                }
            }
    """
    print("預計算所有節點的子圖...")
    subgraph_data = {}
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    
    for node_id in tqdm(range(num_nodes), desc="Computing subgraphs"):
        # 提取 1-hop 鄰居（包含中心節點）
        subset_1hop, edge_1hop, _, _ = k_hop_subgraph(
            node_idx=node_id,
            num_hops=1,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes
        )
        
        # 提取 2-hop 鄰居（包含中心節點和 1-hop）
        subset_2hop, _, _, _ = k_hop_subgraph(
            node_idx=node_id,
            num_hops=2,
            edge_index=edge_index,
            relabel_nodes=False,
            num_nodes=num_nodes
        )
        
        # 計算 2-hop ring: N_2(v) \ N_1(v)
        # 創建一個 mask 來排除中心節點和 1-hop 鄰居
        mask = torch.ones(subset_2hop.size(0), dtype=torch.bool)
        for node in subset_1hop:
            mask[subset_2hop == node.item()] = False
        
        ring_nodes = subset_2hop[mask]
        
        # 提取 2-hop ring 內部的邊
        if ring_nodes.size(0) > 0:
            ring_edge_index, _ = subgraph(
                ring_nodes,
                edge_index,
                relabel_nodes=True,
                num_nodes=num_nodes
            )
        else:
            ring_edge_index = torch.empty((2, 0), dtype=edge_index.dtype)
        
        # 找出中心節點在 1-hop 子圖中的索引
        center_node_idx = (subset_1hop == node_id).nonzero(as_tuple=True)[0].item()

        subgraph_data[node_id] = {
            'context_nodes': subset_1hop,
            'target_nodes': ring_nodes,
            'context_edge_index': edge_1hop,
            'target_edge_index': ring_edge_index,
            'center_node_idx': center_node_idx
        }
    
    print(f"完成！共計算 {len(subgraph_data)} 個節點的子圖")
    return subgraph_data


def train_epoch(model, data, subgraph_data, optimizer, device, batch_size=128):
    """
    訓練一個 epoch（使用預計算的子圖）
    
    Args:
        model: GraphJEPA 模型
        data: 完整圖數據
        subgraph_data: 預計算的子圖信息
        optimizer: 優化器
        device: 設備
        batch_size: 批次大小
    
    Returns:
        avg_loss: 平均損失
    """
    model.train()
    total_loss = 0
    num_samples = 0
    
    # 隨機選擇節點
    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes)
    
    for i in range(0, num_nodes, batch_size):
        batch_nodes = perm[i:i+batch_size]
        
        context_data_list = []
        target_data_list = []
        
        for center_node in batch_nodes:
            center_node = center_node.item()
            
            # 從預計算的子圖中獲取數據
            subgraph_info = subgraph_data[center_node]
            context_nodes = subgraph_info['context_nodes']
            target_nodes = subgraph_info['target_nodes']
            context_edge_index = subgraph_info['context_edge_index']
            target_edge_index = subgraph_info['target_edge_index']
            center_node_idx = subgraph_info['center_node_idx']
            
            # 如果 2-hop ring 為空，跳過此節點
            if target_nodes.size(0) == 0:
                continue
            
            # 構建 Context Data
            # 創建 center_mask
            center_mask = torch.zeros(context_nodes.size(0), dtype=torch.bool)
            center_mask[center_node_idx] = True
            
            data_context = Data(
                x=data.x[context_nodes], 
                edge_index=context_edge_index,
                center_mask=center_mask
            )
            context_data_list.append(data_context)
            
            # 構建 Target Data
            data_target = Data(
                x=data.x[target_nodes],
                edge_index=target_edge_index
            )
            target_data_list.append(data_target)
        
        if len(context_data_list) > 0:
            # 建立 Batch
            batch_context = Batch.from_data_list(context_data_list).to(device)
            batch_target = Batch.from_data_list(target_data_list).to(device)
            
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
            
            total_loss += loss.item() * len(context_data_list)
            num_samples += len(context_data_list)
    
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
    
    # 預計算所有節點的子圖
    subgraph_data = precompute_subgraphs(data)
    
    # 統計信息
    valid_nodes = sum(1 for node_id in range(data.num_nodes) 
                     if subgraph_data[node_id]['target_nodes'].size(0) > 0)
    print(f"\n有效節點數（有 2-hop ring）: {valid_nodes}/{data.num_nodes}")
    
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
    
    # 訓練
    num_epochs = Config.num_epochs
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, data, subgraph_data, optimizer, device, batch_size=Config.batch_size)
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

    # ============== 載入模型並提取節點嵌入 ==============
    print("\n" + "="*50)
    print("Loading model and extracting node embeddings...")
    print("="*50 + "\n")
    
    # 初始化新模型（用於載入）
    loaded_model = GraphJEPA(
        in_channels=dataset.num_features,
        hidden_channels=Config.hidden_channels,
        num_layers=Config.num_layers,
        ema_decay=Config.ema_decay,
        dropout=Config.dropout
    ).to(device)
    
    # 載入權重
    checkpoint = torch.load(Config.model_save_path)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    print(f"Model loaded from {Config.model_save_path}")