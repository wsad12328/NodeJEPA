import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid, PPI, Reddit
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from model.GraphJEPA import GraphJEPA
import os  
from config import Config
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import argparse
from utils.seed import set_seed
from utils.visualization import visualize_embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PubMed', help='Dataset name')
    args = parser.parse_args()

    # Load Config
    Config.load(args.dataset)
    
    # Set Seed
    set_seed(Config.seed)

    # ============== 載入模型並提取節點嵌入 ==============  
    print("\n" + "="*50)
    print("Loading model and extracting node embeddings...")
    print("="*50 + "\n")

# 設定設備
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 載入資料集
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

    # 初始化新模型（用於載入）
    loaded_model = GraphJEPA(
        in_channels=dataset.num_features,
        hidden_channels=Config.hidden_channels,
        num_layers=Config.num_layers,
        ema_decay=Config.ema_decay,
        dropout=Config.dropout,
        gnn_type=Config.gnn_type,
        pos_dim=Config.pos_dim
    ).to(device)

    # 載入權重
    checkpoint_dir = os.path.join(Config.checkpoint_dir, Config.dataset_name)
    model_save_path = os.path.join(checkpoint_dir, Config.model_save_path)
    checkpoint = torch.load(model_save_path)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        loaded_model.load_state_dict(checkpoint)
    loaded_model.eval()
    print(f"Model loaded from {model_save_path}")

    # 使用 target_encoder 提取所有節點的嵌入
    print("Extracting embeddings...")
    loaded_model.eval()
    
    # For large datasets like Reddit, use NeighborLoader to avoid OOM
    if Config.dataset_name in ['Reddit','ogbn-arxiv']:
        print(f"Using NeighborLoader for large dataset: {Config.dataset_name}")
        inference_loader = NeighborLoader(
            data,
            num_neighbors=[25, 15, 10],
            batch_size=4096,
            shuffle=False,
        )
        
        all_embeddings = []
        with torch.no_grad():
            for batch in tqdm(inference_loader, desc="Inference"):
                batch = batch.to(device)
                out = loaded_model.target_encoder(batch.x, batch.edge_index)
                # Only take the embeddings of the seed nodes (first batch_size)
                out = out[:batch.batch_size]
                all_embeddings.append(out.cpu())
        
        node_embeddings = torch.cat(all_embeddings, dim=0)
    else:
        # Full batch for small datasets
        with torch.no_grad():
            data = data.to(device)
            node_embeddings = loaded_model.target_encoder(data.x, data.edge_index)

    print(f"Node embeddings shape: {node_embeddings.shape}")

    # ============== t-SNE Visualization ==============
    # visualize_embeddings(node_embeddings, data.y, Config.dataset_name, Config.gnn_type, seed=Config.seed)

    # ============== 下游任務：節點分類 ==============
    print("\n" + "="*50)
    print("Downstream Task: Node Classification")
    print("="*50 + "\n")

    # 準備訓練/驗證/測試數據
    X = node_embeddings.cpu().numpy()
    y = data.y.cpu().numpy()

    is_multilabel = False
    if y.ndim > 1:
        if y.shape[1] == 1:
            y = y.flatten()
        else:
            is_multilabel = True

    if hasattr(data, 'train_mask'):
        train_mask = data.train_mask.cpu().numpy()
        val_mask = data.val_mask.cpu().numpy()
        test_mask = data.test_mask.cpu().numpy()
    elif Config.dataset_name == 'ogbn-arxiv':
        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train']
        val_idx = split_idx['valid']
        test_idx = split_idx['test']
        
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask[val_idx] = True
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask[test_idx] = True
        
        train_mask = train_mask.numpy()
        val_mask = val_mask.numpy()
        test_mask = test_mask.numpy()
    else:
        print("Warning: No masks found. Using random split (60/20/20).")
        indices = np.arange(data.num_nodes)
        np.random.shuffle(indices)
        train_split = int(0.6 * data.num_nodes)
        val_split = int(0.8 * data.num_nodes)
        
        train_mask = np.zeros(data.num_nodes, dtype=bool)
        train_mask[indices[:train_split]] = True
        val_mask = np.zeros(data.num_nodes, dtype=bool)
        val_mask[indices[train_split:val_split]] = True
        test_mask = np.zeros(data.num_nodes, dtype=bool)
        test_mask[indices[val_split:]] = True

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # 訓練 Logistic Regression 分類器
    if is_multilabel:
        from sklearn.multiclass import OneVsRestClassifier
        classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=Config.seed))
    else:
        classifier = LogisticRegression(max_iter=1000, random_state=Config.seed)
    
    classifier.fit(X_train, y_train)

    # 評估
    train_pred = classifier.predict(X_train)
    val_pred = classifier.predict(X_val)
    test_pred = classifier.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("Node Classification Results:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
