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
    parser.add_argument('--seeds', type=int, nargs='+', help='List of random seeds')
    args = parser.parse_args()

    # Load Config
    Config.load(args.dataset)
    
    if args.seeds:
        Config.seeds = args.seeds
    
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

    checkpoint_dir = os.path.join(Config.checkpoint_dir, Config.dataset_name)

    # ============== 下游任務：節點分類 ==============
    print("\n" + "="*50)
    print("Downstream Task: Node Classification")
    print("="*50 + "\n")

    is_multilabel = False
    if data.y.ndim > 1:
        if data.y.shape[1] == 1:
            pass
        else:
            is_multilabel = True

    test_accuracies = []
    val_accuracies = []

    for seed in Config.seeds:
        print(f"\nRunning with seed: {seed}")
        set_seed(seed)

        # Load model for this seed
        model_filename = f'graph_jepa_pretrained_seed_{seed}.pth'
        model_save_path = os.path.join(checkpoint_dir, model_filename)
        
        if not os.path.exists(model_save_path):
            print(f"Warning: Model file {model_save_path} not found. Skipping seed {seed}.")
            continue
            
        checkpoint = torch.load(model_save_path)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            loaded_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            loaded_model.load_state_dict(checkpoint)
        loaded_model.eval()
        print(f"Model loaded from {model_save_path}")

        # 使用 target_encoder 提取所有節點的嵌入
        # For large datasets like Reddit, use NeighborLoader to avoid OOM
        if Config.dataset_name in ['Reddit']:
            # print(f"Using NeighborLoader for large dataset: {Config.dataset_name}")
            inference_loader = NeighborLoader(
                data,
                num_neighbors=[25, 15, 10],
                batch_size=4096,
                shuffle=False,
            )
            
            all_embeddings = []
            with torch.no_grad():
                for batch in tqdm(inference_loader, desc="Inference", leave=False):
                    batch = batch.to(device)
                    out = loaded_model.target_encoder(batch.x, batch.edge_index)
                    # Only take the embeddings of the seed nodes (first batch_size)
                    out = out[:batch.batch_size]
                    all_embeddings.append(out.cpu())
            
            node_embeddings = torch.cat(all_embeddings, dim=0)
        else:
            # Full batch for small datasets
            with torch.no_grad():
                data_gpu = data.to(device)
                node_embeddings = loaded_model.target_encoder(data_gpu.x, data_gpu.edge_index)

        # 準備訓練/驗證/測試數據
        X = node_embeddings.cpu().numpy()
        y = data.y.cpu().numpy()

        # Flatten y if it's a column vector and not multilabel
        if not is_multilabel and y.ndim > 1:
            y = y.ravel()

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
            classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=seed))
        else:
            classifier = LogisticRegression(max_iter=1000, random_state=seed)
        
        classifier.fit(X_train, y_train)

        # 評估
        train_pred = classifier.predict(X_train)
        val_pred = classifier.predict(X_val)
        test_pred = classifier.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        test_acc = accuracy_score(y_test, test_pred)

        print(f"Seed {seed} Results:")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        test_accuracies.append(test_acc)
        val_accuracies.append(val_acc)

    print("\n" + "="*50)
    print(f"Final Results over {len(Config.seeds)} seeds:")
    print(f"Test Accuracy: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")
    print("="*50 + "\n")
