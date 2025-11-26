import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from model.GraphJEPA import GraphJEPA
import os  
from config import Config
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# ============== 載入模型並提取節點嵌入 ==============  
print("\n" + "="*50)
print("Loading model and extracting node embeddings...")
print("="*50 + "\n")

# 設定設備
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# 載入資料集
dataset = Planetoid(root=Config.dataset_root, name=Config.dataset_name)
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

# 使用 target_encoder 提取所有節點的嵌入（在完整圖上）
with torch.no_grad():
    data = data.to(device)
    node_embeddings = loaded_model.target_encoder(data.x, data.edge_index)

print(f"Node embeddings shape: {node_embeddings.shape}")

# ============== t-SNE Visualization ==============
# print("\n" + "="*50)
# print("Generating t-SNE visualization...")
# print("="*50 + "\n")

# labels = data.y.cpu().numpy()
# # t-SNE
# tsne = TSNE(n_components=2, random_state=42)
# embeddings_2d = tsne.fit_transform(node_embeddings.cpu().numpy())

# # Plot
# plt.figure(figsize=(10, 8))

# # Get unique labels
# unique_labels = np.unique(labels)
# colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

# for i, label in enumerate(unique_labels):
#     mask = labels == label
#     plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
#                 color=colors[i], label=f'Class {label}', alpha=0.7, s=20)

# plt.legend(title="Classes")
# plt.title(f't-SNE Visualization of GraphJEPA Embeddings ({Config.dataset_name})')

# # Save plot
# tsne_dir = './tsne_plots'
# os.makedirs(tsne_dir, exist_ok=True)
# tsne_path = os.path.join(tsne_dir, f'tsne_{Config.dataset_name}_{Config.gnn_type}.png')
# plt.savefig(tsne_path)
# print(f"t-SNE plot saved to {tsne_path}")
# plt.close()

# ============== 下游任務：節點分類 ==============
print("\n" + "="*50)
print("Downstream Task: Node Classification")
print("="*50 + "\n")

# 準備訓練/驗證/測試數據
X = node_embeddings.cpu().numpy()
y = data.y.cpu().numpy()

train_mask = data.train_mask.cpu().numpy()
val_mask = data.val_mask.cpu().numpy()
test_mask = data.test_mask.cpu().numpy()

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

# 訓練 Logistic Regression 分類器
classifier = LogisticRegression(max_iter=1000, random_state=42)
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