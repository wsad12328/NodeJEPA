import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from model.GraphJEPA import GraphJEPA
import os  
from config import Config

# ============== 載入模型並提取節點嵌入 ==============
print("\n" + "="*50)
print("Loading model and extracting node embeddings...")
print("="*50 + "\n")

# 設定設備
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 載入資料集
dataset = Planetoid(root=Config.dataset_root, name=Config.dataset_name)
data = dataset[0]

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

# 使用 target_encoder 提取所有節點的嵌入（在完整圖上）
with torch.no_grad():
    data = data.to(device)
    node_embeddings = loaded_model.target_encoder.gnn(data.x, data.edge_index)

print(f"Node embeddings shape: {node_embeddings.shape}")

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

train_f1 = f1_score(y_train, train_pred, average='macro')
val_f1 = f1_score(y_val, val_pred, average='macro')
test_f1 = f1_score(y_test, test_pred, average='macro')

print("Node Classification Results:")
print(f"Train Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
print(f"Val Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
print(f"Test Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")