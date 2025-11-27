import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
import torch

def visualize_embeddings(node_embeddings, labels, dataset_name, gnn_type, save_dir='./tsne_plots', seed=42):
    """
    Generates and saves a t-SNE visualization of node embeddings.
    
    Args:
        node_embeddings (torch.Tensor or np.ndarray): Node embeddings.
        labels (torch.Tensor or np.ndarray): Node labels.
        dataset_name (str): Name of the dataset.
        gnn_type (str): Type of GNN used.
        save_dir (str): Directory to save the plot.
        seed (int): Random seed for t-SNE.
    """
    print("\n" + "="*50)
    print("Generating t-SNE visualization...")
    print("="*50 + "\n")

    if isinstance(node_embeddings, torch.Tensor):
        node_embeddings = node_embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # t-SNE
    tsne = TSNE(n_components=2, random_state=seed)
    embeddings_2d = tsne.fit_transform(node_embeddings)

    # Plot
    plt.figure(figsize=(10, 8))

    # Get unique labels
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                    color=colors[i], label=f'Class {label}', alpha=0.7, s=20)

    plt.legend(title="Classes")
    plt.title(f't-SNE Visualization of GraphJEPA Embeddings ({dataset_name})')

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    tsne_path = os.path.join(save_dir, f'tsne_{dataset_name}_{gnn_type}.png')
    plt.savefig(tsne_path)
    print(f"t-SNE plot saved to {tsne_path}")
    plt.close()
