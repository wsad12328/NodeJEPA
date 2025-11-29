import torch
import os
import argparse
import yaml
from torch_geometric.datasets import Planetoid, PPI, Reddit
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from utils.pe import compute_pe
from config import Config
from utils.seed import set_seed

def load_config(dataset_name):
    config_path = os.path.join('configs', f'{dataset_name}.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Pre-compute Positional Encodings')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (Cora, CiteSeer, PubMed)')
    args = parser.parse_args()

    dataset_name = args.dataset
    print(f"Processing dataset: {dataset_name}")

    # Load config to get parameters
    try:
        config = load_config(dataset_name)
    except FileNotFoundError:
        print(f"Config file for {dataset_name} not found. Using default Config class.")
        return

    # Set seed
    seed = config['training'].get('seed', 42)
    set_seed(seed)

    dataset_root = config['dataset']['root']
    pe_type = config['pe']['type']
    pos_dim = config['pe']['pos_dim']
    pe_dir = config['paths'].get('pe_dir', f'./data/Planetoid/{dataset_name}/processed')

    # Load Dataset
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root=dataset_root, name=dataset_name)
    elif dataset_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name=dataset_name, root=dataset_root)
        dataset.transform = T.ToUndirected()
    elif dataset_name == 'PPI':
        dataset = PPI(root=dataset_root)
    elif dataset_name == 'Reddit':
        dataset = Reddit(root=dataset_root)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    data = dataset[0]
    
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")

    # Compute PE
    pe = compute_pe(data, pe_type, pos_dim)
    
    # Save PE
    os.makedirs(pe_dir, exist_ok=True)
    pe_path = os.path.join(pe_dir, f'{pe_type.lower()}_pe_{pos_dim}.pt')
    torch.save(pe, pe_path)
    print(f"PE saved to {pe_path}")

if __name__ == "__main__":
    main()
