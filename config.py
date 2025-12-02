import yaml
import os
import argparse

class Config:
    # Default values (will be overwritten by load_config)
    dataset_root = './data/Planetoid'
    dataset_name = 'PubMed'
    
    hidden_channels = 256
    num_layers = 1
    ema_decay = 0.996
    dropout = 0.5
    gnn_type = 'GCN'
    
    pe_type = 'RWSE'
    pos_dim = 8
    
    walk_length = 8
    num_targets = 4
    num_neighbors = [-1] * 6

    lr = 0.001
    weight_decay = 5e-4
    num_epochs = 20
    batch_size = 64
    tau_start = 0.996
    tau_end = 1.0
    seed = 42
    seeds = [42]
    
    checkpoint_dir = './checkpoints'
    model_save_path = 'graph_jepa_pretrained.pth'
    pe_dir = './data/Planetoid/PubMed/processed'

    @classmethod
    def load(cls, dataset_name):
        config_path = os.path.join('configs', f'{dataset_name}.yaml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found.")
            
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        # Update class attributes
        cls.dataset_root = config_dict['dataset']['root']
        cls.dataset_name = config_dict['dataset']['name']
        
        cls.hidden_channels = config_dict['model']['hidden_channels']
        cls.num_layers = config_dict['model']['num_layers']
        cls.ema_decay = config_dict['model']['ema_decay']
        cls.dropout = config_dict['model']['dropout']
        cls.gnn_type = config_dict['model']['gnn_type']
        
        cls.pe_type = config_dict['pe']['type']
        cls.pos_dim = config_dict['pe']['pos_dim']
        
        cls.walk_length = config_dict['sampling']['walk_length']
        cls.num_targets = config_dict['sampling']['num_targets']
        cls.num_neighbors = config_dict['sampling'].get('num_neighbors', [-1] * 6)
        
        cls.lr = config_dict['training']['lr']
        cls.weight_decay = config_dict['training']['weight_decay']
        cls.num_epochs = config_dict['training']['num_epochs']
        cls.batch_size = config_dict['training']['batch_size']
        cls.tau_start = config_dict['training']['tau_start']
        cls.tau_end = config_dict['training']['tau_end']
        cls.seed = config_dict['training'].get('seed', 42)
        cls.seeds = config_dict['training'].get('seeds', [42])
        
        cls.checkpoint_dir = config_dict['paths']['checkpoint_dir']
        cls.model_save_path = config_dict['paths']['model_save_path']
        cls.pe_dir = config_dict['paths'].get('pe_dir', f'./data/Planetoid/{cls.dataset_name}/processed')
        
        print(f"Configuration loaded for {dataset_name}")
