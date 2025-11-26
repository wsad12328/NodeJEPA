class Config:
    # Dataset
    dataset_root = './data/Planetoid'
    dataset_name = 'PubMed'  # 'Cora', 'CiteSeer', 'PubMed'
    
    # Model
    hidden_channels = 256
    num_layers = 1
    ema_decay = 0.996
    dropout = 0.5
    gnn_type = 'GCN' # 'GCN' or 'GAT'
    
    # Positional Encoding
    pe_type = 'RWSE' # 'Laplacian' or 'RWSE'
    pos_dim = 8  # k for RWSE
    
    # Random Walk Sampling
    walk_length = 8 # max_steps
    num_targets = 4 # M (number of target nodes per context node)

    # Training
    lr = 0.001
    weight_decay = 5e-4
    num_epochs = 20
    batch_size = 64 # Increased batch size for node-level sampling
    tau_start = 0.996
    tau_end = 1.0
    
    # Paths
    checkpoint_dir = './checkpoints'
    model_save_path = 'graph_jepa_pretrained.pth'
