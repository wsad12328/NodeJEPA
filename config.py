class Config:
    # Dataset
    dataset_root = './data/Planetoid'
    dataset_name = 'Cora'
    
    # Model
    hidden_channels = 256
    num_layers = 2
    ema_decay = 0.999
    dropout = 0.5
    
    # Training
    lr = 0.001
    weight_decay = 5e-5
    num_epochs = 50
    batch_size = 64
    
    # Paths
    checkpoint_dir = './checkpoints'
    model_save_path = './checkpoints/graph_jepa_pretrained.pth'
    # model_save_path = './checkpoints/graph_jepa_epoch_30.pth'
