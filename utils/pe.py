import torch
from torch_geometric.utils import get_laplacian, to_dense_adj
from torch_geometric.transforms import AddRandomWalkPE, AddLaplacianEigenvectorPE

def compute_laplacian_pe(data, k):
    """
    Compute Laplacian Positional Encodings using PyG transform (Sparse Eigendecomposition)
    with fallback to dense implementation if convergence fails.
    """
    print(f"Computing Laplacian PE (k={k})...")
    try:
        # Try sparse eigendecomposition first
        # Compute k+1 eigenvectors to skip the first trivial one (eigenvalue 0)
        transform = AddLaplacianEigenvectorPE(k=k+1, attr_name='pe', is_undirected=True)
        data_pe = transform(data.clone())
        pe = data_pe.pe
        pe = pe[:, 1:] # Skip trivial
        print(f"Laplacian PE computed (Sparse). Shape: {pe.shape}")
        return pe
    except Exception as e:
        print(f"Sparse eigendecomposition failed: {e}")
        print("Falling back to dense eigendecomposition...")
        
        # Fallback to dense
        edge_index, edge_weight = get_laplacian(
            data.edge_index, 
            normalization='sym', 
            num_nodes=data.num_nodes
        )
        L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=data.num_nodes)[0]
        
        # Eigendecomposition
        vals, vecs = torch.linalg.eigh(L)
        
        # Take k smallest non-trivial eigenvectors
        pe = vecs[:, 1:k+1] 
        print(f"Laplacian PE computed (Dense). Shape: {pe.shape}")
        return pe

def compute_rwse(data, k):
    """
    Compute Random Walk Structural Encoding (RWSE)
    Returns P_ii^t for t = 1...k
    """
    print(f"Computing Random Walk PE (k={k})...")
    try:
        transform = AddRandomWalkPE(walk_length=k, attr_name='pe')
        data_pe = transform(data.clone())
        pe = data_pe.pe
        print(f"Random Walk PE computed. Shape: {pe.shape}")
        return pe
    except Exception as e:
        print(f"RWSE computation failed: {e}")
        # Fallback or re-raise? 
        # For now, let's re-raise as RWSE is usually robust
        raise e

def compute_pe(data, pe_type, k):
    if pe_type == 'Laplacian' or pe_type == 'laplacian':
        return compute_laplacian_pe(data, k)
    elif pe_type == 'RWSE' or pe_type == 'rwse':
        return compute_rwse(data, k)
    else:
        raise ValueError(f"Unknown PE type: {pe_type}")
