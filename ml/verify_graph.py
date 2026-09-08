import torch
from torch_geometric.data import HeteroData
import numpy as np

def verify_graph():
    path = "ml/mplads_hetero_graph.pt"
    try:
        data = torch.load(path, weights_only=False)
    except FileNotFoundError:
        print(f"Error: {path} not found.")
        return

    # 1. Number of nodes
    n_mp = data["mp"].x.shape[0]
    n_ida = data["ida"].x.shape[0]
    print(f"Number of 'mp' nodes: {n_mp}")
    print(f"Number of 'ida' nodes: {n_ida}")

    # 2. Number of edges
    edge_index = data["mp", "sanctions", "ida"].edge_index
    n_edges = edge_index.shape[1]
    print(f"Number of edges ('mp', 'sanctions', 'ida'): {n_edges}")

    # 3. Shape of edge_attr
    edge_attr = data["mp", "sanctions", "ida"].edge_attr
    print(f"Shape of edge_attr tensor: {edge_attr.shape}")

    # 4. NaN or Inf check
    has_nan = torch.isnan(edge_attr).any().item()
    has_inf = torch.isinf(edge_attr).any().item()
    print(f"Contains NaN: {has_nan}")
    print(f"Contains Inf: {has_inf}")

    # 5. Min/Max of each column
    print("\nEdge feature min/max:")
    for i in range(edge_attr.shape[1]):
        col = edge_attr[:, i]
        print(f" Column {i}: min={col.min().item():.4f}, max={col.max().item():.4f}")

if __name__ == "__main__":
    verify_graph()
