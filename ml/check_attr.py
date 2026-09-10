import torch
import numpy as np

data = torch.load('ml/mplads_hetero_graph.pt', weights_only=False)
edge_attr = data['mp', 'sanctions', 'ida'].edge_attr.numpy()

unique, counts = np.unique(edge_attr, axis=0, return_counts=True)
print(f"Total edge attrs: {len(edge_attr)}")
print(f"Unique edge attrs: {len(unique)}")
print(f"Duplicates found: {len(edge_attr) - len(unique)}")
if len(unique) < len(edge_attr):
    max_idx = np.argmax(counts)
    print(f"Most frequent attribute appears {counts[max_idx]} times.")
