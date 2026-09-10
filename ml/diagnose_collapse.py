import torch
import numpy as np
import pandas as pd

data = torch.load('ml/mplads_hetero_graph.pt', weights_only=False)
edge_index = data['mp', 'sanctions', 'ida'].edge_index
edge_attr = data['mp', 'sanctions', 'ida'].edge_attr.numpy()
df_works = pd.read_csv("data/works_sanctioned_clean.csv")

# Check first 10 works that are duplicates
# A work is a duplicate if (MP, IDA, attr) is same.
# We can just use the embeddings we saved.
embeddings = np.load('ml/gat_embeddings.npy')
unique, counts = np.unique(embeddings, axis=0, return_counts=True)

# Find a highly duplicated embedding
max_idx = np.argmax(counts)
target_emb = unique[max_idx]
indices = np.where((embeddings == target_emb).all(axis=1))[0]

print(f"Embedding at index {max_idx} appears {counts[max_idx]} times.")
print("Sample works with this embedding:")
for idx in indices[:5]:
    row = df_works.iloc[idx]
    print(f"Idx: {idx} | MP: {row['MP']} | IDA: {row['IDA']} | Amt: {row['sanction_amount']} | Lag: {row['sanction_lag_days']}")
