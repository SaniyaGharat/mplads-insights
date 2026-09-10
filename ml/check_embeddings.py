import numpy as np

embeddings = np.load('ml/gat_embeddings.npy')
unique, counts = np.unique(embeddings, axis=0, return_counts=True)

print(f"Total embeddings: {len(embeddings)}")
print(f"Unique embeddings: {len(unique)}")
print(f"Duplicates found: {len(embeddings) - len(unique)}")

# Find the most common duplicate
if len(unique) < len(embeddings):
    max_idx = np.argmax(counts)
    print(f"Most frequent embedding appears {counts[max_idx]} times.")
