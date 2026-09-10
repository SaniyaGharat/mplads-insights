import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import itertools

# 1. Siamese Network Architecture
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=44, hidden_dim=16, output_dim=8):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# 2. Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        # label=1 for same class (positive pair), 0 for different class (negative pair)
        # CORRECTED: same-label (1) -> distance should be small (Loss = d^2)
        # different-label (0) -> distance should be large (Loss = max(0, margin-d)^2)
        loss = 0.5 * label * torch.pow(distance, 2) + \
               0.5 * (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        return torch.mean(loss)

def run_siamese_training():
    embeddings_path = 'ml/gat_embeddings.npy'
    labels_path = 'ml/anomaly_labels.csv'
    works_path = 'data/works_sanctioned_clean.csv'

    # Load data
    all_embeddings = np.load(embeddings_path)
    labels_df = pd.read_csv(labels_path)
    df_works = pd.read_csv(works_path)

    # Filter out 'skip'
    labeled_df = labels_df[labels_df['label'].isin(['s', 'n'])].reset_index(drop=True)

    s_indices = labeled_df[labeled_df['label'] == 's']['work_index'].tolist()
    n_indices = labeled_df[labeled_df['label'] == 'n']['work_index'].tolist()

    print(f"Total labeled for training: {len(labeled_df)} (s: {len(s_indices)}, n: {len(n_indices)})")

    # Split labeled works: 80% train, 20% test
    train_df, test_df = train_test_split(labeled_df, test_size=0.2, stratify=labeled_df['label'], random_state=42)

    train_indices = train_df['work_index'].tolist()
    test_indices = test_df['work_index'].tolist()

    print(f"Split: Train set size = {len(train_df)}, Test set size = {len(test_df)}")

    # 3. Construct pairs from train set
    train_s = [idx for idx in train_indices if labeled_df.loc[labeled_df['work_index']==idx, 'label'].values[0] == 's']
    train_n = [idx for idx in train_indices if labeled_df.loc[labeled_df['work_index']==idx, 'label'].values[0] == 'n']

    pairs = []
    for i, j in itertools.combinations(train_s, 2):
        pairs.append((i, j, 1))
    for i, j in itertools.combinations(train_n, 2):
        pairs.append((i, j, 1))
    for s, n in itertools.product(train_s, train_n):
        pairs.append((s, n, 0))

    print(f"Generated {len(pairs)} training pairs: Positive={len([p for p in pairs if p[2]==1])}, Negative={len([p for p in pairs if p[2]==0])}")

    # Prepare tensors
    proj_head = ProjectionHead(input_dim=all_embeddings.shape[1])
    optimizer = optim.Adam(proj_head.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = ContrastiveLoss(margin=1.0)

    # Training loop
    epochs = 100
    train_losses = []
    proj_head.train()

    for epoch in range(epochs):
        random.shuffle(pairs)
        epoch_loss = 0

        batch_size = 16
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i+batch_size]
            u_idx = torch.tensor([p[0] for p in batch])
            v_idx = torch.tensor([p[1] for p in batch])
            labels = torch.tensor([p[2] for p in batch], dtype=torch.float32)

            optimizer.zero_grad()

            z_u = torch.tensor(all_embeddings[u_idx], dtype=torch.float32)
            z_v = torch.tensor(all_embeddings[v_idx], dtype=torch.float32)

            p_u = proj_head(z_u)
            p_v = proj_head(z_v)

            dist = torch.sqrt(torch.sum((p_u - p_v)**2, dim=1) + 1e-7)

            loss = criterion(dist, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / (len(pairs) // batch_size + 1)
        train_losses.append(avg_loss)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {avg_loss:.4f}")

    # 4. Evaluation on held-out set
    proj_head.eval()
    with torch.no_grad():
        all_proj = proj_head(torch.tensor(all_embeddings, dtype=torch.float32)).numpy()
        support_s = [all_proj[idx] for idx in train_s]
        support_n = [all_proj[idx] for idx in train_n]

        def evaluate_shot(k):
            correct = 0
            total = 0
            for idx in test_indices:
                true_label = labeled_df.loc[labeled_df['work_index']==idx, 'label'].values[0]
                s_samples = random.sample(support_s, k)
                n_samples = random.sample(support_n, k)
                dist_s = np.mean([np.linalg.norm(all_proj[idx] - s) for s in s_samples])
                dist_n = np.mean([np.linalg.norm(all_proj[idx] - n) for n in n_samples])
                pred = 's' if dist_s < dist_n else 'n'
                if pred == true_label:
                    correct += 1
                total += 1
            return correct, total

        c1, t1 = evaluate_shot(1)
        c5, t5 = evaluate_shot(5)
        print(f"\nHeld-out Raw Results:")
        print(f"1-shot: {c1}/{t1} correct ({c1/t1:.2%})")
        print(f"5-shot: {c5}/{t5} correct ({c5/t5:.2%})")

    # 5. Confirm Collapsing is Gone
    s_centroid = np.mean(support_s, axis=0)
    distances = np.linalg.norm(all_proj - s_centroid, axis=1)
    sorted_dist = np.sort(distances)
    top_50_unique = len(np.unique(np.round(sorted_dist[:50], 6)))
    print(f"\nCollapsing Check: {top_50_unique} unique distance values in top 50 (out of 50).")

    # Only generate Top-15 if Accuracy > 50% and Collapsing is fixed
    if (c5/t5 > 0.5) and (top_50_unique > 40):
        gat_scores = np.load('ml/gat_scores.npy')
        siamese_top_15_idx = np.argsort(distances)[:15]
        gat_top_15_idx = np.argsort(gat_scores)[-15:]

        print("\n" + "="*60)
        print("Top 15 Works by Siamese Similarity (Few-Shot - CORRECTED)")
        print("="*60)
        print(f"{'Rank':<5} | {'WorkIdx':<8} | {'SiamDist':<10} | {'GATScore':<10} | {'Status'}")
        print("-" * 60)
        new_found = 0
        for rank, idx in enumerate(siamese_top_15_idx, 1):
            gat_score = gat_scores[idx]
            is_new = idx not in gat_top_15_idx
            if is_new: new_found += 1
            status = "NEW" if is_new else "GAT-TOP"
            print(f"{rank:<5} | {idx:<8} | {distances[idx]:<10.4f} | {gat_score:<10.4f} | {status}")
        print(f"\nSiamese found {new_found} NEW suspicious works not in GAT top-15.")
    else:
        print("\nTop-15 table suppressed: Accuracy too low or collapsing still present.")

if __name__ == "__main__":
    run_siamese_training()
