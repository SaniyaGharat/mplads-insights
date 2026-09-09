import torch
import numpy as np
import pandas as pd
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.nn import VGAE, GCNConv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Encoder Architecture

# 1. Encoder Architecture
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        # Generic 2-layer GCN
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, 2 * out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        # Return mu and logvar for the VGAE
        return torch.chunk(x, 2, dim=-1)

def validate_vgae():
    # 2. Load Real Elliptic Bitcoin Dataset from local raw/ folder
    dataset = EllipticBitcoinDataset(root='./data/elliptic')
    data = dataset[0]

    # 3. Print Dataset Stats
    y = data.y.numpy()
    num_licit = np.sum(y == 0)
    num_illicit = np.sum(y == 1)
    num_unknown = np.sum(y == 2)

    print("\n" + "="*30)
    print("Dataset Sanity Check")
    print("="*30)
    print(f"Num Nodes: {data.num_nodes}")
    print(f"Num Edges: {data.num_edges}")
    print(f"Num Features: {data.num_node_features}")
    print(f"Licit: {num_licit}")
    print(f"Illicit: {num_illicit}")
    print(f"Unknown: {num_unknown}")
    print("="*30 + "\n")

    # Filter labeled nodes for evaluation
    mask_labeled = (data.y != 2)
    y_true = data.y[mask_labeled].numpy()
    labeled_indices = np.where(mask_labeled)[0]

    # 4. Initialize VGAE
    in_channels = data.num_node_features
    out_channels = 16 # Embedding size

    encoder = GCNEncoder(in_channels, out_channels)
    model = VGAE(encoder)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 5. Unsupervised Training
    epochs = 100
    print("Training VGAE unsupervised...")
    losses = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {loss.item():.4f}")

    # 6. Evaluation
    model.eval()
    with torch.no_grad():
        # Use the encoder directly to get mu (the mean of the latent distribution)
        mu, _ = model.encoder(data.x, data.edge_index)
        embeddings = mu.numpy()

    # Filter embeddings to only include labeled nodes
    X_eval = embeddings[labeled_indices]
    y_eval = y_true

    # Train a simple downstream classifier on a subset of labeled embeddings
    X_train, X_test, y_train, y_test = train_test_split(
        X_eval, y_eval, test_size=0.3, stratify=y_eval, random_state=42
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    auprc = average_precision_score(y_test, y_pred_prob)
    f1 = f1_score(y_test, y_pred)

    # 7. Baseline AUPRC (positive class ratio)
    pos_ratio = np.sum(y_true == 1) / len(y_true)

    print("\n" + "="*30)
    print("VGAE Real Elliptic Results")
    print("="*30)
    print(f"Final Training Loss: {losses[-1]:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Baseline AUPRC (Pos Ratio): {pos_ratio:.4f}")
    print("="*30)

    if auprc <= pos_ratio:
        print("\nWARNING: AUPRC is not meaningfully better than baseline.")
    else:
        print("\nSUCCESS: VGAE embeddings provide discriminative power.")

if __name__ == "__main__":
    validate_vgae()
