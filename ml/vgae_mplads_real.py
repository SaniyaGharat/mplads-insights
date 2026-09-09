import torch
import numpy as np
import pandas as pd
from torch_geometric.nn import VGAE, GCNConv
from torch_geometric.data import HeteroData
import torch.nn.functional as F

# 1. Encoder Architecture (EXACT SAME as ml/validate_vgae_elliptic.py)
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, 2 * out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return torch.chunk(x, 2, dim=-1)

def run_vgae_mplads():
    # --- Step 1: Load the Graph ---
    print("Loading MPLADS hetero graph...")
    data = torch.load('ml/mplads_hetero_graph.pt', weights_only=False)

    # Edge features from (mp, sanctions, ida)
    # Index 0: log1p(amount), Index 1: sanction_lag_days
    edge_index_hetero = data['mp', 'sanctions', 'ida'].edge_index
    edge_attr_hetero = data['mp', 'sanctions', 'ida'].edge_attr

    mp_nodes_count = data['mp'].num_nodes
    ida_nodes_count = data['ida'].num_nodes

    # --- Step 2: Build Node Features ---
    print("Building node features from aggregated edge stats...")

    # [num_works, mean(log1p(amount)), mean(sanction_lag_days), std(sanction_lag_days)]
    mp_features = torch.zeros((mp_nodes_count, 4))
    ida_features = torch.zeros((ida_nodes_count, 4))

    # We use temporary buffers for mean/std calculation
    # mp_sums: [count, sum_log_amt, sum_lag, sum_lag_sq]
    mp_sums = torch.zeros((mp_nodes_count, 4))
    ida_sums = torch.zeros((ida_nodes_count, 4))

    log_amounts = edge_attr_hetero[:, 0]
    lags = edge_attr_hetero[:, 1]

    # Aggregation for MP nodes (source)
    for i in range(edge_index_hetero.shape[1]):
        u = edge_index_hetero[0, i]
        mp_sums[u, 0] += 1
        mp_sums[u, 1] += log_amounts[i]
        mp_sums[u, 2] += lags[i]
        mp_sums[u, 3] += lags[i]**2

    # Aggregation for IDA nodes (target)
    for i in range(edge_index_hetero.shape[1]):
        v = edge_index_hetero[1, i]
        ida_sums[v, 0] += 1
        ida_sums[v, 1] += log_amounts[i]
        ida_sums[v, 2] += lags[i]
        ida_sums[v, 3] += lags[i]**2

    def compute_stats(sums):
        count = sums[:, 0].unsqueeze(1)
        # Prevent division by zero
        count_safe = torch.where(count == 0, torch.ones_like(count), count)

        mean_log_amt = sums[:, 1].unsqueeze(1) / count_safe
        mean_lag = sums[:, 2].unsqueeze(1) / count_safe

        # Variance = E[X^2] - (E[X])^2
        var_lag = (sums[:, 3].unsqueeze(1) / count_safe) - (mean_lag**2)
        std_lag = torch.sqrt(torch.clamp(var_lag, min=0))

        return torch.cat([count, mean_log_amt, mean_lag, std_lag], dim=1)

    mp_features = compute_stats(mp_sums)
    ida_features = compute_stats(ida_sums)

    print(f"MP node features shape: {mp_features.shape}")
    print(f"IDA node features shape: {ida_features.shape}")

    # --- Step 3: Convert to Homogeneous Graph ---
    print("Converting to homogeneous graph...")

    # Normalize node features to prevent embedding explosion
    def normalize_features(feat):
        mean = feat.mean(dim=0, keepdim=True)
        std = feat.std(dim=0, keepdim=True) + 1e-7
        return (feat - mean) / std

    mp_features_norm = normalize_features(mp_features)
    ida_features_norm = normalize_features(ida_features)

    # Update node features in HeteroData before conversion
    data['mp'].x = mp_features_norm
    data['ida'].x = ida_features_norm

    # to_homogeneous() combines node types and edge types
    data_homo = data.to_homogeneous()
    print(f"Homogeneous Graph -> Nodes: {data_homo.num_nodes}, Edges: {data_homo.num_edges}")

    # --- Step 4: Train VGAE ---
    in_channels = 4
    out_channels = 16

    encoder = GCNEncoder(in_channels, out_channels)
    model = VGAE(encoder)

    # REVERTED to Milestone 3 hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 100

    print("Training VGAE unsupervised...")
    losses = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model.encode(data_homo.x, data_homo.edge_index)
        loss = model.recon_loss(z, data_homo.edge_index)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {loss.item():.4f}")

    # --- Step 5: Reconstruction Error Analysis ---
    model.eval()
    with torch.no_grad():
        z = model.encode(data_homo.x, data_homo.edge_index)

        # Per-edge reconstruction error
        # For an existing edge (u, v), the reconstruction is sigmoid(z_u . z_v)
        # Error = -log(sigmoid(z_u . z_v))
        u, v = data_homo.edge_index
        # Dot product of embeddings for each existing edge
        dot_products = (z[u] * z[v]).sum(dim=1)

        print(f"Dot product stats: Mean={dot_products.mean():.4f}, Std={dot_products.std():.4f}, Max={dot_products.max():.4f}, Min={dot_products.min():.4f}")
        print(f"Embedding z stats: Mean={z.mean():.4f}, Std={z.std():.4f}")

        recon_probs = torch.sigmoid(dot_products)

        # Add small epsilon to avoid log(0)
        recon_errors = -torch.log(recon_probs + 1e-7)

    errors_np = recon_errors.numpy()
    print("\n" + "="*30)
    print("Reconstruction Error Distribution")
    print("="*30)
    print(f"Min:  {np.min(errors_np):.4f}")
    print(f"Max:  {np.max(errors_np):.4f}")
    print(f"Mean: {np.mean(errors_np):.4f}")
    print(f"Std:  {np.std(errors_np):.4f}")
    print(f"50th percentile: {np.percentile(errors_np, 50):.4f}")
    print(f"90th percentile: {np.percentile(errors_np, 90):.4f}")
    print(f"99th percentile: {np.percentile(errors_np, 99):.4f}")
    print("="*30 + "\n")

    # --- Step 6: Anomaly Detection & Reporting ---
    # Calculate reconstruction error for the top 50 to check for collapsing
    top_50_idx = np.argsort(errors_np)[-50:][::-1]
    unique_pairs = set()
    for idx in top_50_idx:
        u, v = data_homo.edge_index[:, idx]
        unique_pairs.add(tuple(sorted((int(u), int(v)))))

    print("\n" + "="*30)
    print("Structural Analysis")
    print("="*30)
    print(f"Top 50 high-error edges represent {len(unique_pairs)} unique node pairs.")
    print("="*30 + "\n")

    # --- Step 7: Top-15 Anomalies ---
    # Map edges back to original works
    df_works = pd.read_csv("data/works_sanctioned_clean.csv")
    top_15_idx = np.argsort(errors_np)[-15:][::-1]

    print("Top 15 Highest Reconstruction Error Works:")
    print(f"{'Rank':<5} | {'Error':<10} | {'MP':<15} | {'IDA':<15} | {'Amount':<12} | {'Lag':<8} | {'Status':<10}")
    print("-" * 85)

    for rank, idx in enumerate(top_15_idx, 1):
        row = df_works.iloc[idx]
        print(f"{rank:<5} | {errors_np[idx]:<10.4f} | {row['MP']:<15} | {row['IDA']:<15} | {row['sanction_amount']:<12.2f} | {int(row['sanction_lag_days']):<8} | {row['work_status_code']:<10}")

if __name__ == "__main__":
    run_vgae_mplads()
