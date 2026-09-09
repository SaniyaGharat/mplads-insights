import torch
import numpy as np
import pandas as pd
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
import torch.nn.functional as F

# 1. GAT Encoder Architecture
class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim):
        super(GATEncoder, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hidden_channels, out_channels, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr, return_attention=False):
        if return_attention:
            x, (edge_index_1, att1) = self.conv1(x, edge_index, edge_attr, return_attention_weights=True)
        else:
            x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        if return_attention:
            x, (edge_index_2, att2) = self.conv2(x, edge_index, edge_attr, return_attention_weights=True)
        else:
            x = self.conv2(x, edge_index, edge_attr)
        if return_attention:
            return x, (edge_index_2, att2)
        return x

# 2. Edge Reconstructor (Decoder)
class EdgeReconstructor(torch.nn.Module):
    def __init__(self, node_dim, edge_dim):
        super(EdgeReconstructor, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * node_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, edge_dim)
        )

    def forward(self, z_u, z_v):
        combined = torch.cat([z_u, z_v], dim=-1)
        return self.mlp(combined)

def run_gat_anomaly_detection():
    print("Loading MPLADS hetero graph...")
    data = torch.load('ml/mplads_hetero_graph.pt', weights_only=False)

    edge_index_hetero = data['mp', 'sanctions', 'ida'].edge_index
    edge_attr_hetero = data['mp', 'sanctions', 'ida'].edge_attr
    mp_nodes_count = data['mp'].num_nodes
    ida_nodes_count = data['ida'].num_nodes

    print("Building node features...")
    mp_sums = torch.zeros((mp_nodes_count, 4))
    ida_sums = torch.zeros((ida_nodes_count, 4))
    log_amounts = edge_attr_hetero[:, 0]
    lags = edge_attr_hetero[:, 1]

    for i in range(edge_index_hetero.shape[1]):
        u, v = edge_index_hetero[0, i], edge_index_hetero[1, i]
        mp_sums[u, 0] += 1; mp_sums[u, 1] += log_amounts[i]; mp_sums[u, 2] += lags[i]; mp_sums[u, 3] += lags[i]**2
        ida_sums[v, 0] += 1; ida_sums[v, 1] += log_amounts[i]; ida_sums[v, 2] += lags[i]; ida_sums[v, 3] += lags[i]**2

    def compute_stats(sums):
        count = sums[:, 0].unsqueeze(1)
        count_safe = torch.where(count == 0, torch.ones_like(count), count)
        mean_log_amt = sums[:, 1].unsqueeze(1) / count_safe
        mean_lag = sums[:, 2].unsqueeze(1) / count_safe
        var_lag = (sums[:, 3].unsqueeze(1) / count_safe) - (mean_lag**2)
        std_lag = torch.sqrt(torch.clamp(var_lag, min=0))
        return torch.cat([count, mean_log_amt, mean_lag, std_lag], dim=1)

    def normalize(feat):
        return (feat - feat.mean(dim=0)) / (feat.std(dim=0) + 1e-7)

    mp_x = normalize(compute_stats(mp_sums))
    ida_x = normalize(compute_stats(ida_sums))

    data['mp'].x = mp_x
    data['ida'].x = ida_x

    # Use homogeneous conversion
    data_homo = data.to_homogeneous()

    # MANUALLY create undirected edges to preserve original order and avoid ToUndirected() deduplication
    orig_index = data_homo.edge_index
    orig_attr = data_homo.edge_attr
    rev_index = orig_index[[1, 0], :]
    rev_attr = orig_attr.clone()
    edge_index = torch.cat([orig_index, rev_index], dim=1)
    edge_attr = torch.cat([orig_attr, rev_attr], dim=0)

    # Standardize edge attributes
    edge_mean = edge_attr.mean(dim=0)
    edge_std = edge_attr.std(dim=0) + 1e-7
    edge_attr_norm = (edge_attr - edge_mean) / edge_std

    node_in_dim = 4
    hidden_dim = 16
    node_out_dim = 16
    edge_dim = 12

    encoder = GATEncoder(node_in_dim, hidden_dim, node_out_dim, edge_dim)
    reconstructor = EdgeReconstructor(node_out_dim, edge_dim)

    class GATAnomalyModel(torch.nn.Module):
        def __init__(self, encoder, reconstructor):
            super().__init__()
            self.encoder = encoder
            self.reconstructor = reconstructor
        def forward(self, x, edge_index, edge_attr):
            z = self.encoder(x, edge_index, edge_attr)
            u, v = edge_index
            return self.reconstructor(z[u], z[v])

    model = GATAnomalyModel(encoder, reconstructor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 100

    print("Training GAT unsupervised...")
    losses = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model.encoder(data_homo.x, edge_index, edge_attr_norm)
        u, v = edge_index
        e_hat = model.reconstructor(z[u], z[v])
        loss = F.mse_loss(e_hat, edge_attr_norm)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        z = model.encoder(data_homo.x, edge_index, edge_attr_norm)
        u, v = edge_index
        e_hat = model.reconstructor(z[u], z[v])
        edge_errors = torch.sum((edge_attr_norm - e_hat)**2, dim=1)
        errors_np = edge_errors.numpy()

    original_num_edges = 15000
    original_errors = errors_np[:original_num_edges]

    top_50_idx = np.argsort(original_errors)[-50:][::-1]
    unique_pairs = set()
    for idx in top_50_idx:
        u, v = edge_index[:, idx]
        unique_pairs.add(tuple(sorted((int(u), int(v)))))

    print("\n" + "="*30)
    print("GAT Structural Analysis")
    print("="*30)
    print(f"Top 50 high-error edges represent {len(unique_pairs)} unique node pairs.")
    print("="*30 + "\n")

    df_works = pd.read_csv("data/works_sanctioned_clean.csv")
    top_15_idx = np.argsort(original_errors)[-15:][::-1]

    print("Top 15 Highest-Anomaly Works (GAT):")
    print(f"{'Rank':<5} | {'Score':<10} | {'MP':<15} | {'IDA':<15} | {'Amount':<12} | {'Lag':<8} | {'Status':<10}")
    print("-" * 85)
    for rank, idx in enumerate(top_15_idx, 1):
        row = df_works.iloc[idx]
        print(f"{rank:<5} | {original_errors[idx]:<10.4f} | {row['MP']:<15} | {row['IDA']:<15} | {row['sanction_amount']:<12.2f} | {int(row['sanction_lag_days']):<8} | {row['work_status_code']:<10}")

    print("\n" + "="*30)
    print("Explainability Analysis")
    print("="*30)

    mp_nodes = sorted(list(set(df_works["MP"])))
    ida_nodes = sorted(list(set(df_works["IDA"])))

    found_example = False
    for rank in range(1, 11):
        best_idx = top_15_idx[rank-1]
        u_top = edge_index[0, best_idx]

        z_final, (edge_idx_att, weights_att) = model.encoder(data_homo.x, edge_index, edge_attr_norm, return_attention=True)
        if isinstance(weights_att, tuple):
            weights_att = torch.stack(weights_att).mean(dim=0)

        mask = edge_idx_att[1] == u_top
        incident_edges = edge_idx_att[0][mask]
        incident_weights = weights_att[mask]

        non_self_mask = incident_edges != u_top
        filtered_edges = incident_edges[non_self_mask]
        filtered_weights = incident_weights[non_self_mask]

        if len(filtered_edges) >= 1:
            print(f"Analyzing Rank {rank} Anomaly: Work index {best_idx}")
            print(f"Node {u_top.item()} (MP/IDA) top neighbors (excluding self):")

            weights_flat = filtered_weights.view(-1)
            k_val = min(3, weights_flat.size(0))
            if k_val > 0:
                top_vals, top_idx = torch.topk(weights_flat, k=k_val)
                for i in range(top_vals.shape[0]):
                    neighbor_idx = filtered_edges[top_idx[i]].item()
                    name = "Unknown"
                    if neighbor_idx < mp_nodes_count:
                        name = mp_nodes[neighbor_idx]
                    elif neighbor_idx < (mp_nodes_count + ida_nodes_count):
                        name = ida_nodes[neighbor_idx - mp_nodes_count]
                    print(f"Neighbor: {name[:20]:<20} | Weight: {top_vals[i].item():.4f}")
            else:
                print("No non-self neighbors found.")

            found_example = True
            break

    if not found_example:
        print("Could not find a top-anomaly with non-self neighbors.")
    print("="*30)

if __name__ == "__main__":
    run_gat_anomaly_detection()
