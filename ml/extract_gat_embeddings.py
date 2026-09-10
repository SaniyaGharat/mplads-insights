import torch
import numpy as np
import pandas as pd
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import HeteroData
import torch.nn.functional as F

# Reuse the exact architecture from gat_anomaly_scorer.py
class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim):
        super(GATEncoder, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hidden_channels, out_channels, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x

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

def extract_embeddings():
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

    data_homo = data.to_homogeneous()
    orig_index = data_homo.edge_index
    orig_attr = data_homo.edge_attr
    rev_index = orig_index[[1, 0], :]
    rev_attr = orig_attr.clone()
    edge_index = torch.cat([orig_index, rev_index], dim=1)
    edge_attr = torch.cat([orig_attr, rev_attr], dim=0)

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
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model.encoder(data_homo.x, edge_index, edge_attr_norm)
        u, v = edge_index
        e_hat = model.reconstructor(z[u], z[v])
        loss = F.mse_loss(e_hat, edge_attr_norm)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        z = model.encoder(data_homo.x, edge_index, edge_attr_norm)
        # Corrected per-work embedding: [z_u, z_v, edge_attr_norm]
        original_num_edges = 15000
        u_orig = edge_index[0, :original_num_edges]
        v_orig = edge_index[1, :original_num_edges]
        e_attr_orig = edge_attr_norm[:original_num_edges]

        work_embeddings = torch.cat([z[u_orig], z[v_orig], e_attr_orig], dim=-1)
        embeddings_np = work_embeddings.numpy()

    np.save('ml/gat_embeddings.npy', embeddings_np)
    print(f"Saved embeddings for 15,000 works to ml/gat_embeddings.npy. Shape: {embeddings_np.shape}")

if __name__ == "__main__":
    extract_embeddings()
