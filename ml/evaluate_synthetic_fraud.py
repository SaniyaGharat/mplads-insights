"""
ml/evaluate_synthetic_fraud.py
==============================
Evaluates the ALREADY-TRAINED GAT model (ml/gat_model_artifacts.pth)
on the augmented MPLADS graph (ml/mplads_hetero_graph_with_synthetic.pt).

DOES NOT RETRAIN ANY MODEL.
DOES NOT MODIFY ml/mplads_hetero_graph.pt OR ml/gat_scores.npy.
Saves scores to ml/gat_scores_with_synthetic.npy.

Computes and reports:
- Precision@30, Precision@50, Precision@100
- Mean and Median rank of all 30 synthetic works (out of 15,030)
- Per-pattern breakdown (Pattern A, Pattern B, Pattern C)
- Comprehensive anomaly score analysis
"""

import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F

# 1. GAT Encoder Architecture (identical to ml/gat_anomaly_scorer.py)
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

# 2. Edge Reconstructor (identical to ml/gat_anomaly_scorer.py)
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

class GATAnomalyModel(torch.nn.Module):
    def __init__(self, encoder, reconstructor):
        super().__init__()
        self.encoder = encoder
        self.reconstructor = reconstructor

    def forward(self, x, edge_index, edge_attr):
        z = self.encoder(x, edge_index, edge_attr)
        u, v = edge_index
        return self.reconstructor(z[u], z[v])

def evaluate():
    graph_path = "ml/mplads_hetero_graph_with_synthetic.pt"
    ckpt_path = "ml/gat_model_artifacts.pth"
    csv_path = "data/works_with_synthetic.csv"

    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Missing {graph_path}. Run ml/build_synthetic_graph.py first.")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing trained model checkpoint at {ckpt_path}.")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing combined dataset at {csv_path}.")

    print("Loading augmented graph...")
    data = torch.load(graph_path, weights_only=False)
    df = pd.read_csv(csv_path)

    edge_index_hetero = data['mp', 'sanctions', 'ida'].edge_index
    edge_attr_hetero = data['mp', 'sanctions', 'ida'].edge_attr
    mp_nodes_count = data['mp'].num_nodes
    ida_nodes_count = data['ida'].num_nodes

    print(f"Augmented graph has {mp_nodes_count} MPs, {ida_nodes_count} IDAs, {edge_index_hetero.shape[1]} edges.")

    # Compute node features using exact same logic as gat_anomaly_scorer.py
    print("Computing node features...")
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

    data['mp'].x = normalize(compute_stats(mp_sums))
    data['ida'].x = normalize(compute_stats(ida_sums))

    data_homo = data.to_homogeneous()

    # Manually create undirected edges to preserve exact forward edge order
    orig_index = data_homo.edge_index
    orig_attr = data_homo.edge_attr
    rev_index = orig_index[[1, 0], :]
    rev_attr = orig_attr.clone()
    edge_index = torch.cat([orig_index, rev_index], dim=1)
    edge_attr = torch.cat([orig_attr, rev_attr], dim=0)

    edge_mean = edge_attr.mean(dim=0)
    edge_std = edge_attr.std(dim=0) + 1e-7
    edge_attr_norm = (edge_attr - edge_mean) / edge_std

    # Load ALREADY-TRAINED model
    print(f"Loading pre-trained GAT model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, weights_only=False)

    node_in_dim = 4
    hidden_dim = 16
    node_out_dim = 16
    edge_dim = 12

    encoder = GATEncoder(node_in_dim, hidden_dim, node_out_dim, edge_dim)
    reconstructor = EdgeReconstructor(node_out_dim, edge_dim)
    model = GATAnomalyModel(encoder, reconstructor)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    print("Computing reconstruction anomaly scores (no retraining)...")
    with torch.no_grad():
        z = model.encoder(data_homo.x, edge_index, edge_attr_norm)
        u, v = edge_index
        e_hat = model.reconstructor(z[u], z[v])
        edge_errors = torch.sum((edge_attr_norm - e_hat)**2, dim=1)
        errors_np = edge_errors.cpu().numpy()

    num_works = len(df) # 15030
    work_errors = errors_np[:num_works]

    # Save scores to separate file
    out_scores_path = "ml/gat_scores_with_synthetic.npy"
    np.save(out_scores_path, work_errors)
    print(f"Saved scores to {out_scores_path}")

    # ---------------------------------------------------------
    # Evaluation Metrics
    # ---------------------------------------------------------
    # Rank 1 = highest error / highest anomaly
    sorted_order = np.argsort(work_errors)[::-1] # indices sorted by descending score
    ranks = np.empty_like(sorted_order)
    ranks[sorted_order] = np.arange(1, num_works + 1)

    df['anomaly_score'] = work_errors
    df['rank'] = ranks

    # Identify synthetic works
    synthetic_mask = df['is_synthetic_fraud'] == True
    df_synthetic = df[synthetic_mask].copy()
    synth_indices = set(df_synthetic.index)

    top_30_indices = set(sorted_order[:30])
    top_50_indices = set(sorted_order[:50])
    top_100_indices = set(sorted_order[:100])

    p_at_30 = len(top_30_indices.intersection(synth_indices)) / 30.0
    p_at_50 = len(top_50_indices.intersection(synth_indices)) / 50.0
    p_at_100 = len(top_100_indices.intersection(synth_indices)) / 100.0

    all_synth_ranks = df_synthetic['rank'].values
    mean_rank = np.mean(all_synth_ranks)
    median_rank = np.median(all_synth_ranks)

    print("\n" + "="*80)
    print("SYNTHETIC FRAUD INJECTION VALIDATION RESULTS")
    print("="*80)
    print(f"Total works evaluated:        {num_works} (15,000 real + 30 synthetic)")
    print(f"Synthetic fraud works planted: {len(df_synthetic)}")
    print("-" * 80)
    print(f"Precision@30:  {p_at_30:.4f}  ({len(top_30_indices.intersection(synth_indices))}/30 in Top 30)")
    print(f"Precision@50:  {p_at_50:.4f}  ({len(top_50_indices.intersection(synth_indices))}/50 in Top 50)")
    print(f"Precision@100: {p_at_100:.4f}  ({len(top_100_indices.intersection(synth_indices))}/100 in Top 100)")
    print("-" * 80)
    print(f"Overall Synthetic Mean Rank:   {mean_rank:.1f} / {num_works} (Top {mean_rank/num_works*100:.2f}%)")
    print(f"Overall Synthetic Median Rank: {median_rank:.1f} / {num_works} (Top {median_rank/num_works*100:.2f}%)")
    print(f"Synthetic Min Rank (Best):     {np.min(all_synth_ranks)}")
    print(f"Synthetic Max Rank (Worst):    {np.max(all_synth_ranks)}")

    # ---------------------------------------------------------
    # Breakdown by Pattern
    # ---------------------------------------------------------
    print("\n" + "="*80)
    print("PER-PATTERN BREAKDOWN")
    print("="*80)
    print(f"{'Pattern':<35} | {'In Top30':<9} | {'In Top50':<9} | {'In Top100':<10} | {'Mean Rank':<10} | {'Median Rank':<11} | {'Rank Range'}")
    print("-" * 110)

    patterns = [
        "Pattern A: Shell contractor cluster",
        "Pattern B: Fast-tracked large sanction",
        "Pattern C: Ghost work"
    ]

    for p in patterns:
        sub = df_synthetic[df_synthetic['fraud_pattern'] == p]
        sub_indices = set(sub.index)
        in_30 = len(top_30_indices.intersection(sub_indices))
        in_50 = len(top_50_indices.intersection(sub_indices))
        in_100 = len(top_100_indices.intersection(sub_indices))
        m_rk = sub['rank'].mean()
        med_rk = sub['rank'].median()
        min_rk = sub['rank'].min()
        max_rk = sub['rank'].max()
        print(f"{p:<35} | {in_30:<9} | {in_50:<9} | {in_100:<10} | {m_rk:<10.1f} | {med_rk:<11.1f} | [{min_rk}, {max_rk}]")

    print("\n" + "="*120)
    print("INDIVIDUAL SYNTHETIC WORKS SCORES AND RANKS")
    print("="*120)
    print(f"{'Work Idx':<9} | {'Pattern':<12} | {'Rank':<6} | {'Score':<8} | {'Amount':<12} | {'Lag':<5} | {'Status':<22} | {'MP':<26} | {'IDA':<20}")
    print("-" * 120)
    for _, r in df_synthetic.sort_values(by='rank').iterrows():
        p_short = r['fraud_pattern'].split(':')[0]
        st_str = f"{r['Work Status']} ({r['work_status_code']})"
        print(f"{int(r['work_index']):<9} | {p_short:<12} | {int(r['rank']):<6} | {r['anomaly_score']:<8.4f} | {r['sanction_amount']:<12,.0f} | {int(r['sanction_lag_days']):<5} | {st_str:<22} | {r['MP'][:24]:<26} | {r['IDA'][:18]:<20}")
    print("="*120 + "\n")

    return df

if __name__ == "__main__":
    evaluate()
