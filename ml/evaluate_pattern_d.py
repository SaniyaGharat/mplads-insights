"""
ml/evaluate_pattern_d.py
========================
Independent Generalization Test of Community Anomaly Scorer vs. GAT
on Pattern D (Slower-moving collusion with loose amounts).

Does NOT modify existing models, checkpoints, or existing synthetic evaluation files.
Saves to:
- ml/mplads_hetero_graph_with_pattern_d.pt
- ml/gat_scores_with_pattern_d.npy
- ml/community_scores_with_pattern_d.npy
- ml/combined_scores_with_pattern_d.npy
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F

from ml.community_anomaly_scorer import compute_community_scores


# -------------------------------------------------------------
# 1. Graph Construction for Pattern D
# -------------------------------------------------------------
def build_pattern_d_graph(csv_path="data/works_with_pattern_d.csv",
                          out_graph_path="ml/mplads_hetero_graph_with_pattern_d.pt"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    n_status = 6
    n_cat = 4

    mp_nodes = sorted(list(set(df["MP"])))
    ida_nodes = sorted(list(set(df["IDA"])))

    mp_map = {name: i for i, name in enumerate(mp_nodes)}
    ida_map = {name: i for i, name in enumerate(ida_nodes)}

    data = HeteroData()
    data["mp"].x = torch.empty((len(mp_nodes), 0))
    data["ida"].x = torch.empty((len(ida_nodes), 0))

    edge_indices = []
    edge_attrs = []

    for _, row in df.iterrows():
        u = mp_map[row["MP"]]
        v = ida_map[row["IDA"]]
        f_amount = np.log1p(float(row["sanction_amount"]))
        f_lag = float(row["sanction_lag_days"])

        status_code = int(row["work_status_code"])
        status_oh = [0.0] * n_status
        status_oh[status_code] = 1.0

        cat_code = int(row["work_category_code"])
        cat_oh = [0.0] * n_cat
        cat_oh[cat_code] = 1.0

        feat = [f_amount, f_lag] + status_oh + cat_oh
        edge_indices.append([u, v])
        edge_attrs.append(feat)

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    data["mp", "sanctions", "ida"].edge_index = edge_index
    data["mp", "sanctions", "ida"].edge_attr = edge_attr

    torch.save(data, out_graph_path)
    print(f"Built and saved graph to {out_graph_path} ({len(mp_nodes)} MPs, {len(ida_nodes)} IDAs, {edge_index.shape[1]} edges)")
    return data, df


# -------------------------------------------------------------
# 2. GAT Architecture & Evaluation (Identical to trained checkpoint)
# -------------------------------------------------------------
class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim):
        super(GATEncoder, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hidden_channels, out_channels, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
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


class GATAnomalyModel(torch.nn.Module):
    def __init__(self, encoder, reconstructor):
        super().__init__()
        self.encoder = encoder
        self.reconstructor = reconstructor

    def forward(self, x, edge_index, edge_attr):
        z = self.encoder(x, edge_index, edge_attr)
        u, v = edge_index
        return self.reconstructor(z[u], z[v])


def run_gat_inference(data, df, ckpt_path="ml/gat_model_artifacts.pth",
                      out_scores_path="ml/gat_scores_with_pattern_d.npy"):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint at {ckpt_path}")

    edge_index_hetero = data['mp', 'sanctions', 'ida'].edge_index
    edge_attr_hetero = data['mp', 'sanctions', 'ida'].edge_attr
    mp_nodes_count = data['mp'].num_nodes
    ida_nodes_count = data['ida'].num_nodes

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

    orig_index = data_homo.edge_index
    orig_attr = data_homo.edge_attr
    rev_index = orig_index[[1, 0], :]
    rev_attr = orig_attr.clone()
    edge_index = torch.cat([orig_index, rev_index], dim=1)
    edge_attr = torch.cat([orig_attr, rev_attr], dim=0)

    edge_mean = edge_attr.mean(dim=0)
    edge_std = edge_attr.std(dim=0) + 1e-7
    edge_attr_norm = (edge_attr - edge_mean) / edge_std

    ckpt = torch.load(ckpt_path, weights_only=False)
    encoder = GATEncoder(4, 16, 16, 12)
    reconstructor = EdgeReconstructor(16, 12)
    model = GATAnomalyModel(encoder, reconstructor)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    with torch.no_grad():
        z = model.encoder(data_homo.x, edge_index, edge_attr_norm)
        u, v = edge_index
        e_hat = model.reconstructor(z[u], z[v])
        edge_errors = torch.sum((edge_attr_norm - e_hat)**2, dim=1)
        errors_np = edge_errors.cpu().numpy()

    num_works = len(df)
    work_errors = errors_np[:num_works]
    np.save(out_scores_path, work_errors)
    print(f"Computed and saved GAT scores to {out_scores_path}")
    return work_errors


# -------------------------------------------------------------
# 3. Main Evaluation Pipeline
# -------------------------------------------------------------
def evaluate_all():
    csv_path = "data/works_with_pattern_d.csv"
    graph_path = "ml/mplads_hetero_graph_with_pattern_d.pt"
    gat_scores_path = "ml/gat_scores_with_pattern_d.npy"
    comm_scores_path = "ml/community_scores_with_pattern_d.npy"
    comb_scores_path = "ml/combined_scores_with_pattern_d.npy"

    data, df = build_pattern_d_graph(csv_path, graph_path)

    # 1. GAT Inference
    print("\n--- Running GAT Inference (Pre-trained, No Retraining) ---")
    gat_scores = run_gat_inference(data, df, out_scores_path=gat_scores_path)

    # 2. Community Scorer (AS-IS, no retuning, 30-day window)
    print("\n--- Running Community Scorer (AS-IS, 30-day window, no changes) ---")
    comm_scores, meta = compute_community_scores(df, window_days=30, eps=0.001)
    np.save(comm_scores_path, comm_scores)
    print(f"Computed and saved Community scores to {comm_scores_path}")

    # 3. Combined Log-Normalized Max
    print("\n--- Computing Combined Log-Normalized Ensemble ---")
    log_gat = np.log1p(gat_scores)
    log_gat_norm = (log_gat - log_gat.min()) / (log_gat.max() - log_gat.min())
    log_comm = np.log1p(comm_scores)
    log_comm_norm = (log_comm - log_comm.min()) / (log_comm.max() - log_comm.min())
    comb_scores = np.maximum(log_gat_norm, log_comm_norm)
    np.save(comb_scores_path, comb_scores)
    print(f"Computed and saved Combined scores to {comb_scores_path}")

    # 4. Evaluation Function
    def eval_score(scores, name):
        order = np.argsort(scores)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(scores) + 1)
        df_res = df.copy()
        df_res['score'] = scores
        df_res['rank'] = ranks
        sub = df_res[df_res['is_synthetic_fraud'] == True]

        top30 = sum(sub['rank'] <= 30)
        top50 = sum(sub['rank'] <= 50)
        top100 = sum(sub['rank'] <= 100)

        print("\n" + "=" * 80)
        print(f"RESULTS: {name}")
        print("=" * 80)
        print(f"Precision@30:  {top30/30:.4f}  ({top30:2d}/30 in Top 30)")
        print(f"Precision@50:  {top50/50:.4f}  ({top50:2d}/50 in Top 50)")
        print(f"Precision@100: {top100/100:.4f}  ({top100:2d}/100 in Top 100)")
        print(f"Pattern D Mean Rank:   {sub['rank'].mean():7.1f} / {len(df)} (Top {sub['rank'].mean()/len(df)*100:.2f}%)")
        print(f"Pattern D Median Rank: {sub['rank'].median():7.1f} / {len(df)} (Top {sub['rank'].median()/len(df)*100:.2f}%)")
        print(f"Pattern D Rank Range:  [{sub['rank'].min()}, {sub['rank'].max()}]")
        return df_res

    df_gat = eval_score(gat_scores, "1. GAT Model Standalone")
    df_comm = eval_score(comm_scores, "2. Community Scorer Standalone (AS-IS)")
    df_comb = eval_score(comb_scores, "3. Combined Signal (Log-Normalized Max)")

    # Per-work breakdown
    print("\n" + "=" * 110)
    print("INDIVIDUAL PATTERN D WORKS: SCORES AND RANKS")
    print("=" * 110)
    print(f"{'Idx':<7} | {'Sanc Date':<11} | {'Amount (Rs)':<12} | {'MP':<32} | {'GAT Rk':<7} | {'Comm Rk':<7} | {'Comm Sc':<8} | {'Comb Rk':<7}")
    print("-" * 110)

    synth_indices = df[df['is_synthetic_fraud'] == True].index
    for idx in synth_indices:
        r = df.loc[idx]
        gat_rk = int(df_gat.loc[idx, 'rank'])
        comm_rk = int(df_comm.loc[idx, 'rank'])
        comm_sc = float(df_comm.loc[idx, 'score'])
        comb_rk = int(df_comb.loc[idx, 'rank'])
        print(f"{r['work_index']:<7} | {r['Sanction Date']:<11} | {r['sanction_amount']:>10,.0f} | {r['MP'][:30]:<32} | {gat_rk:<7} | {comm_rk:<7} | {comm_sc:<8.3f} | {comb_rk:<7}")
    print("=" * 110)

if __name__ == "__main__":
    evaluate_all()
