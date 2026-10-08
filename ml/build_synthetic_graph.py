"""
ml/build_synthetic_graph.py
===========================
Builds PyG HeteroData graph containing the 15,000 real works PLUS
the 30 synthetic fraud works from data/works_with_synthetic.csv.

IMPORTANT: Does NOT modify ml/mplads_hetero_graph.pt.
Saves to ml/mplads_hetero_graph_with_synthetic.pt instead.
"""

import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData

def build_synthetic_graph():
    csv_path = "data/works_with_synthetic.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}. Run ml/inject_synthetic_fraud.py first.")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} works from {csv_path}")

    # Standard dimensions
    n_status = 6
    n_cat = 4

    # 1. Create node mappings (consistent with original pipeline sorted lists)
    mp_nodes = sorted(list(set(df["MP"])))
    ida_nodes = sorted(list(set(df["IDA"])))

    mp_map = {name: i for i, name in enumerate(mp_nodes)}
    ida_map = {name: i for i, name in enumerate(ida_nodes)}

    print(f"Nodes: {len(mp_nodes)} MPs, {len(ida_nodes)} IDAs")

    # 2. PyG HeteroData
    data = HeteroData()
    data["mp"].x = torch.empty((len(mp_nodes), 0))
    data["ida"].x = torch.empty((len(ida_nodes), 0))

    # 3. Edge attributes
    edge_indices = []
    edge_attrs = []

    for _, row in df.iterrows():
        u = mp_map[row["MP"]]
        v = ida_map[row["IDA"]]

        # Feature 1: log1p(amount)
        f_amount = np.log1p(float(row["sanction_amount"]))

        # Feature 2: sanction_lag_days
        f_lag = float(row["sanction_lag_days"])

        # Feature 3-8: one-hot status (6 classes)
        status_code = int(row["work_status_code"])
        status_oh = [0.0] * n_status
        status_oh[status_code] = 1.0

        # Feature 9-12: one-hot category (4 classes)
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

    out_path = "ml/mplads_hetero_graph_with_synthetic.pt"
    torch.save(data, out_path)
    print(f"Successfully saved augmented PyG graph to {out_path}")
    print(f"Edges: {edge_index.shape[1]}")
    print(f"Edge attr shape: {edge_attr.shape}")
    return data

if __name__ == "__main__":
    build_synthetic_graph()
