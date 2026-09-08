import torch
import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.data import HeteroData
from clean_and_build_graph import clean, load_raw, build_graph

def build_pyg_graph():
    # 1. Get the data using existing logic
    csv_path = "data/works_sanctioned_clean.csv"

    # To be completely consistent with the cleaning pipeline:
    try:
        # Try to load raw if it exists to ensure we run the full clean() logic
        raw_df = pd.read_csv("data/Works_Sanctioned.csv")
        df = clean(raw_df)
    except FileNotFoundError:
        # Fallback to cleaned file
        df = pd.read_csv(csv_path)

    # 2. Create node mappings
    mp_nodes = sorted(list(set(df["MP"])))
    ida_nodes = sorted(list(set(df["IDA"])))

    mp_map = {name: i for i, name in enumerate(mp_nodes)}
    ida_map = {name: i for i, name in enumerate(ida_nodes)}

    # 3. PyG HeteroData
    data = HeteroData()

    # Node types
    data["mp"].x = torch.empty((len(mp_nodes), 0))
    data["ida"].x = torch.empty((len(ida_nodes), 0))

    # 4. Edge attributes with One-Hot Encoding
    edge_indices = []
    edge_attrs = []

    # We expect 6 unique statuses and 4 unique categories as per requirements
    # If the data has fewer, we still pad to 6 and 4.
    n_status = 6
    n_cat = 4

    for _, row in df.iterrows():
        u = mp_map[row["MP"]]
        v = ida_map[row["IDA"]]

        # Feature 1: log1p(amount)
        f_amount = np.log1p(row["sanction_amount"])

        # Feature 2: sanction_lag_days
        f_lag = row["sanction_lag_days"]

        # Feature 3-8: one-hot status
        status_code = int(row["work_status_code"])
        status_oh = [0.0] * n_status
        if 0 <= status_code < n_status:
            status_oh[status_code] = 1.0

        # Feature 9-12: one-hot category
        cat_code = int(row["work_category_code"])
        cat_oh = [0.0] * n_cat
        if 0 <= cat_code < n_cat:
            cat_oh[cat_code] = 1.0

        feat = [f_amount, f_lag] + status_oh + cat_oh

        edge_indices.append([u, v])
        edge_attrs.append(feat)

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    data["mp", "sanctions", "ida"].edge_index = edge_index
    data["mp", "sanctions", "ida"].edge_attr = edge_attr

    # Save the HeteroData object
    torch.save(data, "ml/mplads_hetero_graph.pt")
    print(f"Saved PyG graph to ml/mplads_hetero_graph.pt")
    print(f"Nodes: mp={len(mp_nodes)}, ida={len(ida_nodes)}")
    print(f"Edges: {edge_index.shape[1]}")
    print(f"Edge attr shape: {edge_attr.shape}")

if __name__ == "__main__":
    build_pyg_graph()
