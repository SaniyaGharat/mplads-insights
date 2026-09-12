import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os

# --- Models (copied from ml scripts) ---
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

class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim):
        super(GATEncoder, self).__init__()
        from torch_geometric.nn import GATv2Conv
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
            return x, (edge_index_2, att2)
        else:
            x = self.conv2(x, edge_index, edge_attr)
            return x

# --- Backend App ---
app = FastAPI(title="MPLADS Fraud Detection Backend")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
DATA = {}

@app.on_event("startup")
async def startup_event():
    print("Loading artifacts...")
    # 1. Load Base Data
    df_works = pd.read_csv('data/works_sanctioned_clean.csv')
    labels_df = pd.read_csv('ml/anomaly_labels.csv')
    gat_scores = np.load('ml/gat_scores.npy')
    all_embeddings = np.load('ml/gat_embeddings.npy')

    # 2. Load GAT Artifacts for Explainability
    gat_art = torch.load('ml/gat_model_artifacts.pth', weights_only=False)
    gat_encoder = GATEncoder(4, 16, 16, 12)

    # Extract only encoder weights from the GATAnomalyModel state_dict
    encoder_state = {k.replace('encoder.', ''): v for k, v in gat_art['model_state'].items() if k.startswith('encoder.')}
    gat_encoder.load_state_dict(encoder_state)
    # Note: gat_anomaly_scorer saved a GATAnomalyModel which contains an encoder and reconstructor.
    # We only need the encoder for explainability.
    # Let's fix the loading logic below.

    # 3. Load Siamese Model
    siamese_state = torch.load('ml/siamese_model.pth')

    # Determine dimensions from state_dict to avoid size mismatch
    # weight shape is [out, in]
    h_dim = siamese_state['net.0.weight'].shape[0]
    o_dim = siamese_state['net.2.weight'].shape[0]

    siamese_model = ProjectionHead(44, h_dim, o_dim)
    siamese_model.load_state_dict(siamese_state)
    siamese_model.eval()

    # 4. Precompute Siamese Scores
    with torch.no_grad():
        all_proj = siamese_model(torch.tensor(all_embeddings, dtype=torch.float32)).numpy()
        s_indices = labels_df[labels_df['label'] == 's']['work_index'].tolist()
        s_centroid = np.mean(all_proj[s_indices], axis=0)
        siamese_scores = np.linalg.norm(all_proj - s_centroid, axis=1)

    # 5. Store in memory
    DATA['df_works'] = df_works
    DATA['labels_df'] = labels_df
    DATA['gat_scores'] = gat_scores
    DATA['siamese_scores'] = siamese_scores
    DATA['all_embeddings'] = all_embeddings
    DATA['gat_art'] = gat_art
    DATA['siamese_model'] = siamese_model
    DATA['all_proj'] = all_proj

    # For GAT a bit of a hack: the saved model was a GATAnomalyModel
    # We need to recreate that class to load the state_dict
    class GATAnomalyModel(torch.nn.Module):
        def __init__(self, encoder, reconstructor):
            super().__init__()
            self.encoder = encoder
            self.reconstructor = None # Not needed for explain
        def forward(self, x, edge_index, edge_attr):
            return self.encoder(x, edge_index, edge_attr)

    full_gat = GATAnomalyModel(GATEncoder(4, 16, 16, 12), None)
    full_gat.load_state_dict(gat_art['model_state'], strict=False)
    DATA['gat_model'] = full_gat.encoder

    print("Backend initialized successfully.")

class LabelRequest(BaseModel):
    work_index: int
    label: str

@app.get("/api/risk-scores")
async def get_risk_scores(method: str = "gat", page: int = 1, page_size: int = 50):
    scores = DATA['gat_scores'] if method == "gat" else DATA['siamese_scores']

    # Rank indices
    sorted_indices = np.argsort(scores if method == "siamese" else -scores)

    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    page_indices = sorted_indices[start:end]

    # Labels logic
    labeled_indices = set(DATA['labels_df']['work_index'].tolist())
    gat_top_15 = set(np.argsort(DATA['gat_scores'])[-15:])

    results = []
    for idx in page_indices:
        if idx >= len(DATA['df_works']): continue
        row = DATA['df_works'].iloc[idx]

        status = "NEW"
        if idx in labeled_indices: status = "TRAINING LABEL"
        elif idx in gat_top_15: status = "GAT-TOP"

        results.append({
            "work_index": int(idx),
            "MP": row['MP'],
            "IDA": row['IDA'],
            "amount": float(row['sanction_amount']),
            "sanction_lag_days": int(row['sanction_lag_days']),
            "status": row['Work Status'],
            "category": row['Work category'],
            "score": float(scores[idx]),
            "label_status": status
        })

    return {"data": results, "page": page, "page_size": page_size, "total": len(scores)}

@app.get("/api/graph")
async def get_graph(top_n: int = 50):
    # Return nodes and edges for top N suspicious works
    scores = DATA['gat_scores']
    top_indices = np.argsort(scores)[-top_n:][::-1]

    df_works = DATA['df_works']
    edges = []
    nodes = set()

    for idx in top_indices:
        row = df_works.iloc[idx]
        mp = row['MP']
        ida = row['IDA']
        nodes.add(mp)
        nodes.add(ida)
        edges.append({
            "work_index": int(idx),
            "source": mp,
            "target": ida,
            "amount": float(row['sanction_amount']),
            "lag": int(row['sanction_lag_days']),
            "score": float(scores[idx])
        })

    return {
        "nodes": [{"id": n, "type": "MP" if "2024-2029" in n else "IDA"} for n in nodes],
        "edges": edges
    }

@app.get("/api/explain/{work_index}")
async def explain_work(work_index: int):
    if work_index < 0 or work_index >= len(DATA['df_works']):
        raise HTTPException(status_code=404, detail="Work index not found")

    # Explainability logic from gat_anomaly_scorer.py
    gat_art = DATA['gat_art']
    model = DATA['gat_model']

    # We need the homogeneous graph structure
    edge_index = gat_art['edge_index']
    edge_attr_norm = gat_art['edge_attr_norm']
    data_homo_x = gat_art['data_homo_x']

    # The work_index in the CSV refers to the original edge index in the hetero graph.
    # In the homo graph, orig_index is the first 15k edges.
    u_top = edge_index[0, work_index]

    model.eval()
    with torch.no_grad():
        # We only need attention weights for the layer that produces the final embedding
        # conv2 is the one we want.
        z, (edge_idx_att, weights_att) = model.conv2(
            model.conv1(data_homo_x, edge_index, edge_attr_norm),
            edge_index,
            edge_attr_norm,
            return_attention_weights=True
        )

    if isinstance(weights_att, tuple):
        weights_att = torch.stack(weights_att).mean(dim=0)

    mask = edge_idx_att[1] == u_top
    incident_edges = edge_idx_att[0][mask]
    incident_weights = weights_att[mask]

    non_self_mask = incident_edges != u_top
    filtered_edges = incident_edges[non_self_mask]
    filtered_weights = incident_weights[non_self_mask]

    if len(filtered_edges) == 0:
        return {"work_index": work_index, "top_neighbors": []}

    weights_flat = filtered_weights.view(-1)
    k_val = min(3, weights_flat.size(0))
    top_vals, top_idx = torch.topk(weights_flat, k=k_val)

    mp_nodes = gat_art['mp_nodes']
    ida_nodes = gat_art['ida_nodes']
    mp_count = gat_art['mp_nodes_count']

    neighbors = []
    for i in range(top_vals.shape[0]):
        neighbor_idx = filtered_edges[top_idx[i]].item()
        name = "Unknown"
        if neighbor_idx < mp_count:
            name = mp_nodes[neighbor_idx]
        elif neighbor_idx < (mp_count + len(ida_nodes)):
            name = ida_nodes[neighbor_idx - mp_count]
        neighbors.append({"neighbor": name, "weight": float(top_vals[i].item())})

    return {"work_index": work_index, "top_neighbors": neighbors}

@app.post("/api/label")
async def add_label(req: LabelRequest):
    # Input Validation
    if not (0 <= req.work_index < 15000):
        raise HTTPException(status_code=400, detail="work_index must be in range [0, 14999]")

    if req.label not in ["s", "n", "skip"]:
        raise HTTPException(status_code=400, detail="label must be one of 's', 'n', or 'skip'")

    # Append to CSV
    with open('ml/anomaly_labels.csv', 'a') as f:
        f.write(f"{req.label},{req.work_index}\n")

    # Refresh memory
    DATA['labels_df'] = pd.read_csv('ml/anomaly_labels.csv')

    counts = DATA['labels_df']['label'].value_counts().to_dict()
    return {
        "status": "success",
        "totals": {
            "s": int(counts.get('s', 0)),
            "n": int(counts.get('n', 0)),
            "skip": int(counts.get('skip', 0))
        }
    }
