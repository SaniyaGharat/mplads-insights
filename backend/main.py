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
app = FastAPI(title="MPLADS Anomaly Detection Backend")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def is_labelling_enabled() -> bool:
    return os.getenv("LABELLING_ENABLED", "false").strip().lower() in ("true", "1", "yes")

@app.get("/health")
@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "message": "MPLADS backend is running",
        "labelling_enabled": is_labelling_enabled(),
    }

# Global state
DATA = {}

def _prepare_df_works(df: pd.DataFrame):
    if '_district' not in df.columns:
        df['_district'] = df['IDA'].apply(lambda x: str(x).split('(')[0].strip())
    if '_norm_mp' not in df.columns:
        df['_norm_mp'] = df['MP'].apply(lambda x: ' '.join(str(x).replace('(2024-2029)', '').split()).lower())

@app.on_event("startup")
async def startup_event():
    print("Loading artifacts...")
    print("Labels CSV absolute path:", os.path.abspath('ml/anomaly_labels.csv'))
    # 1. Load Base Data
    df_works = pd.read_csv('data/works_sanctioned_clean.csv')
    _prepare_df_works(df_works)
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

    # 5. Load Real-Only Detector Scores & Fusion Scores for Deployment
    real_det_path = 'data/deploy/real_detector_scores.csv'
    if os.path.exists(real_det_path):
        df_real_det = pd.read_csv(real_det_path)
        community_scores = df_real_det['community_score'].values
    else:
        comm_fb = pd.read_csv('data/benchmark/fix3/community_scores_real_only.csv')
        community_scores = comm_fb['community_score'].values

    real_fus_path = 'data/deploy/real_fusion_scores.csv'
    if os.path.exists(real_fus_path):
        df_real_fus = pd.read_csv(real_fus_path)
        fusion_scores = df_real_fus['fusion_score'].values
    else:
        fusion_scores = np.zeros(len(df_works))

    DATA['community_scores'] = community_scores
    DATA['fusion_scores'] = fusion_scores

    # 6. Store in memory
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

@app.get("/api/stats")
async def get_stats():
    df_works = DATA['df_works']
    gat_scores = DATA['gat_scores']
    labels_df = DATA['labels_df']

    total_works = len(df_works)
    total_amount = float(df_works['sanction_amount'].sum())
    avg_lag_days = float(df_works['sanction_lag_days'].mean())

    # label_counts
    labeled_indices = set(labels_df['work_index'].tolist())
    gat_top_15 = set(np.argsort(gat_scores)[-15:])

    training_label_count = 0
    gat_top_count = 0
    new_count = 0

    for idx in range(total_works):
        if idx in labeled_indices:
            training_label_count += 1
        elif idx in gat_top_15:
            gat_top_count += 1
        else:
            new_count += 1

    label_counts = {
        "training_label": training_label_count,
        "gat_top": gat_top_count,
        "new": new_count
    }

    # status_breakdown
    status_counts = df_works['Work Status'].value_counts()
    status_breakdown = [{"status": s, "count": int(c)} for s, c in status_counts.items()]

    # score_distribution
    n_buckets = 10
    percentiles = np.linspace(0, 100, n_buckets + 1)
    bin_edges = np.percentile(gat_scores, percentiles)
    counts, bins = np.histogram(gat_scores, bins=bin_edges)
    score_distribution = []
    for i in range(len(counts)):
        bucket = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
        score_distribution.append({"bucket": bucket, "count": int(counts[i])})

    # top_risky_mps
    # Consistent flagged definition: Training labels OR GAT Top 15
    flagged_indices = labeled_indices.union(gat_top_15)
    flagged_mask = np.zeros(total_works, dtype=bool)
    flagged_mask[list(flagged_indices)] = True

    df_flagged = df_works[flagged_mask]
    mp_stats = df_flagged.groupby('MP').agg(
        flagged_count=('sanction_amount', 'count'),
        total_flagged_amount=('sanction_amount', 'sum')
    ).reset_index()

    top_risky_mps = mp_stats.sort_values('flagged_count', ascending=False).head(10)
    top_risky_mps_list = [
        {"MP": row['MP'], "flagged_count": int(row['flagged_count']), "total_flagged_amount": float(row['total_flagged_amount'])}
        for _, row in top_risky_mps.iterrows()
    ]

    # avg_lag_flagged_vs_normal
    flagged_avg_lag = float(df_works.loc[flagged_mask, 'sanction_lag_days'].mean())
    normal_avg_lag = float(df_works.loc[~flagged_mask, 'sanction_lag_days'].mean())

    return {
        "total_works": total_works,
        "total_amount": total_amount,
        "avg_lag_days": avg_lag_days,
        "label_counts": label_counts,
        "status_breakdown": status_breakdown,
        "score_distribution": score_distribution,
        "top_risky_mps": top_risky_mps_list,
        "avg_lag_flagged_vs_normal": {
            "flagged_avg_lag": flagged_avg_lag,
            "normal_avg_lag": normal_avg_lag
        }
    }


@app.get("/api/methods")
async def get_methods():
    return {
        "methods": [
            {
                "id": "gat",
                "name": "Graph Attention Network (GAT)",
                "direction": "higher_is_riskier",
                "note": "Reconstruction error: flags statistically unusual works. Favours rare values; not evidence of misconduct."
            },
            {
                "id": "siamese",
                "name": "Siamese Network",
                "direction": "lower_is_riskier",
                "note": "Similarity to 51 analyst-labelled works (drawn from GAT-ranked lists, so biased)."
            },
            {
                "id": "community",
                "name": "Community Anomaly Scorer",
                "direction": "higher_is_riskier",
                "note": "Several MPs sanctioning similar amounts at one agency within 30 days. Round-sum coincidences are common."
            },
            {
                "id": "fusion",
                "name": "Learned Fusion",
                "direction": "higher_is_riskier",
                "note": "Experimental. Trained on synthetic fraud patterns, not validated on real fraud labels. A rank-ordering score, not a probability. It disagreed with analyst labels (see report)."
            }
        ]
    }


@app.get("/api/filter-options")
async def get_filter_options():
    df_works = DATA.get('df_works')
    if df_works is None:
        raise HTTPException(status_code=503, detail="Backend data not loaded")
    _prepare_df_works(df_works)

    states = sorted(df_works['State'].dropna().unique().tolist())
    districts_by_state = {}
    for st, group in df_works.groupby('State'):
        districts_by_state[st] = sorted(group['_district'].dropna().unique().tolist())
    districts = sorted(df_works['_district'].dropna().unique().tolist())
    statuses = sorted(df_works['Work Status'].dropna().unique().tolist())

    amount_min = float(df_works['sanction_amount'].min())
    amount_max = float(df_works['sanction_amount'].max())
    lag_min = int(df_works['sanction_lag_days'].min())
    lag_max = int(df_works['sanction_lag_days'].max())

    return {
        "states": states,
        "districts_by_state": districts_by_state,
        "districts": districts,
        "statuses": statuses,
        "amount_min": amount_min,
        "amount_max": amount_max,
        "lag_min": lag_min,
        "lag_max": lag_max,
        "min_amount": amount_min,
        "max_amount": amount_max,
        "min_lag": lag_min,
        "max_lag": lag_max,
    }


def _get_method_ranks(method_name: str) -> np.ndarray:
    cache_key = f"{method_name}_ranks"
    if cache_key in DATA:
        return DATA[cache_key]
    score_key = f"{method_name}_scores"
    if score_key not in DATA:
        if method_name == "gat" and os.path.exists("ml/gat_scores.npy"):
            DATA["gat_scores"] = np.load("ml/gat_scores.npy")
        elif "df_works" in DATA:
            DATA[score_key] = np.zeros(len(DATA['df_works']))
        else:
            DATA[score_key] = np.zeros(15000)
    scores = DATA[score_key]
    if method_name == "siamese":
        sorted_indices = np.argsort(scores)
    else:
        sorted_indices = np.argsort(-scores)
    ranks = np.empty(len(scores), dtype=int)
    ranks[sorted_indices] = np.arange(1, len(scores) + 1)
    DATA[cache_key] = ranks
    return ranks


@app.get("/api/risk-scores")
async def get_risk_scores(
    method: str = "gat",
    page: int = 1,
    page_size: int = 50,
    search: Optional[str] = None,
    state: Optional[str] = None,
    district: Optional[str] = None,
    status: Optional[str] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    min_lag: Optional[int] = None,
    max_lag: Optional[int] = None,
    label_status: Optional[str] = None,
    flag_pct: int = 5,
):
    method_lower = method.lower()
    method_note = ""

    if method_lower == "gat":
        scores = DATA['gat_scores']
        sorted_indices = np.argsort(-scores)
        method_note = "Reconstruction error: flags statistically unusual works. Favours rare values; not evidence of misconduct."
    elif method_lower == "siamese":
        scores = DATA['siamese_scores']
        sorted_indices = np.argsort(scores)
        method_note = "Similarity to 51 analyst-labelled works (drawn from GAT-ranked lists, so biased)."
    elif method_lower == "community":
        scores = DATA['community_scores']
        sorted_indices = np.argsort(-scores)
        method_note = "Several MPs sanctioning similar amounts at one agency within 30 days. Round-sum coincidences are common."
    elif method_lower == "fusion":
        scores = DATA['fusion_scores']
        sorted_indices = np.argsort(-scores)
        method_note = "Experimental. Trained on synthetic fraud patterns, not validated on real fraud labels. A rank-ordering score, not a probability. It disagreed with analyst labels (see report)."
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown method '{method}'. Available methods: gat, siamese, community, fusion"
        )

    df_works = DATA['df_works']
    _prepare_df_works(df_works)

    # 1-based global rank for chosen method across all works
    global_ranks = np.empty(len(scores), dtype=int)
    global_ranks[sorted_indices] = np.arange(1, len(scores) + 1)
    DATA[f"{method_lower}_ranks"] = global_ranks

    # Validate review flag cutoff percentage (in {1, 5, 10})
    if flag_pct not in (1, 5, 10):
        flag_pct = 5
    threshold_rank = int(len(scores) * (flag_pct / 100.0))

    # Labels logic
    labeled_indices = set(DATA['labels_df']['work_index'].tolist())
    gat_top_15 = set(np.argsort(DATA['gat_scores'])[-15:])

    # Filtering mask
    mask = np.ones(len(df_works), dtype=bool)

    if search is not None and search.strip():
        clean_search = search.replace('(2024-2029)', '')
        q = ' '.join(clean_search.strip().lower().split())
        mask &= df_works['_norm_mp'].str.contains(q, regex=False)

    if state is not None and state.strip():
        mask &= (df_works['State'] == state.strip())

    if district is not None and district.strip():
        mask &= (df_works['_district'] == district.strip())

    if status is not None and status.strip():
        mask &= (df_works['Work Status'] == status.strip())

    if min_amount is not None:
        mask &= (df_works['sanction_amount'] >= float(min_amount))

    if max_amount is not None:
        mask &= (df_works['sanction_amount'] <= float(max_amount))

    if min_lag is not None:
        mask &= (df_works['sanction_lag_days'] >= int(min_lag))

    if max_lag is not None:
        mask &= (df_works['sanction_lag_days'] <= int(max_lag))

    if label_status is not None and label_status.strip():
        norm_target = label_status.strip().upper().replace('_', ' ')
        if norm_target not in ("ALL", ""):
            if norm_target in ("TRAINING LABEL", "TRAINING_LABEL"):
                lbl_mask = np.zeros(len(df_works), dtype=bool)
                lbl_mask[list(labeled_indices)] = True
                mask &= lbl_mask
            elif norm_target in ("GAT-TOP", "GAT_TOP"):
                lbl_mask = np.zeros(len(df_works), dtype=bool)
                lbl_mask[list(gat_top_15 - labeled_indices)] = True
                mask &= lbl_mask
            elif norm_target == "NEW":
                lbl_mask = np.ones(len(df_works), dtype=bool)
                lbl_mask[list(labeled_indices | gat_top_15)] = False
                mask &= lbl_mask

    filtered_indices = sorted_indices[mask[sorted_indices]]
    total = int(len(filtered_indices))

    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    page_indices = filtered_indices[start:end]

    results = []
    for idx in page_indices:
        if idx >= len(df_works): continue
        row = df_works.iloc[idx]

        row_status = "NEW"
        if idx in labeled_indices: row_status = "TRAINING LABEL"
        elif idx in gat_top_15: row_status = "GAT-TOP"

        results.append({
            "work_index": int(idx),
            "global_rank": int(global_ranks[idx]),
            "MP": row['MP'],
            "IDA": row['IDA'],
            "amount": float(row['sanction_amount']),
            "sanction_lag_days": int(row['sanction_lag_days']),
            "status": row['Work Status'],
            "category": row['Work category'],
            "score": float(scores[idx]),
            "label_status": row_status,
            "review_flag": bool(global_ranks[idx] <= threshold_rank),
        })

    return {
        "data": results,
        "page": page,
        "page_size": page_size,
        "total": total,
        "method": method_lower,
        "method_note": method_note
    }


@app.get("/api/work/{work_index}/compare")
async def compare_work(work_index: int):
    df_works = DATA.get('df_works')
    if df_works is None or work_index < 0 or work_index >= len(df_works):
        raise HTTPException(status_code=404, detail="Work index not found")

    total_works = len(df_works)
    method_defs = [
        ("gat", "Graph Attention Network (GAT)", "higher_is_riskier"),
        ("siamese", "Siamese Network", "lower_is_riskier"),
        ("community", "Community Anomaly Scorer", "higher_is_riskier"),
        ("fusion", "Learned Fusion", "higher_is_riskier"),
    ]

    methods_data = {}
    for m_id, m_name, direction in method_defs:
        _ = _get_method_ranks(m_id)
        ranks = DATA[f"{m_id}_ranks"]
        scores = DATA[f"{m_id}_scores"]
        rank = int(ranks[work_index])
        pct = float(round((rank / total_works) * 100, 2))
        raw_score = float(scores[work_index])
        methods_data[m_id] = {
            "id": m_id,
            "name": m_name,
            "score": raw_score,
            "global_rank": rank,
            "percentile": pct,
            "direction": direction,
        }

    return {
        "work_index": work_index,
        "total_works": total_works,
        "methods": methods_data,
    }


EVALUATION_CACHE = {}


@app.get("/api/evaluation")
async def get_evaluation():
    if "evaluation_data" in EVALUATION_CACHE:
        return EVALUATION_CACHE["evaluation_data"]

    recall_path = "data/benchmark/fix4/review_budget_recall.csv"
    p1_path = "data/benchmark/fusion/fix2/p1_pooled_summary.csv"
    p2_path = "data/benchmark/fusion/fix2/p2_bootstrap_primary_fgh.csv"

    if not os.path.exists(recall_path) or not os.path.exists(p1_path) or not os.path.exists(p2_path):
        raise HTTPException(status_code=503, detail="Evaluation CSV files not found")

    recall_df = pd.read_csv(recall_path)
    p1_df = pd.read_csv(p1_path)
    p2_df = pd.read_csv(p2_path)

    recall_records = recall_df.replace({np.nan: None}).to_dict(orient="records")
    p1_records = p1_df.replace({np.nan: None}).to_dict(orient="records")
    p2_records = p2_df.replace({np.nan: None}).to_dict(orient="records")

    limitations = [
        "No ground truth, and a 15,000-work sample: True misconduct ground truth does not exist in this 15,000-work sample; labels reflect offline triage inspection, not adjudicated findings.",
        "Synthetic generator signatures: Benchmarks evaluate synthetically injected episodes; models may detect artifactual signatures of the generator rather than real-world corruption.",
        "Typology F probably driven by host selection: Detection performance on Typology F is probably driven by host selection patterns rather than genuine anomaly detection.",
        "Typologies G and H caught by no detector: Subtle distributed patterns (Typologies G and H) are caught by virtually no detector across low triage budgets, remaining near random chance.",
        "Fusion disagreement with analyst labels: Learned fusion models were trained on synthetic fraud patterns and disagreed with offline analyst review labels (which were drawn from GAT-ranked lists).",
    ]

    result = {
        "review_budget_recall": recall_records,
        "p1_pooled_summary": p1_records,
        "p2_bootstrap_primary_fgh": p2_records,
        "limitations": limitations,
    }
    EVALUATION_CACHE["evaluation_data"] = result
    return result

@app.get("/api/graph")
async def get_graph(top_n: int = 50):
    # Return nodes and edges for top N flagged works
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

@app.get("/api/label-totals")
async def get_label_totals():
    labels_df = pd.read_csv('ml/anomaly_labels.csv')
    counts = labels_df['label'].value_counts().to_dict()
    return {
        "s": int(counts.get('s', 0)),
        "n": int(counts.get('n', 0)),
        "skip": int(counts.get('skip', 0))
    }

@app.post("/api/label")
async def add_label(req: LabelRequest):
    if not is_labelling_enabled():
        raise HTTPException(
            status_code=403,
            detail="Labelling is disabled in this deployment"
        )

    # Input Validation
    if not (0 <= req.work_index < 15000):
        raise HTTPException(status_code=400, detail="work_index must be in range [0, 14999]")

    if req.label not in ["s", "n", "skip"]:
        raise HTTPException(status_code=400, detail="label must be one of 's', 'n', or 'skip'")

    # Duplicate check: Prevent adding the same work_index twice
    labels_df = pd.read_csv('ml/anomaly_labels.csv')
    if req.work_index in labels_df['work_index'].values:
        raise HTTPException(status_code=409, detail="This record has already been labeled")

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
