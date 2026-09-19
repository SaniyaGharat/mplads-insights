# MPLADS Fraud & Anomaly Detection (SIH26102)

## 1. Project Summary

This project tackles Problem Statement **SIH26102**: detecting anomalies, irregularities, and potential fund-misallocation patterns in the **Member of Parliament Local Area Development Scheme (MPLADS)** sanctioning data. We formulate the fund-flow ecosystem as a bipartite/heterogeneous graph where **MPs** recommend works and **Implementing District Authorities (IDAs)** execute sanctions. Our multi-stage AI framework combines:
1. **Unsupervised Graph Autoencoder (VGAE)**: Establishes baseline topological reconstruction error on node and structural connectivity (benchmarked on the Elliptic transaction dataset).
2. **Edge-Feature-Aware Graph Attention Network (GAT)**: Evaluates complex multidimensional work sanctions (sanction amount, recommendation-to-sanction lag, category, status) with intrinsic attention-based explainability that isolates the top influencing neighbor nodes for each flagged work.
3. **Few-Shot Siamese Network**: Refines anomaly scoring via contrastive loss trained on 60 expert analyst labels (`s`, `n`, `skip`) to align graph representations with human investigative feedback.
4. **Analyst Review Console**: A modern dashboard connected to a high-performance FastAPI backend allowing investigators to sort works by risk score, inspect topological explanations, view interactive network subgraphs, and submit real-time labels with duplicate guards.

---

## 2. Prerequisites

- **Python**: 3.10 to 3.13 (64-bit)
- **Node.js**: 18+ (for local frontend development)
- **Core Python Packages**:
  - `torch` (PyTorch 2.x)
  - `torch_geometric` (PyG)
  - `fastapi`
  - `uvicorn`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `networkx`
  - `pydantic`

---

## 3. Setup Instructions

### Step 1: Clone and Create Virtual Environment
```bash
# Clone the repository
git clone <repo-url>
cd atml_project

# Create and activate a Python virtual environment
python -m venv venv

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate
```

### Step 2: Install Python Dependencies
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install fastapi uvicorn pandas numpy scikit-learn networkx pydantic requests
```

### Step 3: Required Data Layout
Ensure the following raw input and baseline label files are present:
```text
atml_project/
├── data/
│   ├── Works_Sanctioned.csv          # Raw MPLADS sanction export file
│   └── elliptic/                     # (Optional) Benchmark dataset for VGAE validation
│       ├── raw/
│       └── processed/
├── ml/
│   ├── anomaly_labels.csv            # 60 canonical analyst labels (28 s, 23 n, 9 skip)
│   └── ... (scripts and models)
├── backend/
│   └── main.py                       # FastAPI application & ML inference endpoints
└── frontend/
    └── ... (TanStack Start / React UI)
```

*(Note: `data/works_sanctioned_clean.csv` is generated automatically during Step 1 of the pipeline below).*

---

## 4. Pipeline Artifact Regeneration (Order of Execution)

To reproduce the machine learning models and dataset artifacts from scratch, execute the pipeline scripts in the exact order below:

```bash
# 1. Clean raw sanction data (if starting from raw export Works_Sanctioned.csv)
python ml/clean_and_build_graph.py data/Works_Sanctioned.csv

# 2. Build PyTorch Geometric HeteroData Graph (Outputs: ml/mplads_hetero_graph.pt)
python ml/build_pyg_graph.py

# 3. Validate VGAE implementation on Elliptic benchmark dataset
python ml/validate_vgae_elliptic.py

# 4. Train Unsupervised VGAE baseline on MPLADS graph
python ml/vgae_mplads_real.py

# 5. Train Edge-Aware GAT Anomaly Model & Scorer
#    (Outputs: ml/gat_model_artifacts.pth and ml/gat_scores.npy)
python ml/gat_anomaly_scorer.py

# 6. Extract 44-dimensional Work Embeddings [z_u, z_v, edge_attr]
#    (Outputs: ml/gat_embeddings.npy)
python ml/extract_gat_embeddings.py

# 7. Train Few-Shot Siamese Projection Head with Contrastive Loss
#    (Outputs: ml/siamese_model.pth)
python ml/siamese_trainer.py
```

---

## 5. Starting the Backend API

To launch the FastAPI backend server:

```bash
# Start backend on http://localhost:8000
python -m uvicorn backend.main:app --port 8000 --reload
```

- **Interactive API Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Key Endpoints**:
  - `GET /api/stats` — Overall statistics, lag comparisons, score distributions.
  - `GET /api/risk-scores?method=gat&page=1&page_size=50` — Paginated ranked works.
  - `GET /api/explain/{work_index}` — Top 3 neighbor GAT attention attribution.
  - `GET /api/graph?top_n=50` — Subgraph topology visualization.
  - `GET /api/label-totals` — Live analyst label counts (`{"s": 28, "n": 23, "skip": 9}`).
  - `POST /api/label` — Submit analyst review (`s`, `n`, `skip`) with 409 duplicate protection.

---

## 6. Frontend Configuration

The analyst user interface can be run locally or accessed via Lovable:

### Running Frontend Locally:
```bash
cd frontend
npm install
npm run dev
```
The local console is available at [http://localhost:5173](http://localhost:5173).

### Remote / Lovable Hosting:
When using the Lovable cloud deployment, the backend must be reachable over the internet. You can expose your local backend securely via tunneling:
```bash
ngrok http 8000
```
Update `API_BASE_URL` in [`frontend/src/lib/api.ts`](frontend/src/lib/api.ts) with the generated public HTTPS URL.

---

## 7. Known Limitations

1. **Few-Shot Label Set Size & Memorization Risk**: The Siamese contrastive projection head is trained on 60 verified human analyst annotations (28 suspicious, 23 normal, 9 skipped). While effective for guided separation, small sample sizes show tendencies toward memorization. Expanding the label corpus via continuous active learning in production is recommended.
2. **Hardware & GNN Nondeterminism**: Despite setting fixed random seeds across PyTorch and NumPy, certain sparse graph convolution operations and CUDA/CPU atomic reductions can exhibit slight floating-point score variations across different hardware architectures.
3. **Data Scope**: Experiments and artifacts are built using a curated 15,000-sanctioned-work export representing 350 MPs and 478 IDAs, rather than the exhaustive multi-decade national MPLADS registry.
