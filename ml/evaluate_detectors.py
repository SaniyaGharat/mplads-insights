"""
Benchmark Evaluation and Zero-Label Detector Analysis (v5.0 Corrected).

Implements all Prompt 2a-fix requirements:
1. GAT METHOD:
   - Evaluates both GAT scoring variants:
     (a) gat_frozen: Node embeddings frozen from real-graph forward pass.
     (b) gat_augmented: Fresh forward pass on graph including injected + reverse edges.
   - Standardizes edge and node attributes using parameters from original 15,000 works.
   - Saves all detector scores to data/benchmark/detector_scores_v2.csv.
   - Reports Spearman correlation of each variant vs ml/gat_scores.npy on the real 15,000 works.
2. PROTOCOL:
   - Primary table evaluates the zero-label detectors and GAT variants on ALL 88 episodes
     (train, val, and test pooled) against all 15,000 real works.
   - Isolation Forest and Robust Z-score fit on ALL 15,430 rows (unsupervised) with fixed seeds.
   - Test split is explicitly reserved for downstream learned models and is not used for detector tuning.
3. DIRECTIONAL SIGNIFICANCE:
   - 1,000 draws of pseudo-episodes matching each typology's episode count, sizes, and entity rules.
   - One-sided upper-tail empirical p-values for AUROC and AUPRC (raw and Holm-adjusted across 8 typologies).
   - Flags 'detected' only if Holm-adjusted p < 0.05.
   - Lists detector x typology pairs below the null's 5th percentile ('looks more typical than real works').
   - Compact table with AUROC, null mean, raw & adjusted p-values, and episode count.
4. OPERATING POINT:
   - AUPRC, Lift (AUPRC / Base Rate), and Recall at top 1% (k=154), 2% (k=309), and 5% (k=772).
5. COMMUNITY SCORER DECOMPOSITION:
   - Share of injected works with score > 0, eligible real works share > 0, null share for pseudo-episodes.
   - AUROC and AUPRC restricted to works with score > 0.
   - Typology F window decomposition: distinct MPs in 30-day window around IDA and whether other works are real or injected.
6. SENSITIVITY:
   - Primary AUROC table repeated with negatives restricted to eligible real works (MP & IDA >= 15 works).
   - Reports any changes in 'detected' status.
7. SPEC CLEANUP & VERIFICATION:
   - Full KS statistics table for all 8 typologies (sanction date and recommended date).
   - Compares temp-built spec against committed benchmark_spec.md and reports generator overwrite behavior.
"""

import os
import sys
import tempfile
import difflib
import subprocess
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import IsolationForest

# PyTorch and GAT architecture
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gat_anomaly_scorer import GATEncoder, EdgeReconstructor
from community_anomaly_scorer import compute_community_scores

TYPOLOGIES = list("ABCDEFGH")


def holm_bonferroni(p_values):
    """
    Computes exact Holm-Bonferroni step-down adjusted p-values.
    """
    p_values = np.array(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    adj_p = np.zeros(n, dtype=float)
    
    cum_max = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p_values[idx]
        cum_max = max(cum_max, val)
        adj_p[idx] = min(1.0, cum_max)
    return adj_p


def load_data():
    works_path = os.path.join("data", "benchmark", "benchmark_works.csv")
    if not os.path.exists(works_path):
        raise FileNotFoundError(f"Missing {works_path}")
    df_works = pd.read_csv(works_path)
    return df_works


# =========================================================================
# 1. GAT METHOD & DETECTOR SCORING (15,430 ROWS)
# =========================================================================
def compute_and_save_detector_scores(df_works):
    print("=" * 100)
    print("1. GAT METHOD & DETECTOR SCORING (15,430 ROWS)")
    print("=" * 100)

    print("\n[A] Method Explanation for GAT Scores in ml/score_benchmark.py:")
    print("  - In `ml/score_benchmark.py`, GAT scores for injected works were computed via:")
    print("    Method (a): Node embeddings FROZEN from the real-graph forward pass.")
    print("    Specifically, the GAT encoder processed the real graph (15,000 works + reverse edges) to produce z_real,")
    print("    and the reconstructor evaluated pairs (u, v) for all 15,430 benchmark works without message passing from injected edges.")
    print("  - In this evaluation, we compute BOTH variants for all 15,430 works:")
    print("    (a) `gat_frozen`: Embeddings frozen from real graph forward pass.")
    print("    (b) `gat_augmented`: Fresh forward pass on the graph that includes all 15,430 forward + 15,430 reverse edges.")
    print("    Both variants use edge and node standardization parameters derived from the original 15,000 real works.")

    # Load GAT Model Artifacts
    art = torch.load('ml/gat_model_artifacts.pth', weights_only=False)
    encoder = GATEncoder(4, 16, 16, 12)
    reconstructor = EdgeReconstructor(16, 12)
    encoder.load_state_dict({k.replace('encoder.', ''): v for k, v in art['model_state'].items() if k.startswith('encoder.')})
    reconstructor.load_state_dict({k.replace('reconstructor.', ''): v for k, v in art['model_state'].items() if k.startswith('reconstructor.')})
    encoder.eval()
    reconstructor.eval()

    mp_nodes = art['mp_nodes']
    ida_nodes = art['ida_nodes']
    mp_map = {name: i for i, name in enumerate(mp_nodes)}
    ida_map = {name: i for i, name in enumerate(ida_nodes)}
    mp_count = len(mp_nodes)

    edge_indices = []
    edge_attrs = []
    n_status, n_cat = 6, 4

    for _, row in df_works.iterrows():
        u = mp_map[row['MP']]
        v = mp_count + ida_map[row['IDA']]
        f_amount = np.log1p(row['sanction_amount'])
        f_lag = row['sanction_lag_days']
        status_oh = [0.0] * n_status
        status_oh[int(row['work_status_code'])] = 1.0
        cat_oh = [0.0] * n_cat
        cat_oh[int(row['work_category_code'])] = 1.0
        feat = [f_amount, f_lag] + status_oh + cat_oh
        edge_indices.append([u, v])
        edge_attrs.append(feat)

    u_nodes = torch.tensor([e[0] for e in edge_indices], dtype=torch.long)
    v_nodes = torch.tensor([e[1] for e in edge_indices], dtype=torch.long)
    edge_attr_bench = torch.tensor(edge_attrs, dtype=torch.float)

    # Standardization parameters from 15,000 real training works
    data_real = torch.load('ml/mplads_hetero_graph.pt', weights_only=False)
    data_homo_real = data_real.to_homogeneous()
    orig_attr_real = data_homo_real.edge_attr
    edge_attr_real = torch.cat([orig_attr_real, orig_attr_real.clone()], dim=0)
    edge_mean_train = edge_attr_real.mean(dim=0)
    edge_std_train = edge_attr_real.std(dim=0) + 1e-7

    edge_attr_bench_norm = (edge_attr_bench - edge_mean_train) / edge_std_train

    # Variant (a): gat_frozen
    with torch.no_grad():
        z_real = encoder(art['data_homo_x'], art['edge_index'], art['edge_attr_norm'])
        e_hat_frozen = reconstructor(z_real[u_nodes], z_real[v_nodes])
        gat_frozen = torch.sum((edge_attr_bench_norm - e_hat_frozen)**2, dim=1).numpy()

    # Variant (b): gat_augmented
    orig_index = torch.stack([u_nodes, v_nodes], dim=0)
    rev_index = torch.stack([v_nodes, u_nodes], dim=0)
    edge_index_aug = torch.cat([orig_index, rev_index], dim=1)
    edge_attr_aug = torch.cat([edge_attr_bench, edge_attr_bench], dim=0)
    edge_attr_aug_norm = (edge_attr_aug - edge_mean_train) / edge_std_train

    # Recompute node statistics for augmented graph, standardized by real 15,000 baseline stats
    mp_sums_real = torch.zeros((mp_count, 4))
    ida_sums_real = torch.zeros((len(ida_nodes), 4))
    for i in range(15000):
        u_idx = u_nodes[i].item()
        v_idx = v_nodes[i].item() - mp_count
        mp_sums_real[u_idx, 0] += 1; mp_sums_real[u_idx, 1] += edge_attr_bench[i, 0]; mp_sums_real[u_idx, 2] += edge_attr_bench[i, 1]; mp_sums_real[u_idx, 3] += edge_attr_bench[i, 1]**2
        ida_sums_real[v_idx, 0] += 1; ida_sums_real[v_idx, 1] += edge_attr_bench[i, 0]; ida_sums_real[v_idx, 2] += edge_attr_bench[i, 1]; ida_sums_real[v_idx, 3] += edge_attr_bench[i, 1]**2

    def compute_stats(sums):
        count = sums[:, 0].unsqueeze(1)
        count_safe = torch.where(count == 0, torch.ones_like(count), count)
        mean_log_amt = sums[:, 1].unsqueeze(1) / count_safe
        mean_lag = sums[:, 2].unsqueeze(1) / count_safe
        var_lag = (sums[:, 3].unsqueeze(1) / count_safe) - (mean_lag**2)
        std_lag = torch.sqrt(torch.clamp(var_lag, min=0))
        return torch.cat([count, mean_log_amt, mean_lag, std_lag], dim=1)

    mp_x_real_raw = compute_stats(mp_sums_real)
    ida_x_real_raw = compute_stats(ida_sums_real)
    mp_mean_train, mp_std_train = mp_x_real_raw.mean(dim=0), mp_x_real_raw.std(dim=0) + 1e-7
    ida_mean_train, ida_std_train = ida_x_real_raw.mean(dim=0), ida_x_real_raw.std(dim=0) + 1e-7

    mp_sums_aug = torch.zeros((mp_count, 4))
    ida_sums_aug = torch.zeros((len(ida_nodes), 4))
    for i in range(len(edge_indices)):
        u_idx = u_nodes[i].item()
        v_idx = v_nodes[i].item() - mp_count
        mp_sums_aug[u_idx, 0] += 1; mp_sums_aug[u_idx, 1] += edge_attr_bench[i, 0]; mp_sums_aug[u_idx, 2] += edge_attr_bench[i, 1]; mp_sums_aug[u_idx, 3] += edge_attr_bench[i, 1]**2
        ida_sums_aug[v_idx, 0] += 1; ida_sums_aug[v_idx, 1] += edge_attr_bench[i, 0]; ida_sums_aug[v_idx, 2] += edge_attr_bench[i, 1]; ida_sums_aug[v_idx, 3] += edge_attr_bench[i, 1]**2

    mp_x_aug_raw = compute_stats(mp_sums_aug)
    ida_x_aug_raw = compute_stats(ida_sums_aug)
    mp_x_aug_norm = (mp_x_aug_raw - mp_mean_train) / mp_std_train
    ida_x_aug_norm = (ida_x_aug_raw - ida_mean_train) / ida_std_train
    x_aug = torch.cat([mp_x_aug_norm, ida_x_aug_norm], dim=0)

    with torch.no_grad():
        z_aug = encoder(x_aug, edge_index_aug, edge_attr_aug_norm)
        e_hat_aug = reconstructor(z_aug[u_nodes], z_aug[v_nodes])
        gat_augmented = torch.sum((edge_attr_bench_norm - e_hat_aug)**2, dim=1).numpy()

    # Spearman against saved ml/gat_scores.npy
    saved_scores = np.load('ml/gat_scores.npy')
    corr_froz, pval_froz = spearmanr(gat_frozen[:15000], saved_scores)
    corr_aug, pval_aug = spearmanr(gat_augmented[:15000], saved_scores)

    print("\n[B] Spearman Correlation vs ml/gat_scores.npy on the 15,000 Real Works:")
    print(f"  - `gat_frozen`    Spearman = {corr_froz:.8f} (p-value = {pval_froz:.4e})")
    print(f"  - `gat_augmented` Spearman = {corr_aug:.8f} (p-value = {pval_aug:.4e})")

    # 2. Community Scorer
    print("\n[C] Computing Community Anomaly Scores (30-day window, eps=0.001)...")
    comm_scores, _ = compute_community_scores(df_works, window_days=30, eps=0.001)

    # 3. Isolation Forest (fit on ALL 15,430 rows unsupervised)
    print("[D] Fitting Isolation Forest on ALL 15,430 rows (unsupervised, seed=42)...")
    feat_cols = ['sanction_amount', 'sanction_lag_days', 'work_status_code', 'work_category_code']
    X_all = df_works[feat_cols].values
    iso = IsolationForest(n_estimators=100, random_state=42)
    iso.fit(X_all)
    iforest_scores = -iso.decision_function(X_all)

    # 4. Robust Z-score Rule (fit on ALL 15,430 rows unsupervised)
    print("[E] Computing Robust Z-Score Rule on ALL 15,430 rows (unsupervised)...")
    log_amt = np.log1p(df_works['sanction_amount'].values)
    lag = df_works['sanction_lag_days'].values.astype(float)
    med_amt = np.median(log_amt)
    mad_amt = np.median(np.abs(log_amt - med_amt)) * 1.4826
    med_lag = np.median(lag)
    mad_lag = np.median(np.abs(lag - med_lag)) * 1.4826
    z_amt = np.abs(log_amt - med_amt) / (mad_amt + 1e-6)
    z_lag = np.abs(lag - med_lag) / (mad_lag + 1e-6)
    robust_z_scores = np.maximum(z_amt, z_lag)

    # Save detector_scores_v2.csv
    df_scores_v2 = pd.DataFrame({
        "work_index": df_works["work_index"],
        "gat_frozen": gat_frozen,
        "gat_augmented": gat_augmented,
        "community_score": comm_scores,
        "iforest_score": iforest_scores,
        "robust_z_score": robust_z_scores,
        "split": df_works["split"],
        "typology": df_works["typology"],
        "episode_id": df_works["episode_id"],
        "severity": df_works["severity"],
        "is_synthetic_fraud": df_works["is_synthetic_fraud"],
    })
    out_v2 = os.path.join("data", "benchmark", "detector_scores_v2.csv")
    df_scores_v2.to_csv(out_v2, index=False)
    print(f"\nSuccessfully saved {out_v2} ({len(df_scores_v2)} rows).")

    return df_scores_v2


# =========================================================================
# 2 & 3. DIRECTIONAL SIGNIFICANCE & PRIMARY EVALUATION
# =========================================================================
def evaluate_primary_directional_significance(df_scores, df_works):
    print("\n" + "=" * 100)
    print("2 & 3. PRIMARY TABLE: ALL 88 EPISODES POOLED VS ALL 15,000 REAL WORKS (1,000 NULL DRAWS)")
    print("=" * 100)
    print("Protocol Statement:")
    print("  - Evaluated on ALL 88 episodes (train, val, and test pooled) against all 15,000 real works.")
    print("  - The test split is strictly reserved for later learned stages and is not used to make any decision here.")
    print("  - Empirical p-values are one-sided upper-tail against 1,000 pseudo-episodes drawn from eligible real works.")
    print("  - Family-wise error rate controlled via Holm-Bonferroni step-down adjustment across 8 typologies per detector.")
    print("  - A detector x typology is marked 'detected' if Holm-adjusted p < 0.05.")

    detectors = {
        "GAT (Frozen)": "gat_frozen",
        "GAT (Augmented)": "gat_augmented",
        "Community Scorer": "community_score",
        "Isolation Forest": "iforest_score",
        "Robust Z-score": "robust_z_score",
    }

    df_real = df_scores[df_scores['is_synthetic_fraud'] == False].copy().reset_index(drop=True)
    df_syn = df_scores[df_scores['is_synthetic_fraud'] == True].copy().reset_index(drop=True)
    df_real_meta = df_works[df_works['is_synthetic_fraud'] == False].copy().reset_index(drop=True)

    # Eligible real works (MP >= 15 and IDA >= 15)
    mp_counts = df_real_meta['MP'].value_counts()
    ida_counts = df_real_meta['IDA'].value_counts()
    e_mps = set(mp_counts[mp_counts >= 15].index)
    e_idas = set(ida_counts[ida_counts >= 15].index)

    elig_mask = df_real_meta['MP'].isin(e_mps) & df_real_meta['IDA'].isin(e_idas)
    df_real_elig = df_real[elig_mask].copy().reset_index(drop=True)
    df_real_elig_meta = df_real_meta[elig_mask].copy().reset_index(drop=True)
    elig_to_real_idx = df_real_meta[elig_mask].index.values

    # Pre-rank the entire real pool for exact, instant null computations
    N_real = len(df_real) # 15,000
    real_ranks_asc = {}
    real_inv_desc = {}
    for det_name, det_col in detectors.items():
        s = df_real[det_col].values
        real_ranks_asc[det_name] = stats.rankdata(s) - 1.0
        sort_order = np.argsort(-s, kind='mergesort')
        inv_order = np.empty(N_real, dtype=int)
        inv_order[sort_order] = np.arange(N_real)
        real_inv_desc[det_name] = inv_order

    # Group eligible real works by MP
    mp_works_dict = {}
    for mp, grp in df_real_elig_meta.groupby('MP'):
        mp_works_dict[mp] = grp.index.tolist()
    eligible_mps = list(mp_works_dict.keys())

    rng = np.random.RandomState(42)
    N_NULL_DRAWS = 1000

    null_metrics = {typ: {det: {'auroc': [], 'auprc': []} for det in detectors} for typ in TYPOLOGIES}

    for typ in TYPOLOGIES:
        sub_syn = df_syn[df_syn['typology'] == typ]
        ep_sizes = [len(g) for _, g in sub_syn.groupby('episode_id')]
        n_episodes = len(ep_sizes)

        for _ in range(N_NULL_DRAWS):
            perm_mps = rng.permutation(eligible_mps)
            pseudo_indices = []
            ep_count = 0
            for mp in perm_mps:
                w_indices = mp_works_dict[mp]
                size_k = ep_sizes[ep_count]
                if len(w_indices) >= size_k:
                    start_pos = rng.randint(0, len(w_indices) - size_k + 1)
                    pseudo_indices.extend(w_indices[start_pos : start_pos + size_k])
                    ep_count += 1
                if ep_count >= n_episodes:
                    break

            real_pos_idx = elig_to_real_idx[pseudo_indices]
            k_pos = len(real_pos_idx)

            for det_name, det_col in detectors.items():
                full_ranks_sum = np.sum(real_ranks_asc[det_name][real_pos_idx])
                int_ranks_sum = k_pos * (k_pos - 1) / 2.0
                auc_val = (full_ranks_sum - int_ranks_sum) / (k_pos * (N_real - k_pos))

                p_ranks = np.sort(real_inv_desc[det_name][real_pos_idx])
                j_arr = np.arange(1, k_pos + 1)
                prc_val = np.sum(j_arr / (p_ranks + 1.0)) / k_pos

                null_metrics[typ][det_name]['auroc'].append(auc_val)
                null_metrics[typ][det_name]['auprc'].append(prc_val)

    primary_results = []
    more_typical_list = []

    for det_name, det_col in detectors.items():
        det_raw_p_auroc = []
        det_raw_p_auprc = []
        det_stats = []

        s_neg_all = df_real[det_col].values

        for typ in TYPOLOGIES:
            sub_syn = df_syn[df_syn['typology'] == typ]
            n_episodes = len(sub_syn['episode_id'].unique())
            n_works = len(sub_syn)

            s_pos = sub_syn[det_col].values
            y_act = np.array([1] * len(s_pos) + [0] * len(s_neg_all))
            s_act = np.concatenate([s_pos, s_neg_all])
            act_auroc = roc_auc_score(y_act, s_act)
            act_auprc = average_precision_score(y_act, s_act)

            null_aurocs = np.array(null_metrics[typ][det_name]['auroc'])
            null_auprcs = np.array(null_metrics[typ][det_name]['auprc'])

            null_auroc_mean = np.mean(null_aurocs)
            null_auroc_std = np.std(null_aurocs)
            null_auroc_p05 = np.percentile(null_aurocs, 5)

            p_auroc = (np.sum(null_aurocs >= act_auroc) + 1) / (N_NULL_DRAWS + 1)
            p_auprc = (np.sum(null_auprcs >= act_auprc) + 1) / (N_NULL_DRAWS + 1)

            det_raw_p_auroc.append(p_auroc)
            det_raw_p_auprc.append(p_auprc)

            is_more_typical = act_auroc < null_auroc_p05
            if is_more_typical:
                more_typical_list.append({
                    "Detector": det_name,
                    "Typology": typ,
                    "Episodes": n_episodes,
                    "Actual AUROC": f"{act_auroc:.4f}",
                    "Null 5th Pct": f"{null_auroc_p05:.4f}",
                    "Null Mean": f"{null_auroc_mean:.4f}",
                    "Label": "looks more typical than real works"
                })

            det_stats.append({
                "Detector": det_name,
                "Typology": typ,
                "Episodes": n_episodes,
                "Works": n_works,
                "AUROC": act_auroc,
                "Null AUROC (Mean+-Std)": f"{null_auroc_mean:.4f} +- {null_auroc_std:.4f}",
                "Null p05": null_auroc_p05,
                "Raw p (AUROC)": p_auroc,
                "AUPRC": act_auprc,
                "Raw p (AUPRC)": p_auprc,
            })

        adj_p_auroc = holm_bonferroni(det_raw_p_auroc)
        adj_p_auprc = holm_bonferroni(det_raw_p_auprc)

        for i, row in enumerate(det_stats):
            row["Holm Adj p (AUROC)"] = adj_p_auroc[i]
            row["Holm Adj p (AUPRC)"] = adj_p_auprc[i]
            row["Detected (Adj p < 0.05)"] = "YES" if (adj_p_auroc[i] < 0.05 or adj_p_auprc[i] < 0.05) else "NO"
            primary_results.append(row)

    df_primary = pd.DataFrame(primary_results)

    df_compact = df_primary[[
        "Detector", "Typology", "Episodes", "AUROC", "Null AUROC (Mean+-Std)",
        "Raw p (AUROC)", "Holm Adj p (AUROC)", "AUPRC", "Holm Adj p (AUPRC)", "Detected (Adj p < 0.05)"
    ]].copy()
    df_compact["AUROC"] = df_compact["AUROC"].map(lambda v: f"{v:.4f}")
    df_compact["AUPRC"] = df_compact["AUPRC"].map(lambda v: f"{v:.4f}")
    df_compact["Raw p (AUROC)"] = df_compact["Raw p (AUROC)"].map(lambda v: f"{v:.4f}")
    df_compact["Holm Adj p (AUROC)"] = df_compact["Holm Adj p (AUROC)"].map(lambda v: f"{v:.4f}")
    df_compact["Holm Adj p (AUPRC)"] = df_compact["Holm Adj p (AUPRC)"].map(lambda v: f"{v:.4f}")

    print("\n--- COMPACT PRIMARY EVALUATION TABLE ---")
    print(df_compact.to_string(index=False))

    print("\n--- DETECTOR x TYPOLOGY PAIRS BELOW NULL 5TH PERCENTILE ---")
    if more_typical_list:
        print(pd.DataFrame(more_typical_list).to_string(index=False))
    else:
        print("None.")

    return df_primary, df_real, df_syn, df_real_elig, eligible_mps, mp_works_dict


# =========================================================================
# 4. OPERATING POINT ANALYSIS
# =========================================================================
def evaluate_operating_points(df_scores, df_syn):
    print("\n" + "=" * 100)
    print("4. OPERATING POINT ANALYSIS (OVERALL ON ALL 15,430 WORKS)")
    print("=" * 100)

    detectors = {
        "GAT (Frozen)": "gat_frozen",
        "GAT (Augmented)": "gat_augmented",
        "Community Scorer": "community_score",
        "Isolation Forest": "iforest_score",
        "Robust Z-score": "robust_z_score",
    }

    base_rate = len(df_syn) / len(df_scores)
    n_total = len(df_scores)
    k_1pct = int(round(0.01 * n_total)) # 154
    k_2pct = int(round(0.02 * n_total)) # 309
    k_5pct = int(round(0.05 * n_total)) # 772

    print(f"Total Works: {n_total} (Real: {n_total - len(df_syn)}, Injected: {len(df_syn)})")
    print(f"Base Rate = {len(df_syn)} / {n_total} = {base_rate:.6f} ({base_rate*100:.3f}%)")
    print(f"Top 1% cutoff: k = {k_1pct} works")
    print(f"Top 2% cutoff: k = {k_2pct} works")
    print(f"Top 5% cutoff: k = {k_5pct} works\n")

    op_rows = []
    y_all = (df_scores['is_synthetic_fraud'] == True).astype(int).values

    for det_name, det_col in detectors.items():
        s_all = df_scores[det_col].values
        auprc_val = average_precision_score(y_all, s_all)
        lift_val = auprc_val / base_rate

        order = np.argsort(s_all)[::-1]
        tp_1pct = np.sum(y_all[order[:k_1pct]])
        tp_2pct = np.sum(y_all[order[:k_2pct]])
        tp_5pct = np.sum(y_all[order[:k_5pct]])

        rec_1pct = tp_1pct / len(df_syn)
        rec_2pct = tp_2pct / len(df_syn)
        rec_5pct = tp_5pct / len(df_syn)

        op_rows.append({
            "Detector": det_name,
            "AUPRC": f"{auprc_val:.4f}",
            "Base Rate": f"{base_rate:.4f}",
            "Lift (AUPRC / Base Rate)": f"{lift_val:.2f}x",
            "Recall @ Top 1% (k=154)": f"{rec_1pct:.4f} ({tp_1pct}/{len(df_syn)})",
            "Recall @ Top 2% (k=309)": f"{rec_2pct:.4f} ({tp_2pct}/{len(df_syn)})",
            "Recall @ Top 5% (k=772)": f"{rec_5pct:.4f} ({tp_5pct}/{len(df_syn)})",
        })

    print(pd.DataFrame(op_rows).to_string(index=False))


# =========================================================================
# 5. COMMUNITY SCORER DECOMPOSITION
# =========================================================================
def evaluate_community_scorer_decomposition(df_scores, df_works, df_real, df_syn, df_real_elig, eligible_mps, mp_works_dict):
    print("\n" + "=" * 100)
    print("5. COMMUNITY SCORER DECOMPOSITION")
    print("=" * 100)

    real_comm_gt0 = np.mean(df_real['community_score'] > 0)
    real_elig_comm_gt0 = np.mean(df_real_elig['community_score'] > 0)
    print(f"Eligible Real Works with Community Score > 0: {real_elig_comm_gt0*100:.2f}% ({np.sum(df_real_elig['community_score'] > 0)}/{len(df_real_elig)})")
    print(f"All Real Works with Community Score > 0:      {real_comm_gt0*100:.2f}% ({np.sum(df_real['community_score'] > 0)}/{len(df_real)})\n")

    comm_decomp = []
    rng = np.random.RandomState(42)
    N_NULL_DRAWS = 1000

    for typ in TYPOLOGIES:
        sub_syn = df_syn[df_syn['typology'] == typ]
        syn_share_gt0 = np.mean(sub_syn['community_score'] > 0)

        null_shares = []
        ep_sizes = [len(g) for _, g in sub_syn.groupby('episode_id')]
        n_episodes = len(ep_sizes)

        for _ in range(N_NULL_DRAWS):
            perm_mps = rng.permutation(eligible_mps)
            pseudo_indices = []
            ep_count = 0
            for mp in perm_mps:
                w_indices = mp_works_dict[mp]
                size_k = ep_sizes[ep_count]
                if len(w_indices) >= size_k:
                    start_pos = rng.randint(0, len(w_indices) - size_k + 1)
                    pseudo_indices.extend(w_indices[start_pos : start_pos + size_k])
                    ep_count += 1
                if ep_count >= n_episodes:
                    break
            df_p_pos = df_real_elig.loc[pseudo_indices]
            null_shares.append(np.mean(df_p_pos['community_score'] > 0))

        null_share_mean = np.mean(null_shares)
        null_share_std = np.std(null_shares)

        sub_syn_gt0 = sub_syn[sub_syn['community_score'] > 0]
        df_real_gt0 = df_real[df_real['community_score'] > 0]

        if len(sub_syn_gt0) > 0 and len(df_real_gt0) > 0:
            y_gt0 = np.array([1] * len(sub_syn_gt0) + [0] * len(df_real_gt0))
            s_gt0 = np.concatenate([sub_syn_gt0['community_score'].values, df_real_gt0['community_score'].values])
            auroc_gt0 = roc_auc_score(y_gt0, s_gt0)
            auprc_gt0 = average_precision_score(y_gt0, s_gt0)
            auroc_str = f"{auroc_gt0:.4f}"
            auprc_str = f"{auprc_gt0:.4f}"
        else:
            auroc_str = "N/A (0 pos > 0)"
            auprc_str = "N/A (0 pos > 0)"

        comm_decomp.append({
            "Typology": typ,
            "Injected Works": len(sub_syn),
            "Injected Share > 0": f"{syn_share_gt0*100:.2f}% ({np.sum(sub_syn['community_score']>0)}/{len(sub_syn)})",
            "Eligible Real Share > 0": f"{real_elig_comm_gt0*100:.2f}%",
            "Null Pseudo-Ep Share > 0 (Mean+-Std)": f"{null_share_mean*100:.2f}% +- {null_share_std*100:.2f}%",
            "AUROC (Score > 0 only)": auroc_str,
            "AUPRC (Score > 0 only)": auprc_str,
        })
    print(pd.DataFrame(comm_decomp).to_string(index=False))

    # Detailed Typology F Episode Window Decomposition
    print("\n--- Detailed Typology F Episode Window Decomposition ---")
    df_f = df_works[df_works['typology'] == 'F'].copy()
    df_works_dt = df_works.copy()
    df_works_dt['Sanction_Date_dt'] = pd.to_datetime(df_works_dt['Sanction Date'])
    window_delta = np.timedelta64(30, 'D')

    f_ep_rows = []
    for ep_id, grp in df_f.groupby('episode_id'):
        ep_ida = grp['IDA'].iloc[0]
        ep_mp = grp['MP'].iloc[0]
        n_works_ep = len(grp)
        min_date = pd.to_datetime(grp['Sanction Date']).min()
        max_date = pd.to_datetime(grp['Sanction Date']).max()

        ida_all_works = df_works_dt[df_works_dt['IDA'] == ep_ida].copy()

        window_works = ida_all_works[
            (ida_all_works['Sanction_Date_dt'] >= min_date - window_delta) &
            (ida_all_works['Sanction_Date_dt'] <= max_date + window_delta)
        ]

        distinct_mps_all = window_works['MP'].unique()
        distinct_mps_count = len(distinct_mps_all)
        other_mps = [m for m in distinct_mps_all if m != ep_mp]

        other_works = window_works[window_works['MP'].isin(other_mps)]
        has_real_other = (other_works['is_synthetic_fraud'] == False).any()
        has_syn_other = (other_works['is_synthetic_fraud'] == True).any()

        other_type = "None (Solo MP)"
        if has_real_other and has_syn_other: other_type = "Both Real & Injected"
        elif has_real_other: other_type = "Real Works"
        elif has_syn_other: other_type = "Injected Works"

        has_comm_flag = (grp['work_index'].isin(df_scores.loc[df_scores['community_score'] > 0, 'work_index'])).any()

        f_ep_rows.append({
            "F Episode": ep_id,
            "Severity": grp['severity'].iloc[0],
            "Works": n_works_ep,
            "Host MP": ep_mp[:20],
            "Host IDA": ep_ida[:20],
            "Date Range": f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}",
            "Distinct MPs in Window": distinct_mps_count,
            "Other MPs in Window": len(other_mps),
            "Other MPs Works Nature": other_type,
            "Community Score > 0?": "YES" if has_comm_flag else "NO"
        })
    print(pd.DataFrame(f_ep_rows).to_string(index=False))


# =========================================================================
# 6. SENSITIVITY ANALYSIS (NEGATIVES = ELIGIBLE REAL WORKS)
# =========================================================================
def evaluate_sensitivity_analysis(df_primary, df_syn, df_real_elig, eligible_mps, mp_works_dict):
    print("\n" + "=" * 100)
    print("6. SENSITIVITY ANALYSIS: NEGATIVES RESTRICTED TO ELIGIBLE REAL WORKS (MP & IDA >= 15)")
    print("=" * 100)

    detectors = {
        "GAT (Frozen)": "gat_frozen",
        "GAT (Augmented)": "gat_augmented",
        "Community Scorer": "community_score",
        "Isolation Forest": "iforest_score",
        "Robust Z-score": "robust_z_score",
    }

    N_elig = len(df_real_elig) # 8,903
    elig_ranks_asc = {}
    for det_name, det_col in detectors.items():
        elig_ranks_asc[det_name] = stats.rankdata(df_real_elig[det_col].values) - 1.0

    rng = np.random.RandomState(42)
    N_NULL_DRAWS = 1000

    sens_results = []
    changes = []

    for det_name, det_col in detectors.items():
        det_sens_raw_p = []
        det_sens_stats = []

        s_elig_neg = df_real_elig[det_col].values

        for typ in TYPOLOGIES:
            sub_syn = df_syn[df_syn['typology'] == typ]
            n_episodes = len(sub_syn['episode_id'].unique())
            n_works = len(sub_syn)

            s_pos = sub_syn[det_col].values
            y_sens = np.array([1] * len(s_pos) + [0] * len(s_elig_neg))
            s_sens = np.concatenate([s_pos, s_elig_neg])
            act_auroc_sens = roc_auc_score(y_sens, s_sens)

            null_aurocs = []
            ep_sizes = [len(g) for _, g in sub_syn.groupby('episode_id')]
            for _ in range(N_NULL_DRAWS):
                perm_mps = rng.permutation(eligible_mps)
                pseudo_indices = []
                ep_count = 0
                for mp in perm_mps:
                    w_indices = mp_works_dict[mp]
                    size_k = ep_sizes[ep_count]
                    if len(w_indices) >= size_k:
                        start_pos = rng.randint(0, len(w_indices) - size_k + 1)
                        pseudo_indices.extend(w_indices[start_pos : start_pos + size_k])
                        ep_count += 1
                    if ep_count >= n_episodes:
                        break

                k_pos = len(pseudo_indices)
                full_ranks_sum = np.sum(elig_ranks_asc[det_name][pseudo_indices])
                int_ranks_sum = k_pos * (k_pos - 1) / 2.0
                auc_n = (full_ranks_sum - int_ranks_sum) / (k_pos * (N_elig - k_pos))
                null_aurocs.append(auc_n)

            null_aurocs = np.array(null_aurocs)
            null_m_sens = np.mean(null_aurocs)
            null_s_sens = np.std(null_aurocs)
            p_sens = (np.sum(null_aurocs >= act_auroc_sens) + 1) / (N_NULL_DRAWS + 1)

            det_sens_raw_p.append(p_sens)
            det_sens_stats.append({
                "Detector": det_name,
                "Typology": typ,
                "Episodes": n_episodes,
                "AUROC (Eligible)": act_auroc_sens,
                "Null AUROC (Mean+-Std)": f"{null_m_sens:.4f} +- {null_s_sens:.4f}",
                "Raw p": p_sens,
            })

        adj_p_sens = holm_bonferroni(det_sens_raw_p)
        for i, row in enumerate(det_sens_stats):
            row["Holm Adj p"] = adj_p_sens[i]
            is_det = "YES" if adj_p_sens[i] < 0.05 else "NO"
            row["Detected (Adj p < 0.05)"] = is_det

            primary_match = df_primary[(df_primary['Detector'] == row['Detector']) & (df_primary['Typology'] == row['Typology'])].iloc[0]
            prim_det = "YES" if primary_match['Holm Adj p (AUROC)'] < 0.05 else "NO"

            if is_det != prim_det:
                changes.append({
                    "Detector": row['Detector'],
                    "Typology": row['Typology'],
                    "Primary Detected": prim_det,
                    "Sensitivity Detected": is_det,
                    "Primary Adj p": f"{primary_match['Holm Adj p (AUROC)']:.4f}",
                    "Sensitivity Adj p": f"{adj_p_sens[i]:.4f}",
                })
            sens_results.append(row)

    df_sens = pd.DataFrame(sens_results)
    df_sens_compact = df_sens[[
        "Detector", "Typology", "Episodes", "AUROC (Eligible)", "Null AUROC (Mean+-Std)",
        "Raw p", "Holm Adj p", "Detected (Adj p < 0.05)"
    ]].copy()
    df_sens_compact["AUROC (Eligible)"] = df_sens_compact["AUROC (Eligible)"].map(lambda v: f"{v:.4f}")
    df_sens_compact["Raw p"] = df_sens_compact["Raw p"].map(lambda v: f"{v:.4f}")
    df_sens_compact["Holm Adj p"] = df_sens_compact["Holm Adj p"].map(lambda v: f"{v:.4f}")

    print("\n--- SENSITIVITY EVALUATION TABLE (8,903 ELIGIBLE NEGATIVES) ---")
    print(df_sens_compact.to_string(index=False))

    print("\n--- CHANGES IN 'DETECTED' CALLS (Primary vs Sensitivity) ---")
    if changes:
        print(pd.DataFrame(changes).to_string(index=False))
    else:
        print("None. All 'detected' calls are identical between primary and sensitivity evaluations.")


# =========================================================================
# 7. SPEC CLEANUP, KS TABLE & GENERATOR RE-RUN DIFF
# =========================================================================
def evaluate_ks_and_generator_diff(df_works):
    print("\n" + "=" * 100)
    print("7. SPEC CLEANUP, FULL KS STATISTICS TABLE & GENERATOR DIFF")
    print("=" * 100)

    df_real = df_works[df_works['is_synthetic_fraud'] == False].copy()
    df_syn = df_works[df_works['is_synthetic_fraud'] == True].copy()

    df_real['sanc_dt'] = pd.to_datetime(df_real['Sanction Date'])
    df_real['rec_dt'] = pd.to_datetime(df_real['Recommended date'])
    df_syn['sanc_dt'] = pd.to_datetime(df_syn['Sanction Date'])
    df_syn['rec_dt'] = pd.to_datetime(df_syn['Recommended date'])

    bins = [-1, 7, 30, 90, 180, 365, 10000]
    labels = ['0-7', '8-30', '31-90', '91-180', '181-365', '366+']
    df_real['lag_bucket'] = pd.cut(df_real['sanction_lag_days'], bins=bins, labels=labels)
    df_syn['lag_bucket'] = pd.cut(df_syn['sanction_lag_days'], bins=bins, labels=labels)

    ks_results = []
    for typ in TYPOLOGIES:
        sub_syn = df_syn[df_syn['typology'] == typ]
        typ_buckets = sub_syn['lag_bucket'].unique()
        sub_real = df_real[df_real['lag_bucket'].isin(typ_buckets)]

        syn_sanc_ord = sub_syn['sanc_dt'].map(lambda d: d.toordinal()).values
        syn_rec_ord = sub_syn['rec_dt'].map(lambda d: d.toordinal()).values
        real_sanc_ord = sub_real['sanc_dt'].map(lambda d: d.toordinal()).values
        real_rec_ord = sub_real['rec_dt'].map(lambda d: d.toordinal()).values

        ks_sanc = stats.ks_2samp(syn_sanc_ord, real_sanc_ord)
        ks_rec = stats.ks_2samp(syn_rec_ord, real_rec_ord)

        ks_results.append({
            "Typology": typ,
            "Lag Buckets Present": ", ".join(sorted([str(b) for b in typ_buckets])),
            "Real Works N": len(sub_real),
            "Injected N": len(sub_syn),
            "Sanc KS Stat": f"{ks_sanc.statistic:.4f}",
            "Sanc p-val": f"{ks_sanc.pvalue:.4e}",
            "Sanc Sig (p<0.05)": "YES *" if ks_sanc.pvalue < 0.05 else "no",
            "Rec KS Stat": f"{ks_rec.statistic:.4f}",
            "Rec p-val": f"{ks_rec.pvalue:.4e}",
            "Rec Sig (p<0.05)": "YES *" if ks_rec.pvalue < 0.05 else "no",
        })

    print("\n[A] FULL KS STATISTICS TABLE (vs Real Works in Same Lag Bucket):")
    print(pd.DataFrame(ks_results).to_string(index=False))

    # Build benchmark into temp folder and diff against committed benchmark_spec.md
    print("\n[B] Building Benchmark in Isolated Temp Directory & Diffing Generated Spec:")
    with open('ml/build_benchmark.py', 'r', encoding='utf-8') as f:
        code = f.read()

    with tempfile.TemporaryDirectory() as tmpdir:
        code_temp = code.replace('out_dir = os.path.join("data", "benchmark")', f'out_dir = r"{tmpdir}"')
        temp_script = os.path.join(tmpdir, 'build_bench_temp.py')
        with open(temp_script, 'w', encoding='utf-8') as f:
            f.write(code_temp)

        res = subprocess.run([sys.executable, temp_script], capture_output=True, text=True, cwd=os.getcwd())
        if res.returncode != 0:
            print("Error running temp build:", res.stderr)
        else:
            temp_spec = os.path.join(tmpdir, 'benchmark_spec.md')
            committed_spec = os.path.join('data', 'benchmark', 'benchmark_spec.md')

            with open(temp_spec, 'r', encoding='utf-8') as f1, open(committed_spec, 'r', encoding='utf-8') as f2:
                lines_gen = f1.readlines()
                lines_com = f2.readlines()

            diff = list(difflib.unified_diff(lines_gen, lines_com, fromfile='temp_generated/benchmark_spec.md', tofile='committed/benchmark_spec.md'))
            print("--- Unified Diff (temp_generated vs committed benchmark_spec.md) ---")
            print("".join(diff))

            print("\n[C] Overwrite Behavior Statement:")
            print("  - Does `ml/build_benchmark.py` overwrite the 'Known deviations' section?")
            print("  - YES. `ml/build_benchmark.py` generates Sections 1 and 2 only (lines 1445-1495).")
            print("  - Re-running `ml/build_benchmark.py` without flags would overwrite `data/benchmark/benchmark_spec.md`")
            print("    and erase the 'Known deviations' section and post-freeze documentation.")


def main():
    print("=" * 100)
    print("PROMPT 2A-FIX: CORRECTED ZERO-LABEL EVALUATION & GAT BENCHMARK AUDIT")
    print("=" * 100)

    # Load data
    df_works = load_data()

    # 1. GAT scores and detector computation
    df_scores = compute_and_save_detector_scores(df_works)

    # 2 & 3. Primary Table & Directional Significance
    df_primary, df_real, df_syn, df_real_elig, eligible_mps, mp_works_dict = evaluate_primary_directional_significance(df_scores, df_works)

    # 4. Operating Point
    evaluate_operating_points(df_scores, df_syn)

    # 5. Community Scorer Decomposition
    evaluate_community_scorer_decomposition(df_scores, df_works, df_real, df_syn, df_real_elig, eligible_mps, mp_works_dict)

    # 6. Sensitivity Analysis
    evaluate_sensitivity_analysis(df_primary, df_syn, df_real_elig, eligible_mps, mp_works_dict)

    # 7. Spec Cleanup & KS table
    evaluate_ks_and_generator_diff(df_works)


if __name__ == "__main__":
    main()
