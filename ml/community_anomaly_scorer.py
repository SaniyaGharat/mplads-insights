"""
ml/community_anomaly_scorer.py
==============================
Community-Level Fraud Detection for MPLADS — Addressing the Pattern A Gap.

Context:
The GAT edge-reconstruction model (ml/gat_anomaly_scorer.py) identifies single-work
anomalies (Patterns B and C) with high precision (top 2-4%), but misses multi-MP
fund-concentration clustering (Pattern A, where 3 MPs channel 10 works to 1 IDA
in a 3-day window with near-identical amounts). GAT only models local edges, not
higher-order community structures.

This module introduces a complementary, graph-structural community anomaly score:
1. For each IDA, computes baseline historical amount variance/CV.
2. For each rolling 30-day window on Sanction Date:
   - Number of distinct MPs sanctioning works to this IDA
   - Total amount sanctioned
   - Amount coefficient of variation (CV = std / mean)
3. Flags multi-MP (>= 2 MPs) clusters with tight amount clustering (low CV).
4. Assigns each individual work a community anomaly score:
   score = 1.0 / (window_cv + 0.001) for the tightest cluster it participates in.
5. Optionally combines GAT reconstruction errors with Community anomaly scores
   via min-max, log-normalized, and percentile ensembling.

DOES NOT MODIFY ANY EXISTING GAT MODEL, WEIGHTS, OR DATASETS.
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def compute_community_scores(df: pd.DataFrame, window_days: int = 30, eps: float = 0.001):
    """
    Computes community-level fund concentration anomaly scores for each work.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'IDA', 'MP', 'sanction_amount', and 'Sanction Date'.
    window_days : int
        Rolling window size in days (default: 30).
    eps : float
        Small constant to avoid division by zero when amounts are identical.
        
    Returns:
    --------
    work_scores : np.ndarray
        Array of community anomaly scores for each row in df.
    work_metadata : pd.DataFrame
        Per-work cluster metrics (cluster distinct MPs, total amount, CV, etc.).
    """
    df = df.copy()
    df['Sanction_Date_dt'] = pd.to_datetime(df['Sanction Date'])
    n_total = len(df)
    
    # 1. Baseline IDA Statistics
    ida_baselines = {}
    for ida, g in df.groupby('IDA'):
        amts = g['sanction_amount'].values
        mean_val = float(np.mean(amts))
        std_val = float(np.std(amts)) if len(amts) > 1 else 0.0
        cv_val = std_val / (mean_val + 1e-6) if len(amts) > 1 else 0.5
        ida_baselines[ida] = {
            'ida_count': len(amts),
            'ida_mean': mean_val,
            'ida_std': std_val,
            'ida_cv': cv_val
        }
        
    work_scores = np.zeros(n_total, dtype=np.float64)
    work_cluster_mps = np.zeros(n_total, dtype=np.int32)
    work_cluster_works = np.zeros(n_total, dtype=np.int32)
    work_cluster_amt = np.zeros(n_total, dtype=np.float64)
    work_cluster_cv = np.full(n_total, np.nan, dtype=np.float64)
    work_cv_ratio = np.zeros(n_total, dtype=np.float64)

    # 2. Fast Rolling Window Analysis per IDA
    window_delta = np.timedelta64(window_days, 'D')
    
    for ida, g in df.groupby('IDA'):
        if len(g) < 2:
            continue
            
        g_sorted = g.sort_values('Sanction_Date_dt')
        indices = g_sorted.index.values
        dates = g_sorted['Sanction_Date_dt'].values
        amounts = g_sorted['sanction_amount'].values
        mps = g_sorted['MP'].values
        n_group = len(g_sorted)
        
        cv_base = ida_baselines[ida]['ida_cv']
        
        # Two-pointer / binary search sliding window
        for i in range(n_group):
            d_start = dates[i]
            d_end = d_start + window_delta
            
            # Find upper bound index in sorted dates
            right_idx = np.searchsorted(dates, d_end, side='right')
            w_idx = indices[i:right_idx]
            w_mps = mps[i:right_idx]
            w_amts = amounts[i:right_idx]
            
            distinct_mps = len(set(w_mps))
            w_count = len(w_amts)
            
            # Community anomaly criteria:
            # Multi-MP (>= 2 distinct MPs) and multi-work (>= 2 works) in window
            if distinct_mps >= 2 and w_count >= 2:
                w_mean = np.mean(w_amts)
                w_std = np.std(w_amts)
                w_cv = float(w_std / (w_mean + 1e-6))
                
                # Fund concentration anomaly score:
                # Inversely proportional to amount variation in cluster
                score = 1.0 / (w_cv + eps)
                tot_amt = float(np.sum(w_amts))
                ratio = float(cv_base / (w_cv + eps))
                
                for row_idx in w_idx:
                    if score > work_scores[row_idx]:
                        work_scores[row_idx] = score
                        work_cluster_mps[row_idx] = distinct_mps
                        work_cluster_works[row_idx] = w_count
                        work_cluster_amt[row_idx] = tot_amt
                        work_cluster_cv[row_idx] = w_cv
                        work_cv_ratio[row_idx] = ratio

    metadata = pd.DataFrame({
        'community_score': work_scores,
        'cluster_distinct_mps': work_cluster_mps,
        'cluster_works_count': work_cluster_works,
        'cluster_total_amount': work_cluster_amt,
        'cluster_amount_cv': work_cluster_cv,
        'cluster_cv_ratio': work_cv_ratio
    }, index=df.index)

    return work_scores, metadata


def evaluate_ranking(scores: np.ndarray, df: pd.DataFrame, score_name: str = "Score"):
    """
    Computes standard ranking metrics on the evaluation dataset.
    """
    num_works = len(df)
    order = np.argsort(scores)[::-1]  # descending score order
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, num_works + 1)
    
    df_eval = df.copy()
    df_eval['eval_score'] = scores
    df_eval['eval_rank'] = ranks
    
    synth_mask = df_eval['is_synthetic_fraud'] == True
    df_synth = df_eval[synth_mask].copy()
    synth_indices = set(df_synth.index)
    
    top_30_idx = set(order[:30])
    top_50_idx = set(order[:50])
    top_100_idx = set(order[:100])
    
    p30 = len(top_30_idx.intersection(synth_indices)) / 30.0
    p50 = len(top_50_idx.intersection(synth_indices)) / 50.0
    p100 = len(top_100_idx.intersection(synth_indices)) / 100.0
    
    synth_ranks = df_synth['eval_rank'].values
    mean_rk = float(np.mean(synth_ranks))
    med_rk = float(np.median(synth_ranks))
    min_rk = int(np.min(synth_ranks))
    max_rk = int(np.max(synth_ranks))
    
    print("\n" + "=" * 90)
    print(f"EVALUATION RESULTS: {score_name}")
    print("=" * 90)
    print(f"Precision@30:  {p30:.4f} ({len(top_30_idx.intersection(synth_indices)):2d}/30 in Top 30)")
    print(f"Precision@50:  {p50:.4f} ({len(top_50_idx.intersection(synth_indices)):2d}/50 in Top 50)")
    print(f"Precision@100: {p100:.4f} ({len(top_100_idx.intersection(synth_indices)):2d}/100 in Top 100)")
    print("-" * 90)
    print(f"Overall Synthetic Mean Rank:   {mean_rk:7.1f} / {num_works} (Top {mean_rk/num_works*100:.2f}%)")
    print(f"Overall Synthetic Median Rank: {med_rk:7.1f} / {num_works} (Top {med_rk/num_works*100:.2f}%)")
    print(f"Synthetic Min / Max Rank:      [{min_rk}, {max_rk}]")
    print("-" * 90)
    print(f"{'Fraud Pattern':<36} | {'In Top30':<8} | {'In Top50':<8} | {'In Top100':<9} | {'Mean Rank':<10} | {'Median Rank':<11} | {'Rank Range'}")
    print("-" * 105)
    
    patterns = [
        "Pattern A: Shell contractor cluster",
        "Pattern B: Fast-tracked large sanction",
        "Pattern C: Ghost work"
    ]
    for p in patterns:
        sub = df_synth[df_synth['fraud_pattern'] == p]
        sub_idx = set(sub.index)
        in30 = len(top_30_idx.intersection(sub_idx))
        in50 = len(top_50_idx.intersection(sub_idx))
        in100 = len(top_100_idx.intersection(sub_idx))
        m_r = sub['eval_rank'].mean()
        md_r = sub['eval_rank'].median()
        mn_r = sub['eval_rank'].min()
        mx_r = sub['eval_rank'].max()
        print(f"{p:<36} | {in30:2d}/10    | {in50:2d}/10    | {in100:2d}/10     | {m_r:8.1f}   | {md_r:8.1f}    | [{mn_r:5d}, {mx_r:5d}]")
    print("=" * 105)
    
    return df_eval


def run_pipeline(csv_path: str = "data/works_with_synthetic.csv",
                 gat_scores_path: str = "ml/gat_scores_with_synthetic.npy",
                 out_comm_scores_path: str = "ml/community_scores_with_synthetic.npy",
                 out_comb_scores_path: str = "ml/combined_scores_with_synthetic.npy"):
    """
    Executes Step 1, Step 2, and Step 3 end-to-end.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing dataset at {csv_path}")
    if not os.path.exists(gat_scores_path):
        raise FileNotFoundError(f"Missing GAT scores at {gat_scores_path}")

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    gat_scores = np.load(gat_scores_path)
    print(f"Loaded {len(df)} works. GAT scores shape: {gat_scores.shape}")

    # =========================================================
    # STEP 1: Compute Community Anomaly Scores
    # =========================================================
    print("\n--- STEP 1: Computing Community-Level Anomaly Scores ---")
    comm_scores, meta = compute_community_scores(df, window_days=30, eps=0.001)
    np.save(out_comm_scores_path, comm_scores)
    print(f"Saved community anomaly scores to {out_comm_scores_path}")

    # Inspect Distribution
    zero_cnt = int(np.sum(comm_scores == 0))
    nonzero_cnt = int(np.sum(comm_scores > 0))
    pcts = np.percentile(comm_scores, [0, 50, 75, 90, 95, 98, 99, 99.5, 99.9, 100])
    
    print("\nCommunity Anomaly Score Distribution across 15,030 works:")
    print(f"  Works with score == 0 (no multi-MP cluster in 30d): {zero_cnt:5d} ({zero_cnt/len(df)*100:.2f}%)")
    print(f"  Works with score >  0 (in multi-MP cluster):        {nonzero_cnt:5d} ({nonzero_cnt/len(df)*100:.2f}%)")
    print(f"  Non-zero score summary: min={comm_scores[comm_scores > 0].min():.4f}, "
          f"median={np.median(comm_scores[comm_scores > 0]):.4f}, "
          f"mean={comm_scores[comm_scores > 0].mean():.4f}, "
          f"max={comm_scores.max():.4f}")
    print(f"  Score Percentiles:")
    print(f"    p50:  {pcts[1]:.4f} | p75:  {pcts[2]:.4f} | p90:  {pcts[3]:.4f} | p95:  {pcts[4]:.4f}")
    print(f"    p98:  {pcts[5]:.4f} | p99:  {pcts[6]:.4f} | p99.5:{pcts[7]:.4f} | p99.9:{pcts[8]:.4f} | max:{pcts[9]:.4f}")

    # =========================================================
    # STEP 2: Evaluate Standalone Community Score & Compare
    # =========================================================
    print("\n--- STEP 2: Evaluating Standalone Performance against Synthetic Ground Truth ---")
    df_gat = evaluate_ranking(gat_scores, df, score_name="GAT Reconstruction Model (Standalone)")
    df_comm = evaluate_ranking(comm_scores, df, score_name="Community Anomaly Scorer (Standalone)")

    # Detail on Pattern A Works
    synth_a = df_comm[df_comm['fraud_pattern'] == 'Pattern A: Shell contractor cluster']
    print("\nPattern A Individual Synthetic Works under Community Score:")
    print(f"{'Work Idx':<9} | {'Rank':<6} | {'Comm Score':<12} | {'Cluster MPs':<12} | {'Cluster CV':<12} | {'Sanction Date':<13} | {'Amount':<14}")
    print("-" * 88)
    for _, r in synth_a.sort_values('eval_rank').iterrows():
        idx = int(r['work_index'])
        rk = int(r['eval_rank'])
        sc = r['eval_score']
        m = meta.loc[_]
        print(f"{idx:<9} | {rk:<6} | {sc:<12.4f} | {int(m['cluster_distinct_mps']):<12} | {m['cluster_amount_cv']:<12.6f} | {r['Sanction Date']:<13} | Rs. {r['sanction_amount']:>10,.0f}")

    # =========================================================
    # STEP 3: Combined Model
    # =========================================================
    print("\n--- STEP 3: Combining GAT and Community Signals ---")
    
    # Strategy 1: Min-Max Normalization Max
    gat_norm_minmax = (gat_scores - gat_scores.min()) / (gat_scores.max() - gat_scores.min())
    comm_norm_minmax = (comm_scores - comm_scores.min()) / (comm_scores.max() - comm_scores.min())
    comb_max_minmax = np.maximum(gat_norm_minmax, comm_norm_minmax)
    
    # Strategy 2: Log-scaled Normalization Max (robust to heavy-tailed GAT edge error)
    log_gat = np.log1p(gat_scores)
    log_gat_norm = (log_gat - log_gat.min()) / (log_gat.max() - log_gat.min())
    log_comm = np.log1p(comm_scores)
    log_comm_norm = (log_comm - log_comm.min()) / (log_comm.max() - log_comm.min())
    comb_max_log = np.maximum(log_gat_norm, log_comm_norm)
    
    # Strategy 3: Percentile Ensemble Max
    gat_pct = (rankdata(gat_scores) - 1.0) / (len(gat_scores) - 1.0)
    comm_pct = np.zeros(len(comm_scores))
    nz = comm_scores > 0
    comm_pct[nz] = rankdata(comm_scores[nz]) / len(comm_scores[nz])
    comb_max_pct = np.maximum(gat_pct, comm_pct)

    # Save primary combined score (comb_max_log provides ideal calibration)
    np.save(out_comb_scores_path, comb_max_log)
    print(f"Saved primary combined anomaly scores to {out_comb_scores_path}")

    # Evaluate all three combination schemes
    evaluate_ranking(comb_max_minmax, df, score_name="Combined Score (Min-Max Max)")
    evaluate_ranking(comb_max_log, df, score_name="Combined Score (Log-Normalized Max - Primary)")
    evaluate_ranking(comb_max_pct, df, score_name="Combined Score (Percentile Max)")

    return df_comm, comb_max_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPLADS Community Anomaly Scorer")
    parser.add_argument("--csv", type=str, default="data/works_with_synthetic.csv")
    parser.add_argument("--gat_scores", type=str, default="ml/gat_scores_with_synthetic.npy")
    parser.add_argument("--out_comm", type=str, default="ml/community_scores_with_synthetic.npy")
    parser.add_argument("--out_comb", type=str, default="ml/combined_scores_with_synthetic.npy")
    args = parser.parse_args()

    run_pipeline(
        csv_path=args.csv,
        gat_scores_path=args.gat_scores,
        out_comm_scores_path=args.out_comm,
        out_comb_scores_path=args.out_comb
    )
