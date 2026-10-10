"""
Backend Integration Script: Real-Only Scores, Fusion Model, and Evaluation
Script: ml/run_backend_integration.py
"""

import os
import sys
import json
import joblib
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import roc_auc_score


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 90)
    print("BACKEND INTEGRATION: REAL-ONLY DETECTOR SCORES, FUSION MODEL & SANITY CHECK")
    print("=" * 90)
    
    os.makedirs(os.path.join("data", "deploy"), exist_ok=True)
    
    # =========================================================================
    # STEP 1: REAL-ONLY DETECTOR SCORES
    # =========================================================================
    print("\n--- STEP 1: REAL-ONLY DETECTOR SCORES ---")
    print("Code reference in ml/evaluate_detectors.py for IForest and Robust Z-score:")
    print("  File: ml/evaluate_detectors.py, Lines 216-223 (Isolation Forest):")
    print("    feat_cols = ['sanction_amount', 'sanction_lag_days', 'work_status_code', 'work_category_code']")
    print("    X_all = df_works[feat_cols].values")
    print("    iso = IsolationForest(n_estimators=100, random_state=42)")
    print("    iso.fit(X_all)")
    print("    iforest_scores = -iso.decision_function(X_all)")
    print("  File: ml/evaluate_detectors.py, Lines 224-235 (Robust Z-Score Rule):")
    print("    log_amt = np.log1p(df_works['sanction_amount'].values)")
    print("    lag = df_works['sanction_lag_days'].values.astype(float)")
    print("    med_amt = np.median(log_amt)")
    print("    mad_amt = np.median(np.abs(log_amt - med_amt)) * 1.4826")
    print("    med_lag = np.median(lag)")
    print("    mad_lag = np.median(np.abs(lag - med_lag)) * 1.4826")
    print("    z_amt = np.abs(log_amt - med_amt) / (mad_amt + 1e-6)")
    print("    z_lag = np.abs(lag - med_lag) / (mad_lag + 1e-6)")
    print("    robust_z_scores = np.maximum(z_amt, z_lag)")
    
    # Load 15,000 real works
    df_clean = pd.read_csv('data/works_sanctioned_clean.csv')
    n_real = len(df_clean)
    assert n_real == 15000, f"Expected 15,000 real works, got {n_real}"
    print(f"\nLoaded data/works_sanctioned_clean.csv: {n_real} works.")
    
    # 1. gat_frozen
    gat_frozen = np.load('ml/gat_scores.npy')
    assert len(gat_frozen) == n_real, f"Expected 15,000 gat scores, got {len(gat_frozen)}"
    
    # 2. community_score (real-only)
    comm_df = pd.read_csv('data/benchmark/fix3/community_scores_real_only.csv')
    assert len(comm_df) == n_real, f"Expected 15,000 community scores, got {len(comm_df)}"
    community_score = comm_df['community_score'].values
    
    # 3. Isolation Forest on real works only
    feat_cols = ['sanction_amount', 'sanction_lag_days', 'work_status_code', 'work_category_code']
    X_real = df_clean[feat_cols].values
    iso = IsolationForest(n_estimators=100, random_state=42)
    iso.fit(X_real)
    iforest_score = -iso.decision_function(X_real)
    
    # 4. Robust Z-score on real works only
    log_amt = np.log1p(df_clean['sanction_amount'].values)
    lag = df_clean['sanction_lag_days'].values.astype(float)
    med_amt = np.median(log_amt)
    mad_amt = np.median(np.abs(log_amt - med_amt)) * 1.4826
    med_lag = np.median(lag)
    mad_lag = np.median(np.abs(lag - med_lag)) * 1.4826
    z_amt = np.abs(log_amt - med_amt) / (mad_amt + 1e-6)
    z_lag = np.abs(lag - med_lag) / (mad_lag + 1e-6)
    robust_z_score = np.maximum(z_amt, z_lag)
    
    # Save data/deploy/real_detector_scores.csv
    df_real_detectors = pd.DataFrame({
        'work_index': np.arange(n_real),
        'gat_frozen': gat_frozen,
        'community_score': community_score,
        'iforest_score': iforest_score,
        'robust_z_score': robust_z_score
    })
    deploy_det_path = os.path.join('data', 'deploy', 'real_detector_scores.csv')
    df_real_detectors.to_csv(deploy_det_path, index=False)
    print(f"Saved real-only detector scores: {deploy_det_path} ({len(df_real_detectors)} rows).")
    
    # Compare with detector_scores_v2.csv real-work rows (first 15,000 rows)
    df_v2 = pd.read_csv('data/benchmark/detector_scores_v2.csv')
    v2_real = df_v2.iloc[:n_real]
    
    sp_gat, _ = spearmanr(gat_frozen, v2_real['gat_frozen'])
    sp_comm, _ = spearmanr(community_score, v2_real['community_score'])
    sp_iforest, _ = spearmanr(iforest_score, v2_real['iforest_score'])
    sp_z, _ = spearmanr(robust_z_score, v2_real['robust_z_score'])
    
    print("\nSpearman Correlation: Real-Only Detectors vs detector_scores_v2.csv (Real Works):")
    print(f"  - gat_frozen:      {sp_gat:.6f}  (Expected ~1.0: Exact match)")
    print(f"  - robust_z_score:  {sp_z:.6f}  (Expected ~1.0: Virtually identical median/MAD)")
    print(f"  - community_score: {sp_comm:.6f}  (Differs: Injections removed from co-occurrence graph)")
    print(f"  - iforest_score:   {sp_iforest:.6f}  (Differs: Injections removed from unsupervised feature space)")
    
    # =========================================================================
    # STEP 2: FUSION MODEL TRAINING ON TRAIN+VAL BENCHMARK
    # =========================================================================
    print("\n--- STEP 2: FUSION MODEL TRAINING ON TRAIN+VAL BENCHMARK ---")
    df_bench_works = pd.read_csv('data/benchmark/benchmark_works.csv')
    df_bench_scores = pd.read_csv('data/benchmark/detector_scores_v2.csv')
    
    # 16 features exactly as in ml/run_fusion_loto.py
    amt_b = np.log1p(df_bench_works['sanction_amount'].values)
    lag_b = df_bench_works['sanction_lag_days'].values
    
    # Status and category dummies
    status_cols = [f'status_{c}' for c in sorted(df_bench_works['work_status_code'].unique())]
    cat_cols = [f'cat_{c}' for c in sorted(df_bench_works['work_category_code'].unique())]
    status_dummies_b = pd.get_dummies(df_bench_works['work_status_code'], prefix='status', drop_first=False).values.astype(float)
    cat_dummies_b = pd.get_dummies(df_bench_works['work_category_code'], prefix='cat', drop_first=False).values.astype(float)
    
    N_bench = len(df_bench_scores)
    gat_r_b = (rankdata(df_bench_scores['gat_frozen']) - 1.0) / (N_bench - 1.0)
    comm_r_b = (rankdata(df_bench_scores['community_score']) - 1.0) / (N_bench - 1.0)
    iforest_r_b = (rankdata(df_bench_scores['iforest_score']) - 1.0) / (N_bench - 1.0)
    z_r_b = (rankdata(df_bench_scores['robust_z_score']) - 1.0) / (N_bench - 1.0)
    
    feature_names = (
        ['log1p_amount', 'sanction_lag_days'] +
        status_cols +
        cat_cols +
        ['gat_frozen_percentile_rank', 'community_score_percentile_rank',
         'iforest_score_percentile_rank', 'robust_z_score_percentile_rank']
    )
    assert len(feature_names) == 16, f"Expected 16 features, got {len(feature_names)}"
    
    tab_feats_b = np.column_stack([amt_b, lag_b, status_dummies_b, cat_dummies_b])
    det_ranks_b = np.column_stack([gat_r_b, comm_r_b, iforest_r_b, z_r_b])
    all_feats_b = np.column_stack([tab_feats_b, det_ranks_b])
    y_bench = (df_bench_works['is_synthetic_fraud'] == True).astype(int).values
    
    # Train mask: train + val split (test split untouched)
    train_mask = df_bench_works['split'].isin(['train', 'val'])
    X_train = all_feats_b[train_mask]
    y_train = y_bench[train_mask]
    
    class_counts = pd.Series(y_train).value_counts().to_dict()
    print("Training Class Counts (Train + Val Works):")
    print(f"  - Negative (Real Works):            {class_counts.get(0, 0)}")
    print(f"  - Positive (Synthetic Injections):  {class_counts.get(1, 0)}")
    print(f"  - Total Training Samples:           {len(y_train)}")
    print(f"  - Test Split Untouched:             {(~train_mask).sum()} works")
    
    print("\n16 Feature Names in Fixed Order:")
    for i, fn in enumerate(feature_names, 1):
        print(f"  {i:2d}. {fn}")
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    clf = HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=100,
        learning_rate=0.1,
        class_weight='balanced',
        random_state=0
    )
    clf.fit(X_train_scaled, y_train)
    print("\nHistGradientBoostingClassifier successfully trained.")
    
    # Save model with joblib
    model_save_path = os.path.join('ml', 'fusion_model.joblib')
    joblib.dump(clf, model_save_path)
    print(f"Saved fusion model: {model_save_path}")
    
    # Save scaler and feature order metadata
    meta = {
        'feature_names': feature_names,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'train_class_counts': {int(k): int(v) for k, v in class_counts.items()},
        'train_total': int(len(y_train)),
        'model_type': 'HistGradientBoostingClassifier',
        'model_params': {
            'max_depth': 3,
            'max_iter': 100,
            'learning_rate': 0.1,
            'class_weight': 'balanced',
            'random_state': 0
        }
    }
    meta_save_path = os.path.join('ml', 'fusion_meta.json')
    with open(meta_save_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved fusion metadata: {meta_save_path}")
    
    # =========================================================================
    # STEP 3: SCORE THE 15,000 REAL WORKS
    # =========================================================================
    print("\n--- STEP 3: SCORE THE 15,000 REAL WORKS ---")
    amt_r = np.log1p(df_clean['sanction_amount'].values)
    lag_r = df_clean['sanction_lag_days'].values
    status_dummies_r = pd.get_dummies(df_clean['work_status_code'], prefix='status', drop_first=False).values.astype(float)
    cat_dummies_r = pd.get_dummies(df_clean['work_category_code'], prefix='cat', drop_first=False).values.astype(float)
    
    # Percentile ranks strictly over the 15,000 real works
    gat_r_r = (rankdata(gat_frozen) - 1.0) / (n_real - 1.0)
    comm_r_r = (rankdata(community_score) - 1.0) / (n_real - 1.0)
    iforest_r_r = (rankdata(iforest_score) - 1.0) / (n_real - 1.0)
    z_r_r = (rankdata(robust_z_score) - 1.0) / (n_real - 1.0)
    
    tab_feats_r = np.column_stack([amt_r, lag_r, status_dummies_r, cat_dummies_r])
    det_ranks_r = np.column_stack([gat_r_r, comm_r_r, iforest_r_r, z_r_r])
    all_feats_r = np.column_stack([tab_feats_r, det_ranks_r])
    
    X_real_scaled = scaler.transform(all_feats_r)
    fusion_scores = clf.predict_proba(X_real_scaled)[:, 1]
    
    # Save data/deploy/real_fusion_scores.csv
    df_fusion_scores = pd.DataFrame({
        'work_index': np.arange(n_real),
        'fusion_score': fusion_scores
    })
    deploy_fus_path = os.path.join('data', 'deploy', 'real_fusion_scores.csv')
    df_fusion_scores.to_csv(deploy_fus_path, index=False)
    print(f"Saved real fusion scores: {deploy_fus_path} ({len(df_fusion_scores)} rows).")
    
    # Print score distribution
    print("\nReal Fusion Score Distribution Summary:")
    print(f"  Min:    {fusion_scores.min():.6f}")
    print(f"  P05:    {np.percentile(fusion_scores, 5):.6f}")
    print(f"  P25:    {np.percentile(fusion_scores, 25):.6f}")
    print(f"  Median: {np.median(fusion_scores):.6f}")
    print(f"  Mean:   {fusion_scores.mean():.6f} (Std: {fusion_scores.std():.6f})")
    print(f"  P75:    {np.percentile(fusion_scores, 75):.6f}")
    print(f"  P90:    {np.percentile(fusion_scores, 90):.6f}")
    print(f"  P95:    {np.percentile(fusion_scores, 95):.6f}")
    print(f"  P99:    {np.percentile(fusion_scores, 99):.6f}")
    print(f"  Max:    {fusion_scores.max():.6f}")
    
    # Top 20 rows
    top_20_idx = np.argsort(-fusion_scores)[:20]
    top_20_df = pd.DataFrame({
        'rank': np.arange(1, 21),
        'work_index': top_20_idx,
        'MP': df_clean.iloc[top_20_idx]['MP'].values,
        'IDA': df_clean.iloc[top_20_idx]['IDA'].values,
        'amount': df_clean.iloc[top_20_idx]['sanction_amount'].values,
        'lag': df_clean.iloc[top_20_idx]['sanction_lag_days'].values,
        'status': df_clean.iloc[top_20_idx]['Work Status'].values,
        'fusion_score': fusion_scores[top_20_idx]
    })
    print("\nTop 20 Real Works Ranked by Fusion Score:")
    print(top_20_df.to_string(index=False))
    
    # Top 50 overlap between Fusion and GAT
    top_50_fusion = set(np.argsort(-fusion_scores)[:50])
    top_50_gat = set(np.argsort(-gat_frozen)[:50])
    overlap_count = len(top_50_fusion.intersection(top_50_gat))
    print(f"\nTop-50 Overlap: Fusion Top-50 vs GAT Top-50: {overlap_count}/50 ({overlap_count / 50 * 100:.1f}%)")
    
    # =========================================================================
    # STEP 5: DESCRIPTIVE REAL-LABEL SANITY CHECK (BIASED)
    # =========================================================================
    print("\n--- STEP 5: DESCRIPTIVE REAL-LABEL SANITY CHECK (BIASED) ---")
    labels_df = pd.read_csv('ml/anomaly_labels.csv')
    print("Read ml/anomaly_labels.csv (read-only):")
    total_labels = len(labels_df)
    counts = labels_df['label'].value_counts().to_dict()
    print(f"  Total Rows: {total_labels}")
    print(f"  Breakdown:  s (suspicious) = {counts.get('s', 0)}, n (normal) = {counts.get('n', 0)}, skip (excluded) = {counts.get('skip', 0)}")
    
    valid_mask = labels_df['label'].isin(['s', 'n'])
    valid_df = labels_df[valid_mask].copy()
    y_eval = (valid_df['label'] == 's').astype(int).values
    eval_indices = valid_df['work_index'].values
    
    n_s = int((y_eval == 1).sum())
    n_n = int((y_eval == 0).sum())
    assert n_s == 28 and n_n == 23, f"Expected 28 s / 23 n, got {n_s} s / {n_n} n"
    print(f"  Evaluated on: {len(y_eval)} valid labeled works ({n_s} 's' vs {n_n} 'n')")
    
    # AUROC of fusion, community, and GAT
    auc_gat = float(roc_auc_score(y_eval, gat_frozen[eval_indices]))
    auc_comm = float(roc_auc_score(y_eval, community_score[eval_indices]))
    auc_fusion = float(roc_auc_score(y_eval, fusion_scores[eval_indices]))
    
    print("\nDescriptive AUROC on 28s / 23n Analyst Labels:")
    print(f"  - GAT:             {auc_gat:.4f}")
    print(f"  - Community:       {auc_comm:.4f}")
    print(f"  - Learned Fusion:  {auc_fusion:.4f}")
    print("\n[IMPORTANT BIAS STATEMENT]:")
    print("  These 51 real-work labels were purposively drawn from the top percentiles of the GAT score")
    print("  during initial exploratory triage (selective audit protocol). As a result, GAT AUROC is")
    print("  strongly elevated by construction due to sampling bias (verification bias / selection on score).")
    print("  Community scorer and Fusion scores were NOT used to select these works for labeling.")
    
    print("\n" + "=" * 90)
    print("STEP 1, 2, 3 & 5 COMPLETED SUCCESSFULLY.")
    print("=" * 90)


if __name__ == '__main__':
    main()
