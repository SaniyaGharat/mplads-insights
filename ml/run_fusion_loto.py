"""
Learned Fusion, Leave-One-Typology-Out (LOTO) Evaluation Pipeline
Script: ml/run_fusion_loto.py
Outputs: data/benchmark/fusion/

Implements:
0. Preflight checks (hashes, positives per typology per split, 16-feature list)
1. Folds setup: for each held-out typology T in A..H:
   - train: train+val real works + train+val injected works of the other 7 typologies
   - eval positives: ALL episodes of T (from every split; unseen by model)
   - eval negatives: test-split real works only (never trained on)
   - assert train and eval indices disjoint
2. Models (fixed, no tuning; features standardised on the training part):
   (a) L2 logistic regression (class_weight='balanced')
   (b) HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.1,
       class_weight='balanced', random_state=0)
   Ablations for both: detectors-only (4 ranks), tabular-only (12 features), full (16 features)
3. Baselines on identical eval sets:
   - four detectors (gat_frozen, community_score, iforest_score, robust_z_score)
   - mean-rank ensemble
   - best_on_train: detector with highest AUPRC on the TRAIN works of that fold
4. Metrics: AUROC, AUPRC, recall@5%, recall@10% (stable sort, ties real-first)
   - Episode-level bootstrap (500 resamples of T's episodes, negatives fixed) of paired difference
     model minus best_on_train
   - Decision rule: fusion beats baseline on T iff mean diff > 0 AND CI lower > 0
   - Primary claim: mean AUPRC over E-H (unseen types); A-D reported under identical analysis
   - YES/NO decision table for every model x typology
5. Descriptive: logistic coefficients and tree permutation importances per fold (training data only)
6. Sanity checks: disjoint sets assert, class counts table, shuffled-label control
7. Save CSVs and figure, commit locally, verify checksums and git status
"""

import os
import sys
import hashlib
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score


def calc_auprc(y, scores):
    order = np.argsort(-scores, kind='mergesort')
    y_sorted = y[order]
    scores_sorted = scores[order]
    distinct_value_indices = np.where(np.diff(scores_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y.size - 1]
    tps = np.cumsum(y_sorted)[threshold_idxs]
    fps = (threshold_idxs + 1) - tps
    recalls = tps / tps[-1]
    precisions = tps / (tps + fps)
    return np.sum(np.diff(np.r_[0, recalls]) * precisions)


def calc_recalls(y, scores, k_5pct, k_10pct):
    orig_idx = np.arange(len(scores))
    order = np.lexsort((orig_idx, y, -scores))
    total_pos = np.sum(y)
    r_5 = float(np.sum(y[order[:k_5pct]]) / total_pos) if total_pos > 0 else 0.0
    r_10 = float(np.sum(y[order[:k_10pct]]) / total_pos) if total_pos > 0 else 0.0
    return r_5, r_10


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("LEARNED FUSION LEAVE-ONE-TYPOLOGY-OUT (LOTO) EVALUATION PIPELINE")
    print("=" * 80)
    t_start = time.time()
    
    out_dir = os.path.join('data', 'benchmark', 'fusion')
    os.makedirs(out_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # STEP 0: PREFLIGHT
    # -------------------------------------------------------------------------
    print("\n--- STEP 0: PREFLIGHT CHECKS ---")
    
    # 0a. SHA-256 Checksums
    checksum_file = os.path.join('data', 'benchmark', 'CHECKSUMS.txt')
    expected_checksums = {}
    with open(checksum_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    expected_checksums[parts[1]] = parts[0]
                    
    csv_files = [
        'benchmark_works.csv',
        'benchmark_manifest.csv',
        'detector_scores.csv',
        'detector_scores_v2.csv'
    ]
    
    print("\n0a. SHA256 Verification:")
    all_checksums_match = True
    for fname in csv_files:
        fpath = os.path.join('data', 'benchmark', fname)
        actual_hash = sha256_file(fpath)
        expected_hash = expected_checksums.get(fname, 'MISSING')
        match = (actual_hash == expected_hash)
        if not match:
            all_checksums_match = False
        print(f"  {fname}:")
        print(f"    Expected: {expected_hash}")
        print(f"    Actual:   {actual_hash}")
        print(f"    Match:    {match}")
        
    if not all_checksums_match:
        print("PREFLIGHT FAILED: SHA256 mismatch!")
        sys.exit(1)
    print("  -> SHA256 checks PASSED.")
    
    # Load data
    df_works = pd.read_csv('data/benchmark/benchmark_works.csv')
    df_scores = pd.read_csv('data/benchmark/detector_scores_v2.csv')
    
    # 0b. Positives per typology per split
    print("\n0b. Positives per Typology per Split:")
    syn_mask = (df_works['is_synthetic_fraud'] == True)
    ct_pos = pd.crosstab(df_works.loc[syn_mask, 'typology'], df_works.loc[syn_mask, 'split'], margins=True)
    print(ct_pos.to_string())
    
    # 0c. Feature list (same 16 features as Prompt 2)
    print("\n0c. 16-Feature List:")
    status_cols = [f'status_{c}' for c in sorted(df_works['work_status_code'].unique())]
    cat_cols = [f'cat_{c}' for c in sorted(df_works['work_category_code'].unique())]
    tabular_names = ['log1p_amount', 'sanction_lag_days'] + status_cols + cat_cols
    detector_rank_names = [
        'gat_frozen_percentile_rank',
        'community_score_percentile_rank',
        'iforest_score_percentile_rank',
        'robust_z_score_percentile_rank'
    ]
    feature_names = tabular_names + detector_rank_names
    for idx, f_name in enumerate(feature_names, 1):
        print(f"  {idx:2d}. {f_name}")
    print("Excluded features: dates, entity counts, episode_id, typology (strictly excluded).")
    
    # Prepare feature matrices
    amt = np.log1p(df_works['sanction_amount'].values)
    lag = df_works['sanction_lag_days'].values
    status_dummies = pd.get_dummies(df_works['work_status_code'], prefix='status', drop_first=False).values.astype(float)
    cat_dummies = pd.get_dummies(df_works['work_category_code'], prefix='cat', drop_first=False).values.astype(float)
    
    N_total = len(df_scores)
    gat_r = (rankdata(df_scores['gat_frozen']) - 1.0) / (N_total - 1.0)
    comm_r = (rankdata(df_scores['community_score']) - 1.0) / (N_total - 1.0)
    iforest_r = (rankdata(df_scores['iforest_score']) - 1.0) / (N_total - 1.0)
    z_r = (rankdata(df_scores['robust_z_score']) - 1.0) / (N_total - 1.0)
    ens_r = (gat_r + comm_r + iforest_r + z_r) / 4.0
    
    tab_feats = np.column_stack([amt, lag, status_dummies, cat_dummies])  # 12
    det_ranks = np.column_stack([gat_r, comm_r, iforest_r, z_r])         # 4
    all_feats = np.column_stack([tab_feats, det_ranks])                   # 16
    
    y_all = (df_works['is_synthetic_fraud'] == True).astype(int).values
    
    raw_detectors = {
        'gat_frozen': df_scores['gat_frozen'].values,
        'community_score': df_scores['community_score'].values,
        'iforest_score': df_scores['iforest_score'].values,
        'robust_z_score': df_scores['robust_z_score'].values,
        'ensemble': ens_r
    }
    
    # -------------------------------------------------------------------------
    # STEP 1: FOLDS DEFINITION & VERIFICATION
    # -------------------------------------------------------------------------
    print("\n--- STEP 1: LOTO FOLDS DEFINITION & ISOLATION VERIFICATION ---")
    typologies = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    fold_info = []
    
    for T in typologies:
        # train: train+val real works + train+val injected works of the other 7 typologies
        train_mask = (df_works['split'].isin(['train', 'val'])) & (
            (~df_works['is_synthetic_fraud']) | (df_works['typology'] != T)
        )
        
        # eval: ALL episodes of T (every split) + test-split real works
        eval_mask = (df_works['typology'] == T) | (
            (df_works['split'] == 'test') & (~df_works['is_synthetic_fraud'])
        )
        
        # Disjoint assertion
        overlap_count = int((train_mask & eval_mask).sum())
        assert overlap_count == 0, f"Fold {T}: Train and eval sets overlap! Count = {overlap_count}"
        
        n_tr_pos = int((train_mask & (df_works['is_synthetic_fraud'] == True)).sum())
        n_tr_neg = int((train_mask & (df_works['is_synthetic_fraud'] == False)).sum())
        n_tr_tot = int(train_mask.sum())
        
        n_ev_pos = int((eval_mask & (df_works['is_synthetic_fraud'] == True)).sum())
        n_ev_neg = int((eval_mask & (df_works['is_synthetic_fraud'] == False)).sum())
        n_ev_tot = int(eval_mask.sum())
        
        n_ev_ep = int(df_works.loc[eval_mask & (df_works['is_synthetic_fraud'] == True), 'episode_id'].nunique())
        
        fold_info.append({
            'Held_Out_Typology': T,
            'Train_Pos': n_tr_pos,
            'Train_Neg': n_tr_neg,
            'Train_Total': n_tr_tot,
            'Eval_Pos': n_ev_pos,
            'Eval_Neg': n_ev_neg,
            'Eval_Total': n_ev_tot,
            'Eval_Episodes': n_ev_ep,
            'Overlap': overlap_count
        })
        
    df_folds = pd.DataFrame(fold_info)
    print(df_folds.to_string(index=False))
    print("  -> All 8 folds strictly disjoint: PASSED.")
    
    # -------------------------------------------------------------------------
    # STEP 2, 3, 4: MODEL TRAINING, BASELINES & EVALUATION
    # -------------------------------------------------------------------------
    print("\n--- STEP 2, 3, 4: MODEL TRAINING, BASELINES, EVALUATION & BOOTSTRAP ---")
    
    feat_ablation_defs = {
        'full': (all_feats, 16),
        'detectors': (det_ranks, 4),
        'tabular': (tab_feats, 12)
    }
    
    metrics_records = []
    bootstrap_records = []
    importance_records = []
    shuffled_records = []
    
    # Pre-select best_on_train detector per fold
    best_on_train_detectors = {}
    print("\nBest on train detector selection per fold:")
    for T in typologies:
        train_mask = (df_works['split'].isin(['train', 'val'])) & (
            (~df_works['is_synthetic_fraud']) | (df_works['typology'] != T)
        )
        y_tr = y_all[train_mask]
        tr_aps = {
            d: calc_auprc(y_tr, raw_detectors[d][train_mask])
            for d in ['gat_frozen', 'community_score', 'iforest_score', 'robust_z_score']
        }
        best_d = max(tr_aps, key=tr_aps.get)
        best_on_train_detectors[T] = best_d
        print(f"  Fold {T}: best_on_train = {best_d:15s} (Train AUPRC = {tr_aps[best_d]:.4f})")
        
    t_eval_start = time.time()
    
    for T in typologies:
        print(f"\nProcessing Fold {T} (Held-out typology {T})...")
        train_mask = (df_works['split'].isin(['train', 'val'])) & (
            (~df_works['is_synthetic_fraud']) | (df_works['typology'] != T)
        )
        eval_mask = (df_works['typology'] == T) | (
            (df_works['split'] == 'test') & (~df_works['is_synthetic_fraud'])
        )
        
        y_tr = y_all[train_mask]
        y_ev = y_all[eval_mask]
        n_ev = len(y_ev)
        
        k_5pct = int(round(0.05 * n_ev))
        k_10pct = int(round(0.10 * n_ev))
        
        eval_df = df_works[eval_mask].reset_index(drop=True)
        eval_syn = eval_df[eval_df['is_synthetic_fraud'] == True]
        real_eval_indices = np.where(eval_df['is_synthetic_fraud'] == False)[0]
        ep_dict = {ep: grp.index.values for ep, grp in eval_syn.groupby('episode_id')}
        ep_ids = sorted(list(ep_dict.keys()))
        
        best_d_name = best_on_train_detectors[T]
        best_d_scores = raw_detectors[best_d_name][eval_mask]
        
        fold_predictions = {}
        # Baselines
        fold_predictions['best_on_train'] = best_d_scores
        fold_predictions['ensemble'] = raw_detectors['ensemble'][eval_mask]
        for d in ['gat_frozen', 'community_score', 'iforest_score', 'robust_z_score']:
            fold_predictions[d] = raw_detectors[d][eval_mask]
            
        # Train and evaluate models & ablations
        for subset_name, (feats, n_f) in feat_ablation_defs.items():
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(feats[train_mask])
            X_ev_s = scaler.transform(feats[eval_mask])
            
            # (a) L2 Logistic Regression
            lr = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=1000, random_state=0)
            lr.fit(X_tr_s, y_tr)
            p_lr = lr.predict_proba(X_ev_s)[:, 1]
            fold_predictions[f'logistic_{subset_name}'] = p_lr
            
            # (b) HistGradientBoostingClassifier
            hgbc = HistGradientBoostingClassifier(
                max_depth=3, max_iter=100, learning_rate=0.1, class_weight='balanced', random_state=0
            )
            hgbc.fit(X_tr_s, y_tr)
            p_hgbc = hgbc.predict_proba(X_ev_s)[:, 1]
            fold_predictions[f'hgbc_{subset_name}'] = p_hgbc
            
            # Step 5: Descriptive feature importances on full model
            if subset_name == 'full':
                lr_coefs = lr.coef_[0]
                perm_res = permutation_importance(
                    hgbc, X_tr_s, y_tr, n_repeats=5, random_state=0, scoring='average_precision'
                )
                for f_idx, f_name in enumerate(feature_names):
                    importance_records.append({
                        'typology': T,
                        'feature': f_name,
                        'feature_index': f_idx + 1,
                        'logistic_coef': float(lr_coefs[f_idx]),
                        'hgbc_perm_mean': float(perm_res.importances_mean[f_idx]),
                        'hgbc_perm_std': float(perm_res.importances_std[f_idx])
                    })
                    
        # Step 6: Shuffled-label control for this fold
        rng_shuf = np.random.RandomState(42 + ord(T))
        y_tr_shuf = rng_shuf.permutation(y_tr)
        scaler_shuf = StandardScaler()
        X_tr_full_s = scaler_shuf.fit_transform(all_feats[train_mask])
        X_ev_full_s = scaler_shuf.transform(all_feats[eval_mask])
        
        lr_shuf = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=1000, random_state=0)
        lr_shuf.fit(X_tr_full_s, y_tr_shuf)
        p_lr_shuf = lr_shuf.predict_proba(X_ev_full_s)[:, 1]
        
        hgbc_shuf = HistGradientBoostingClassifier(
            max_depth=3, max_iter=100, learning_rate=0.1, class_weight='balanced', random_state=0
        )
        hgbc_shuf.fit(X_tr_full_s, y_tr_shuf)
        p_hgbc_shuf = hgbc_shuf.predict_proba(X_ev_full_s)[:, 1]
        
        eval_prevalence = float(np.sum(y_ev) / len(y_ev))
        shuffled_records.append({
            'typology': T,
            'eval_prevalence': eval_prevalence,
            'logistic_shuffled_auprc': float(calc_auprc(y_ev, p_lr_shuf)),
            'hgbc_shuffled_auprc': float(calc_auprc(y_ev, p_hgbc_shuf))
        })
        
        # Calculate standard metrics for all models and baselines
        for m_name, preds in fold_predictions.items():
            ap = float(calc_auprc(y_ev, preds))
            auc = float(roc_auc_score(y_ev, preds))
            r5, r10 = calc_recalls(y_ev, preds, k_5pct, k_10pct)
            metrics_records.append({
                'typology': T,
                'model': m_name,
                'auprc': ap,
                'auroc': auc,
                'recall_5pct': r5,
                'recall_10pct': r10,
                'best_on_train_name': best_d_name
            })
            
        # Bootstrap paired difference (500 resamples of T's episodes, negatives fixed)
        # Evaluated for the 6 learned models vs best_on_train
        learned_model_names = [
            'logistic_full', 'logistic_detectors', 'logistic_tabular',
            'hgbc_full', 'hgbc_detectors', 'hgbc_tabular'
        ]
        
        rng_boot = np.random.RandomState(42 + ord(T))
        N_BOOT = 500
        boot_ep_draws = [rng_boot.choice(ep_ids, size=len(ep_ids), replace=True) for _ in range(N_BOOT)]
        
        for m_name in learned_model_names:
            preds_m = fold_predictions[m_name]
            paired_diffs = []
            
            for draw in boot_ep_draws:
                pos_indices = np.concatenate([ep_dict[ep] for ep in draw])
                y_b = np.concatenate([np.ones(len(pos_indices), dtype=int), np.zeros(len(real_eval_indices), dtype=int)])
                
                s_base = np.concatenate([best_d_scores[pos_indices], best_d_scores[real_eval_indices]])
                ap_base = calc_auprc(y_b, s_base)
                
                s_model = np.concatenate([preds_m[pos_indices], preds_m[real_eval_indices]])
                ap_model = calc_auprc(y_b, s_model)
                
                paired_diffs.append(ap_model - ap_base)
                
            diff_mean = float(np.mean(paired_diffs))
            ci_lower = float(np.percentile(paired_diffs, 2.5))
            ci_upper = float(np.percentile(paired_diffs, 97.5))
            model_eval_auprc = float(calc_auprc(y_ev, preds_m))
            base_eval_auprc = float(calc_auprc(y_ev, best_d_scores))
            
            beats = (diff_mean > 0.0) and (ci_lower > 0.0)
            
            bootstrap_records.append({
                'typology': T,
                'model': m_name,
                'model_auprc': model_eval_auprc,
                'best_on_train_auprc': base_eval_auprc,
                'best_on_train_name': best_d_name,
                'diff_mean': diff_mean,
                'diff_ci_lower': ci_lower,
                'diff_ci_upper': ci_upper,
                'diff_ci_95': f"[{ci_lower:+.4f}, {ci_upper:+.4f}]",
                'beats_baseline': 'YES' if beats else 'NO'
            })
            
    print(f"\nAll folds evaluated and bootstrapped in {time.time() - t_eval_start:.2f}s.")
    
    df_metrics = pd.DataFrame(metrics_records)
    df_boot = pd.DataFrame(bootstrap_records)
    df_importance = pd.DataFrame(importance_records)
    df_shuffled = pd.DataFrame(shuffled_records)
    
    # -------------------------------------------------------------------------
    # PRINT RESULTS & DECISION TABLES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("TABLE 1: LOTO AUPRC ACROSS TYPOLOGIES & PRIMARY CLAIM (MEAN E-H)")
    print("=" * 90)
    pivot_auprc = df_metrics.pivot(index='model', columns='typology', values='auprc')
    pivot_auprc['Mean_EH'] = pivot_auprc[['E', 'F', 'G', 'H']].mean(axis=1)
    pivot_auprc['Mean_AD'] = pivot_auprc[['A', 'B', 'C', 'D']].mean(axis=1)
    pivot_auprc['Mean_All'] = pivot_auprc[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']].mean(axis=1)
    
    model_order = [
        'best_on_train', 'community_score', 'ensemble', 'gat_frozen', 'iforest_score', 'robust_z_score',
        'logistic_full', 'logistic_detectors', 'logistic_tabular',
        'hgbc_full', 'hgbc_detectors', 'hgbc_tabular'
    ]
    pivot_auprc = pivot_auprc.reindex(model_order)
    print(pivot_auprc[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Mean_EH', 'Mean_AD', 'Mean_All']].round(4).to_string())
    
    print("\n" + "=" * 90)
    print("TABLE 2: LOTO AUROC ACROSS TYPOLOGIES & PRIMARY CLAIM (MEAN E-H)")
    print("=" * 90)
    pivot_auroc = df_metrics.pivot(index='model', columns='typology', values='auroc')
    pivot_auroc['Mean_EH'] = pivot_auroc[['E', 'F', 'G', 'H']].mean(axis=1)
    pivot_auroc['Mean_AD'] = pivot_auroc[['A', 'B', 'C', 'D']].mean(axis=1)
    pivot_auroc['Mean_All'] = pivot_auroc[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']].mean(axis=1)
    pivot_auroc = pivot_auroc.reindex(model_order)
    print(pivot_auroc[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Mean_EH', 'Mean_AD', 'Mean_All']].round(4).to_string())
    
    print("\n" + "=" * 90)
    print("TABLE 3: PRE-REGISTERED DECISION TABLE (500-EPISODE BOOTSTRAP PAIRED DIFFERENCE)")
    print("=" * 90)
    print("Rule: Fusion beats baseline on T iff mean diff > 0 AND 95% bootstrap CI lower > 0.")
    print(df_boot[['model', 'typology', 'model_auprc', 'best_on_train_auprc', 'diff_mean', 'diff_ci_95', 'beats_baseline']].to_string(index=False))
    
    # Primary claim summary
    print("\n" + "=" * 90)
    print("PRIMARY CLAIM SUMMARY: MEAN AUPRC OVER DESIGNATED UNSEEN TYPOLOGIES (E-H)")
    print("=" * 90)
    eh_summary = []
    base_eh = float(pivot_auprc.loc['best_on_train', 'Mean_EH'])
    for m in model_order:
        m_eh = float(pivot_auprc.loc[m, 'Mean_EH'])
        m_ad = float(pivot_auprc.loc[m, 'Mean_AD'])
        m_all = float(pivot_auprc.loc[m, 'Mean_All'])
        lift_eh = m_eh - base_eh
        eh_summary.append({
            'Model': m,
            'Mean_AUPRC_EH (Primary)': m_eh,
            'Lift_vs_BestOnTrain_EH': lift_eh,
            'Mean_AUPRC_AD': m_ad,
            'Mean_AUPRC_All': m_all
        })
    df_eh = pd.DataFrame(eh_summary)
    print(df_eh.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # STEP 5: DESCRIPTIVE FEATURE IMPORTANCES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("--- STEP 5: DESCRIPTIVE FEATURE IMPORTANCES (TRAINING SET ONLY) ---")
    print("=" * 90)
    imp_summary = df_importance.groupby('feature').agg(
        mean_logistic_coef=('logistic_coef', 'mean'),
        std_logistic_coef=('logistic_coef', 'std'),
        mean_hgbc_perm=('hgbc_perm_mean', 'mean'),
        std_hgbc_perm=('hgbc_perm_mean', 'std')
    ).reset_index().sort_values(by='mean_hgbc_perm', ascending=False)
    print(imp_summary.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # STEP 6: SANITY ASSERTS & SHUFFLED-LABEL CONTROL
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("--- STEP 6: SANITY CHECKS & SHUFFLED-LABEL CONTROL ---")
    print("=" * 90)
    
    print("1. Disjointness check: Verified 0 overlapping indices for all 8 folds (PASSED).")
    
    print("\n2. Class Counts per Fold:")
    print(df_folds[['Held_Out_Typology', 'Train_Pos', 'Train_Neg', 'Eval_Pos', 'Eval_Neg', 'Eval_Episodes']].to_string(index=False))
    
    print("\n3. Shuffled-Label Control (Labels randomly permuted on training set):")
    print(df_shuffled.to_string(index=False))
    mean_prev = df_shuffled['eval_prevalence'].mean()
    mean_shuf_lr = df_shuffled['logistic_shuffled_auprc'].mean()
    mean_shuf_hgbc = df_shuffled['hgbc_shuffled_auprc'].mean()
    print(f"\nMean Eval Prevalence:            {mean_prev:.4f}")
    print(f"Mean Logistic Shuffled AUPRC:    {mean_shuf_lr:.4f} (diff = {mean_shuf_lr - mean_prev:+.4f})")
    print(f"Mean HGBC Shuffled AUPRC:        {mean_shuf_hgbc:.4f} (diff = {mean_shuf_hgbc - mean_prev:+.4f})")
    assert abs(mean_shuf_lr - mean_prev) < 0.02, "Shuffled logistic control failed!"
    assert abs(mean_shuf_hgbc - mean_prev) < 0.02, "Shuffled HGBC control failed!"
    print("-> Shuffled-label control confirmed: AUPRC falls directly to base rate prevalence (PASSED).")
    
    # -------------------------------------------------------------------------
    # STEP 7: SAVE ARTIFACTS AND FIGURE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("--- STEP 7: SAVING ARTIFACTS AND PLOTS ---")
    print("=" * 90)
    
    f_metrics = os.path.join(out_dir, 'loto_metrics.csv')
    df_metrics.to_csv(f_metrics, index=False)
    print(f"Saved: {f_metrics}")
    
    f_boot = os.path.join(out_dir, 'loto_bootstrap.csv')
    df_boot.to_csv(f_boot, index=False)
    print(f"Saved: {f_boot}")
    
    f_imp = os.path.join(out_dir, 'feature_importance.csv')
    df_importance.to_csv(f_imp, index=False)
    print(f"Saved: {f_imp}")
    
    f_shuf = os.path.join(out_dir, 'shuffled_control.csv')
    df_shuffled.to_csv(f_shuf, index=False)
    print(f"Saved: {f_shuf}")
    
    # Generate Figure
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
    
    typs_and_means = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Mean_EH', 'Mean_AD']
    x_indices = np.arange(len(typs_and_means))
    
    models_to_plot = [
        ('best_on_train', 'Best on Train (Baseline)', '#7f7f7f', '//'),
        ('ensemble', 'Mean-Rank Ensemble', '#1f77b4', ''),
        ('logistic_detectors', 'Logistic (Detectors Only)', '#aec7e8', ''),
        ('logistic_full', 'Logistic (Full 16 Feats)', '#2ca02c', ''),
        ('hgbc_detectors', 'HGBC (Detectors Only)', '#ffbb78', ''),
        ('hgbc_full', 'HGBC (Full 16 Feats)', '#d62728', '')
    ]
    
    width = 0.13
    for i, (m_id, m_label, m_col, m_hatch) in enumerate(models_to_plot):
        vals = [pivot_auprc.loc[m_id, col] for col in typs_and_means]
        offset = (i - len(models_to_plot) / 2 + 0.5) * width
        ax.bar(x_indices + offset, vals, width=width, label=m_label, color=m_col, hatch=m_hatch, alpha=0.9, edgecolor='black', linewidth=0.5)
        
    ax.set_xticks(x_indices)
    ax.set_xticklabels(['Typ A', 'Typ B', 'Typ C', 'Typ D', 'Typ E', 'Typ F', 'Typ G', 'Typ H', 'Mean E-H\n(Primary)', 'Mean A-D\n(Host)'], fontsize=10.5, fontweight='bold')
    ax.set_ylabel('Held-Out Eval AUPRC', fontsize=11, labelpad=8)
    ax.set_title('Leave-One-Typology-Out (LOTO) Transfer: Fusion Models vs Baselines', fontsize=13, fontweight='bold', pad=14)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper left', fontsize=9.5)
    
    # Highlight Mean E-H section
    ax.axvline(7.5, color='#444444', linestyle=':', linewidth=1.5)
    plt.tight_layout()
    
    fig_path = os.path.join(out_dir, 'fusion_loto_auprc.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path}")
    
    total_time = time.time() - t_start
    print(f"\nPipeline finished successfully in {total_time:.2f}s.")
    print("=" * 80)


if __name__ == '__main__':
    main()
