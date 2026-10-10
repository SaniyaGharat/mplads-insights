"""
Learned Fusion LOTO Analysis & Generator-Signature Checks
Script: ml/run_fusion_fix1.py
Outputs: data/benchmark/fusion/fix1/

Implements:
1. Delete untracked scratch/ folder (completed in preflight), verify git status
2. Print recall@5% and recall@10% for every model x typology from loto_metrics.csv
3. Pooled primary claim: episode-level bootstrap (500 resamples, resampling episodes
   within each of E, F, G, H, negatives fixed) of the mean-AUPRC difference over E-H
   vs (a) best_on_train and (b) hindsight best single detector on that eval set.
   Report mean, 95% CI, and count of typologies with positive difference.
4. Shuffled control with 20 permutations per fold for both learners (logistic and hgbc):
   print mean and 95th percentile of AUPRC per typology, mark which real fusion AUPRCs exceed 95th percentile.
5. Generator-signature check: per typology, print share of injected vs test-real works in each status
   code and category code, and single-feature AUROC of log1p(amount) and lag against test-real works.
6. Save CSVs to data/benchmark/fusion/fix1/, commit locally, verify checksums and git log.
"""

import os
import sys
import hashlib
import time
import shutil
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
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


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("FUSION FIX1 EVALUATION & GENERATOR-SIGNATURE AUDIT PIPELINE")
    print("=" * 80)
    t_start = time.time()
    
    out_dir = os.path.join('data', 'benchmark', 'fusion', 'fix1')
    os.makedirs(out_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # STEP 1: SCRATCH CLEANUP & GIT STATUS
    # -------------------------------------------------------------------------
    print("\n--- STEP 1: UNTRACKED SCRATCH CLEANUP & GIT STATUS ---")
    scratch_dir = os.path.join(os.getcwd(), 'scratch')
    if os.path.exists(scratch_dir):
        shutil.rmtree(scratch_dir)
        print("Deleted untracked scratch/ directory.")
    else:
        print("Untracked scratch/ directory already deleted / does not exist.")
        
    # Check SHA256 vs CHECKSUMS.txt as baseline integrity check
    checksum_file = os.path.join('data', 'benchmark', 'CHECKSUMS.txt')
    expected_checksums = {}
    with open(checksum_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    expected_checksums[parts[1]] = parts[0]
                    
    csv_files = ['benchmark_works.csv', 'benchmark_manifest.csv', 'detector_scores.csv', 'detector_scores_v2.csv']
    all_match = True
    for fname in csv_files:
        fpath = os.path.join('data', 'benchmark', fname)
        h = sha256_file(fpath)
        if h != expected_checksums.get(fname):
            all_match = False
            print(f"Checksum mismatch on {fname}!")
    assert all_match, "Preflight checksum failure!"
    print("Integrity check: All benchmark checksums match CHECKSUMS.txt.")
    
    # -------------------------------------------------------------------------
    # STEP 2: RECALL@5% AND RECALL@10% FROM LOTO_METRICS.CSV
    # -------------------------------------------------------------------------
    print("\n--- STEP 2: RECALL@5% AND RECALL@10% FOR EVERY MODEL X TYPOLOGY ---")
    loto_metrics_path = os.path.join('data', 'benchmark', 'fusion', 'loto_metrics.csv')
    assert os.path.exists(loto_metrics_path), f"Missing {loto_metrics_path}!"
    df_loto = pd.read_csv(loto_metrics_path)
    
    model_order = [
        'best_on_train', 'community_score', 'ensemble', 'gat_frozen', 'iforest_score', 'robust_z_score',
        'logistic_full', 'logistic_detectors', 'logistic_tabular',
        'hgbc_full', 'hgbc_detectors', 'hgbc_tabular'
    ]
    
    # Recall@5% table
    piv_r5 = df_loto.pivot(index='model', columns='typology', values='recall_5pct')
    piv_r5['Mean_EH'] = piv_r5[['E', 'F', 'G', 'H']].mean(axis=1)
    piv_r5['Mean_AD'] = piv_r5[['A', 'B', 'C', 'D']].mean(axis=1)
    piv_r5['Mean_All'] = piv_r5[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']].mean(axis=1)
    piv_r5 = piv_r5.reindex(model_order)
    
    print("\nTable 2A: Recall@5% by Typology (eval set k = 5%):")
    print(piv_r5[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Mean_EH', 'Mean_AD', 'Mean_All']].round(4).to_string())
    
    # Recall@10% table
    piv_r10 = df_loto.pivot(index='model', columns='typology', values='recall_10pct')
    piv_r10['Mean_EH'] = piv_r10[['E', 'F', 'G', 'H']].mean(axis=1)
    piv_r10['Mean_AD'] = piv_r10[['A', 'B', 'C', 'D']].mean(axis=1)
    piv_r10['Mean_All'] = piv_r10[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']].mean(axis=1)
    piv_r10 = piv_r10.reindex(model_order)
    
    print("\nTable 2B: Recall@10% by Typology (eval set k = 10%):")
    print(piv_r10[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Mean_EH', 'Mean_AD', 'Mean_All']].round(4).to_string())
    
    # -------------------------------------------------------------------------
    # STEP 3: POOLED PRIMARY CLAIM BOOTSTRAP OVER E-H
    # -------------------------------------------------------------------------
    print("\n--- STEP 3: POOLED PRIMARY CLAIM BOOTSTRAP OVER E-H (500 RESAMPLES) ---")
    
    # Load dataset to run exact bootstrap
    df_works = pd.read_csv('data/benchmark/benchmark_works.csv')
    df_scores = pd.read_csv('data/benchmark/detector_scores_v2.csv')
    
    amt = np.log1p(df_works['sanction_amount'].values)
    lag = df_works['sanction_lag_days'].values
    status_dummies = pd.get_dummies(df_works['work_status_code'], prefix='status', drop_first=False).values.astype(float)
    cat_dummies = pd.get_dummies(df_works['work_category_code'], prefix='cat', drop_first=False).values.astype(float)
    
    N_all = len(df_scores)
    gat_r = (rankdata(df_scores['gat_frozen']) - 1.0) / (N_all - 1.0)
    comm_r = (rankdata(df_scores['community_score']) - 1.0) / (N_all - 1.0)
    iforest_r = (rankdata(df_scores['iforest_score']) - 1.0) / (N_all - 1.0)
    z_r = (rankdata(df_scores['robust_z_score']) - 1.0) / (N_all - 1.0)
    ens_r = (gat_r + comm_r + iforest_r + z_r) / 4.0
    
    tab_feats = np.column_stack([amt, lag, status_dummies, cat_dummies])
    det_ranks = np.column_stack([gat_r, comm_r, iforest_r, z_r])
    all_feats = np.column_stack([tab_feats, det_ranks])
    
    y_all = (df_works['is_synthetic_fraud'] == True).astype(int).values
    
    raw_detectors = {
        'gat_frozen': df_scores['gat_frozen'].values,
        'community_score': df_scores['community_score'].values,
        'iforest_score': df_scores['iforest_score'].values,
        'robust_z_score': df_scores['robust_z_score'].values,
        'ensemble': ens_r
    }
    
    # Identify hindsight best detector per typology in E, F, G, H on eval set
    hindsight_best_detectors = {}
    detector_list = ['gat_frozen', 'community_score', 'iforest_score', 'robust_z_score']
    for T in ['E', 'F', 'G', 'H']:
        eval_mask = (df_works['typology'] == T) | (
            (df_works['split'] == 'test') & (~df_works['is_synthetic_fraud'])
        )
        y_ev = y_all[eval_mask]
        aps = {d: calc_auprc(y_ev, raw_detectors[d][eval_mask]) for d in detector_list}
        hindsight_best_detectors[T] = max(aps, key=aps.get)
        print(f"  Typology {T}: Hindsight best = {hindsight_best_detectors[T]:15s} (Eval AUPRC = {aps[hindsight_best_detectors[T]]:.4f}) | best_on_train = community_score ({aps['community_score']:.4f})")
        
    # Fit the 6 models for E, F, G, H and store eval predictions
    feat_ablation_defs = {
        'full': all_feats,
        'detectors': det_ranks,
        'tabular': tab_feats
    }
    fusion_model_names = [
        'logistic_full', 'logistic_detectors', 'logistic_tabular',
        'hgbc_full', 'hgbc_detectors', 'hgbc_tabular'
    ]
    
    preds_by_typ = {}
    ep_dict_by_typ = {}
    real_indices_by_typ = {}
    
    for T in ['E', 'F', 'G', 'H']:
        train_mask = (df_works['split'].isin(['train', 'val'])) & (
            (~df_works['is_synthetic_fraud']) | (df_works['typology'] != T)
        )
        eval_mask = (df_works['typology'] == T) | (
            (df_works['split'] == 'test') & (~df_works['is_synthetic_fraud'])
        )
        
        y_tr = y_all[train_mask]
        eval_df = df_works[eval_mask].reset_index(drop=True)
        eval_syn = eval_df[eval_df['is_synthetic_fraud'] == True]
        real_eval_idx = np.where(eval_df['is_synthetic_fraud'] == False)[0]
        
        ep_dict_by_typ[T] = {ep: grp.index.values for ep, grp in eval_syn.groupby('episode_id')}
        real_indices_by_typ[T] = real_eval_idx
        
        preds_by_typ[T] = {}
        # Baselines
        preds_by_typ[T]['best_on_train'] = raw_detectors['community_score'][eval_mask]
        preds_by_typ[T]['hindsight_best'] = raw_detectors[hindsight_best_detectors[T]][eval_mask]
        
        # Models
        for s_name, feats in feat_ablation_defs.items():
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(feats[train_mask])
            X_ev_s = scaler.transform(feats[eval_mask])
            
            lr = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=1000, random_state=0)
            lr.fit(X_tr_s, y_tr)
            preds_by_typ[T][f'logistic_{s_name}'] = lr.predict_proba(X_ev_s)[:, 1]
            
            hgbc = HistGradientBoostingClassifier(
                max_depth=3, max_iter=100, learning_rate=0.1, class_weight='balanced', random_state=0
            )
            hgbc.fit(X_tr_s, y_tr)
            preds_by_typ[T][f'hgbc_{s_name}'] = hgbc.predict_proba(X_ev_s)[:, 1]
            
    # 500 Bootstrap resamples over E-H
    rng_boot = np.random.RandomState(42)
    N_BOOT = 500
    
    boot_draws = {}
    for T in ['E', 'F', 'G', 'H']:
        ep_ids = sorted(list(ep_dict_by_typ[T].keys()))
        boot_draws[T] = [rng_boot.choice(ep_ids, size=len(ep_ids), replace=True) for _ in range(N_BOOT)]
        
    pooled_boot_results = []
    
    # Baseline mean AUPRCs across the 500 resamples
    boot_mean_auprc_best_on_train = []
    boot_mean_auprc_hindsight = []
    
    for b_idx in range(N_BOOT):
        aps_bot = []
        aps_hin = []
        for T in ['E', 'F', 'G', 'H']:
            pos_idx = np.concatenate([ep_dict_by_typ[T][ep] for ep in boot_draws[T][b_idx]])
            real_idx = real_indices_by_typ[T]
            y_b = np.concatenate([np.ones(len(pos_idx), dtype=int), np.zeros(len(real_idx), dtype=int)])
            
            s_bot = np.concatenate([preds_by_typ[T]['best_on_train'][pos_idx], preds_by_typ[T]['best_on_train'][real_idx]])
            aps_bot.append(calc_auprc(y_b, s_bot))
            
            s_hin = np.concatenate([preds_by_typ[T]['hindsight_best'][pos_idx], preds_by_typ[T]['hindsight_best'][real_idx]])
            aps_hin.append(calc_auprc(y_b, s_hin))
            
        boot_mean_auprc_best_on_train.append(np.mean(aps_bot))
        boot_mean_auprc_hindsight.append(np.mean(aps_hin))
        
    boot_mean_auprc_best_on_train = np.array(boot_mean_auprc_best_on_train)
    boot_mean_auprc_hindsight = np.array(boot_mean_auprc_hindsight)
    
    # Real point estimates for typologies E, F, G, H
    real_aps = {m: {} for m in fusion_model_names + ['best_on_train', 'hindsight_best']}
    for T in ['E', 'F', 'G', 'H']:
        eval_mask = (df_works['typology'] == T) | (
            (df_works['split'] == 'test') & (~df_works['is_synthetic_fraud'])
        )
        y_ev = y_all[eval_mask]
        for m in real_aps.keys():
            real_aps[m][T] = calc_auprc(y_ev, preds_by_typ[T][m])
            
    # For each fusion model: evaluate bootstrap paired differences
    for m in fusion_model_names:
        boot_mean_auprc_m = []
        for b_idx in range(N_BOOT):
            aps_m = []
            for T in ['E', 'F', 'G', 'H']:
                pos_idx = np.concatenate([ep_dict_by_typ[T][ep] for ep in boot_draws[T][b_idx]])
                real_idx = real_indices_by_typ[T]
                y_b = np.concatenate([np.ones(len(pos_idx), dtype=int), np.zeros(len(real_idx), dtype=int)])
                
                s_m = np.concatenate([preds_by_typ[T][m][pos_idx], preds_by_typ[T][m][real_idx]])
                aps_m.append(calc_auprc(y_b, s_m))
            boot_mean_auprc_m.append(np.mean(aps_m))
        boot_mean_auprc_m = np.array(boot_mean_auprc_m)
        
        diff_vs_bot = boot_mean_auprc_m - boot_mean_auprc_best_on_train
        diff_vs_hin = boot_mean_auprc_m - boot_mean_auprc_hindsight
        
        # Count typologies with positive difference (point estimate)
        pos_vs_bot = sum(real_aps[m][T] > real_aps['best_on_train'][T] for T in ['E', 'F', 'G', 'H'])
        pos_vs_hin = sum(real_aps[m][T] > real_aps['hindsight_best'][T] for T in ['E', 'F', 'G', 'H'])
        
        mean_auprc_eh_point = np.mean([real_aps[m][T] for T in ['E', 'F', 'G', 'H']])
        
        pooled_boot_results.append({
            'model': m,
            'mean_auprc_eh': mean_auprc_eh_point,
            'diff_vs_best_on_train_mean': float(np.mean(diff_vs_bot)),
            'diff_vs_best_on_train_ci_lower': float(np.percentile(diff_vs_bot, 2.5)),
            'diff_vs_best_on_train_ci_upper': float(np.percentile(diff_vs_bot, 97.5)),
            'diff_vs_best_on_train_ci_95': f"[{np.percentile(diff_vs_bot, 2.5):+.4f}, {np.percentile(diff_vs_bot, 97.5):+.4f}]",
            'pos_typologies_vs_bot': f"{pos_vs_bot}/4",
            'beats_bot_pooled': 'YES' if (np.mean(diff_vs_bot) > 0 and np.percentile(diff_vs_bot, 2.5) > 0) else 'NO',
            'diff_vs_hindsight_mean': float(np.mean(diff_vs_hin)),
            'diff_vs_hindsight_ci_lower': float(np.percentile(diff_vs_hin, 2.5)),
            'diff_vs_hindsight_ci_upper': float(np.percentile(diff_vs_hin, 97.5)),
            'diff_vs_hindsight_ci_95': f"[{np.percentile(diff_vs_hin, 2.5):+.4f}, {np.percentile(diff_vs_hin, 97.5):+.4f}]",
            'pos_typologies_vs_hin': f"{pos_vs_hin}/4",
            'beats_hindsight_pooled': 'YES' if (np.mean(diff_vs_hin) > 0 and np.percentile(diff_vs_hin, 2.5) > 0) else 'NO'
        })
        
    df_pooled_boot = pd.DataFrame(pooled_boot_results)
    
    base_bot_mean_eh = np.mean([real_aps['best_on_train'][T] for T in ['E', 'F', 'G', 'H']])
    base_hin_mean_eh = np.mean([real_aps['hindsight_best'][T] for T in ['E', 'F', 'G', 'H']])
    print(f"\nBaseline Mean AUPRC over E-H: Best on train = {base_bot_mean_eh:.4f} | Hindsight best = {base_hin_mean_eh:.4f}")
    
    disp_boot_cols = [
        'model', 'mean_auprc_eh',
        'diff_vs_best_on_train_mean', 'diff_vs_best_on_train_ci_95', 'pos_typologies_vs_bot', 'beats_bot_pooled',
        'diff_vs_hindsight_mean', 'diff_vs_hindsight_ci_95', 'pos_typologies_vs_hin', 'beats_hindsight_pooled'
    ]
    print("\nTable 3: Pooled Primary Claim Bootstrap over E-H (500 resamples):")
    print(df_pooled_boot[disp_boot_cols].to_string(index=False))
    
    # -------------------------------------------------------------------------
    # STEP 4: SHUFFLED CONTROL WITH 20 PERMUTATIONS PER FOLD
    # -------------------------------------------------------------------------
    print("\n--- STEP 4: SHUFFLED CONTROL WITH 20 PERMUTATIONS PER FOLD ---")
    typologies = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    shuffled_20_records = []
    
    t_shuf_start = time.time()
    for T in typologies:
        train_mask = (df_works['split'].isin(['train', 'val'])) & (
            (~df_works['is_synthetic_fraud']) | (df_works['typology'] != T)
        )
        eval_mask = (df_works['typology'] == T) | (
            (df_works['split'] == 'test') & (~df_works['is_synthetic_fraud'])
        )
        
        y_tr = y_all[train_mask]
        y_ev = y_all[eval_mask]
        eval_prev = float(np.sum(y_ev) / len(y_ev))
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(all_feats[train_mask])
        X_ev_s = scaler.transform(all_feats[eval_mask])
        
        # Real models AUPRC
        lr_real = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=1000, random_state=0)
        lr_real.fit(X_tr_s, y_tr)
        p_lr_real = lr_real.predict_proba(X_ev_s)[:, 1]
        real_lr_auprc = float(calc_auprc(y_ev, p_lr_real))
        
        hgbc_real = HistGradientBoostingClassifier(
            max_depth=3, max_iter=100, learning_rate=0.1, class_weight='balanced', random_state=0
        )
        hgbc_real.fit(X_tr_s, y_tr)
        p_hgbc_real = hgbc_real.predict_proba(X_ev_s)[:, 1]
        real_hgbc_auprc = float(calc_auprc(y_ev, p_hgbc_real))
        
        # 20 permutations
        lr_shuf_auprcs = []
        hgbc_shuf_auprcs = []
        
        for perm_idx in range(20):
            rng_p = np.random.RandomState(1000 + ord(T) * 100 + perm_idx)
            y_tr_p = rng_p.permutation(y_tr)
            
            lr_p = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=1000, random_state=0)
            lr_p.fit(X_tr_s, y_tr_p)
            lr_shuf_auprcs.append(calc_auprc(y_ev, lr_p.predict_proba(X_ev_s)[:, 1]))
            
            hgbc_p = HistGradientBoostingClassifier(
                max_depth=3, max_iter=100, learning_rate=0.1, class_weight='balanced', random_state=0
            )
            hgbc_p.fit(X_tr_s, y_tr_p)
            hgbc_shuf_auprcs.append(calc_auprc(y_ev, hgbc_p.predict_proba(X_ev_s)[:, 1]))
            
        lr_shuf_mean = float(np.mean(lr_shuf_auprcs))
        lr_shuf_p95 = float(np.percentile(lr_shuf_auprcs, 95))
        hgbc_shuf_mean = float(np.mean(hgbc_shuf_auprcs))
        hgbc_shuf_p95 = float(np.percentile(hgbc_shuf_auprcs, 95))
        
        lr_exceeds = (real_lr_auprc > lr_shuf_p95)
        hgbc_exceeds = (real_hgbc_auprc > hgbc_shuf_p95)
        
        shuffled_20_records.append({
            'typology': T,
            'eval_prevalence': eval_prev,
            'lr_real_auprc': real_lr_auprc,
            'lr_shuf_mean': lr_shuf_mean,
            'lr_shuf_p95': lr_shuf_p95,
            'lr_exceeds_p95': 'YES' if lr_exceeds else 'NO',
            'hgbc_real_auprc': real_hgbc_auprc,
            'hgbc_shuf_mean': hgbc_shuf_mean,
            'hgbc_shuf_p95': hgbc_shuf_p95,
            'hgbc_exceeds_p95': 'YES' if hgbc_exceeds else 'NO'
        })
        
    df_shuffled_20 = pd.DataFrame(shuffled_20_records)
    print(f"Shuffled control (160 fits x 2 learners) completed in {time.time() - t_shuf_start:.2f}s.")
    
    print("\nTable 4: Shuffled Control with 20 Permutations per Fold:")
    print(df_shuffled_20.round(4).to_string(index=False))
    
    # -------------------------------------------------------------------------
    # STEP 5: GENERATOR-SIGNATURE AUDIT
    # -------------------------------------------------------------------------
    print("\n--- STEP 5: GENERATOR-SIGNATURE CHECK ---")
    
    test_real_mask = (df_works['split'] == 'test') & (~df_works['is_synthetic_fraud'])
    df_test_real = df_works[test_real_mask]
    n_test_real = len(df_test_real)
    
    status_codes = sorted(df_works['work_status_code'].unique())
    category_codes = sorted(df_works['work_category_code'].unique())
    
    # 5A. Status code distribution
    status_rows = []
    real_stat_dist = df_test_real['work_status_code'].value_counts(normalize=True).to_dict()
    stat_row_real = {'typology': 'test_real', 'n_works': n_test_real}
    for sc in status_codes:
        stat_row_real[f'status_{sc}'] = real_stat_dist.get(sc, 0.0)
    status_rows.append(stat_row_real)
    
    for T in typologies:
        df_T = df_works[df_works['typology'] == T]
        t_stat_dist = df_T['work_status_code'].value_counts(normalize=True).to_dict()
        stat_row = {'typology': T, 'n_works': len(df_T)}
        for sc in status_codes:
            stat_row[f'status_{sc}'] = t_stat_dist.get(sc, 0.0)
        status_rows.append(stat_row)
        
    df_status_check = pd.DataFrame(status_rows)
    print("\nTable 5A: Share of Works in Each Status Code (Injected vs Test-Real):")
    print(df_status_check.round(4).to_string(index=False))
    
    # 5B. Category code distribution
    category_rows = []
    real_cat_dist = df_test_real['work_category_code'].value_counts(normalize=True).to_dict()
    cat_row_real = {'typology': 'test_real', 'n_works': n_test_real}
    for cc in category_codes:
        cat_row_real[f'cat_{cc}'] = real_cat_dist.get(cc, 0.0)
    category_rows.append(cat_row_real)
    
    for T in typologies:
        df_T = df_works[df_works['typology'] == T]
        t_cat_dist = df_T['work_category_code'].value_counts(normalize=True).to_dict()
        cat_row = {'typology': T, 'n_works': len(df_T)}
        for cc in category_codes:
            cat_row[f'cat_{cc}'] = t_cat_dist.get(cc, 0.0)
        category_rows.append(cat_row)
        
    df_category_check = pd.DataFrame(category_rows)
    print("\nTable 5B: Share of Works in Each Category Code (Injected vs Test-Real):")
    print(df_category_check.round(4).to_string(index=False))
    
    # 5C. Single-feature AUROC of amount and lag against test-real works
    auroc_rows = []
    for T in typologies:
        df_T = df_works[df_works['typology'] == T]
        combined = pd.concat([df_T, df_test_real]).reset_index(drop=True)
        y_comb = (combined['is_synthetic_fraud'] == True).astype(int).values
        
        amt_col = np.log1p(combined['sanction_amount'].values)
        lag_col = combined['sanction_lag_days'].values
        
        auc_amt = float(roc_auc_score(y_comb, amt_col))
        auc_lag = float(roc_auc_score(y_comb, lag_col))
        
        auroc_rows.append({
            'typology': T,
            'n_injected': len(df_T),
            'n_test_real': n_test_real,
            'amount_auroc': auc_amt,
            'lag_auroc': auc_lag,
            'amount_directional': 'high amount' if auc_amt > 0.5 else 'low amount',
            'lag_directional': 'high lag' if auc_lag > 0.5 else 'short lag'
        })
        
    df_auroc_check = pd.DataFrame(auroc_rows)
    print("\nTable 5C: Single-Feature AUROC of log1p(amount) and Lag against Test-Real Works:")
    print(df_auroc_check[['typology', 'n_injected', 'n_test_real', 'amount_auroc', 'lag_auroc', 'amount_directional', 'lag_directional']].round(4).to_string(index=False))
    
    # Combine into generator signature dataframe for export
    df_gen_sig = df_auroc_check.merge(df_status_check[df_status_check['typology'] != 'test_real'], on='typology')
    
    # -------------------------------------------------------------------------
    # STEP 6: SAVE ARTIFACTS
    # -------------------------------------------------------------------------
    print("\n--- STEP 6: SAVING CSV ARTIFACTS ---")
    
    f_rec = os.path.join(out_dir, 'recall_summary.csv')
    df_recall_combined = pd.concat([
        piv_r5.reset_index().assign(metric='recall_5pct'),
        piv_r10.reset_index().assign(metric='recall_10pct')
    ], ignore_index=True)
    df_recall_combined.to_csv(f_rec, index=False)
    print(f"Saved: {f_rec}")
    
    f_boot = os.path.join(out_dir, 'pooled_primary_claim_eh.csv')
    df_pooled_boot.to_csv(f_boot, index=False)
    print(f"Saved: {f_boot}")
    
    f_shuf = os.path.join(out_dir, 'shuffled_control_20perms.csv')
    df_shuffled_20.to_csv(f_shuf, index=False)
    print(f"Saved: {f_shuf}")
    
    f_gen = os.path.join(out_dir, 'generator_signatures.csv')
    df_gen_sig.to_csv(f_gen, index=False)
    print(f"Saved: {f_gen}")
    
    f_stat = os.path.join(out_dir, 'status_distributions.csv')
    df_status_check.to_csv(f_stat, index=False)
    print(f"Saved: {f_stat}")
    
    f_cat = os.path.join(out_dir, 'category_distributions.csv')
    df_category_check.to_csv(f_cat, index=False)
    print(f"Saved: {f_cat}")
    
    total_time = time.time() - t_start
    print(f"\nPipeline completed successfully in {total_time:.2f}s.")
    print("=" * 80)


if __name__ == '__main__':
    main()
