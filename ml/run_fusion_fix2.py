"""
Learned Fusion Protocols P1 (Leave-Family-Out) & P2 (Strict Unseen)
Script: ml/run_fusion_fix2.py
Outputs: data/benchmark/fusion/fix2/

Implements:
1. Protocol P1 (leave-family-out):
   Families: {A, D}, {B, E}, {C}, {F}, {G}, {H}.
   Train on other typologies' train+val injected works + train+val real works.
   Eval positives = all episodes of the held-out family; eval negatives = test-split real works.
   Assert train/eval disjointness; print sizes and episode counts.
2. Protocol P2 (strict unseen):
   Train on A, C, D only (+ train+val real).
   Eval each of B, E, F, G, H separately; primary claim = mean AUPRC over F, G, H.
3. For both protocols: AUROC, AUPRC, recall@5% and @10%; episode-level bootstrap
   (500 resamples) of the paired difference vs best_on_train and vs hindsight best;
   YES/NO for every cell; number of positive typologies.
4. Shuffled control with 20 permutations per fold for hgbc_full and logistic_full,
   printing mean and 95th percentile of AUPRC; mark every real AUPRC that exceeds it.
5. Cat_1-only restriction: drop the four category features and rerun P1 for hgbc_full
   and logistic_full. Print the change in AUPRC.
6. Save CSVs, commit locally, verify checksums and git log.
"""

import os
import sys
import hashlib
import time
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
    print("FUSION FIX2: PROTOCOLS P1 (LEAVE-FAMILY-OUT) & P2 (STRICT UNSEEN)")
    print("=" * 80)
    t_start = time.time()
    
    out_dir = os.path.join('data', 'benchmark', 'fusion', 'fix2')
    os.makedirs(out_dir, exist_ok=True)
    
    # Preflight Checksums
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
    print("Preflight: All benchmark checksums match CHECKSUMS.txt.")
    
    # Load dataset
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
    
    tab_feats = np.column_stack([amt, lag, status_dummies, cat_dummies])  # 12
    det_ranks = np.column_stack([gat_r, comm_r, iforest_r, z_r])         # 4
    all_feats = np.column_stack([tab_feats, det_ranks])                   # 16
    
    # No-cat features for Step 5 (drop 4 category features -> 12 features)
    no_cat_tab_feats = np.column_stack([amt, lag, status_dummies])       # 8
    no_cat_all_feats = np.column_stack([no_cat_tab_feats, det_ranks])     # 12
    
    y_all = (df_works['is_synthetic_fraud'] == True).astype(int).values
    
    raw_detectors = {
        'gat_frozen': df_scores['gat_frozen'].values,
        'community_score': df_scores['community_score'].values,
        'iforest_score': df_scores['iforest_score'].values,
        'robust_z_score': df_scores['robust_z_score'].values,
        'ensemble': ens_r
    }
    detector_list = ['gat_frozen', 'community_score', 'iforest_score', 'robust_z_score']
    
    model_order = [
        'best_on_train', 'hindsight_best', 'community_score', 'ensemble', 'gat_frozen', 'iforest_score', 'robust_z_score',
        'logistic_full', 'logistic_detectors', 'logistic_tabular',
        'hgbc_full', 'hgbc_detectors', 'hgbc_tabular'
    ]
    learned_model_names = [
        'logistic_full', 'logistic_detectors', 'logistic_tabular',
        'hgbc_full', 'hgbc_detectors', 'hgbc_tabular'
    ]
    
    # =========================================================================
    # STEP 1: PROTOCOL P1 (LEAVE-FAMILY-OUT) FOLDS SETUP
    # =========================================================================
    print("\n--- STEP 1: PROTOCOL P1 (LEAVE-FAMILY-OUT) SETUP ---")
    p1_families = [
        ('AD', ['A', 'D']),
        ('BE', ['B', 'E']),
        ('C', ['C']),
        ('F', ['F']),
        ('G', ['G']),
        ('H', ['H'])
    ]
    
    p1_fold_info = []
    p1_best_on_train = {}
    p1_hindsight_best = {}
    
    for fam_name, fam_members in p1_families:
        train_mask = (df_works['split'].isin(['train', 'val'])) & (
            (~df_works['is_synthetic_fraud']) | (~df_works['typology'].isin(fam_members))
        )
        eval_mask = (df_works['typology'].isin(fam_members)) | (
            (df_works['split'] == 'test') & (~df_works['is_synthetic_fraud'])
        )
        
        # Assert disjointness
        overlap = int((train_mask & eval_mask).sum())
        assert overlap == 0, f"P1 Family {fam_name}: Overlap detected!"
        
        n_tr_pos = int((train_mask & (df_works['is_synthetic_fraud'] == True)).sum())
        n_tr_neg = int((train_mask & (df_works['is_synthetic_fraud'] == False)).sum())
        n_tr_tot = int(train_mask.sum())
        
        n_ev_pos = int((eval_mask & (df_works['is_synthetic_fraud'] == True)).sum())
        n_ev_neg = int((eval_mask & (df_works['is_synthetic_fraud'] == False)).sum())
        n_ev_tot = int(eval_mask.sum())
        
        n_ev_ep = int(df_works.loc[eval_mask & (df_works['is_synthetic_fraud'] == True), 'episode_id'].nunique())
        
        # Best on train detector
        y_tr = y_all[train_mask]
        tr_aps = {d: calc_auprc(y_tr, raw_detectors[d][train_mask]) for d in detector_list}
        best_bot = max(tr_aps, key=tr_aps.get)
        p1_best_on_train[fam_name] = best_bot
        
        # Hindsight best detector
        y_ev = y_all[eval_mask]
        ev_aps = {d: calc_auprc(y_ev, raw_detectors[d][eval_mask]) for d in detector_list}
        best_hin = max(ev_aps, key=ev_aps.get)
        p1_hindsight_best[fam_name] = best_hin
        
        p1_fold_info.append({
            'Family': fam_name,
            'Members': '+'.join(fam_members),
            'Train_Pos': n_tr_pos,
            'Train_Neg': n_tr_neg,
            'Train_Total': n_tr_tot,
            'Eval_Pos': n_ev_pos,
            'Eval_Neg': n_ev_neg,
            'Eval_Total': n_ev_tot,
            'Eval_Episodes': n_ev_ep,
            'Best_On_Train': best_bot,
            'Hindsight_Best': best_hin,
            'Overlap': overlap
        })
        
    df_p1_folds = pd.DataFrame(p1_fold_info)
    print(df_p1_folds.to_string(index=False))
    print("  -> P1 train/eval sets strictly disjoint: PASSED.")
    
    # =========================================================================
    # STEP 2: PROTOCOL P2 (STRICT UNSEEN) SETUP
    # =========================================================================
    print("\n--- STEP 2: PROTOCOL P2 (STRICT UNSEEN) SETUP ---")
    p2_train_mask = (df_works['split'].isin(['train', 'val'])) & (
        (~df_works['is_synthetic_fraud']) | (df_works['typology'].isin(['A', 'C', 'D']))
    )
    p2_eval_targets = ['B', 'E', 'F', 'G', 'H']
    
    p2_fold_info = []
    y_tr_p2 = y_all[p2_train_mask]
    n_tr_p2_pos = int((p2_train_mask & (df_works['is_synthetic_fraud'] == True)).sum())
    n_tr_p2_neg = int((p2_train_mask & (df_works['is_synthetic_fraud'] == False)).sum())
    n_tr_p2_tot = int(p2_train_mask.sum())
    
    # Best on train for P2
    p2_tr_aps = {d: calc_auprc(y_tr_p2, raw_detectors[d][p2_train_mask]) for d in detector_list}
    p2_best_on_train = max(p2_tr_aps, key=p2_tr_aps.get)
    print(f"P2 Training Set (A, C, D + train/val real): Pos={n_tr_p2_pos}, Neg={n_tr_p2_neg}, Total={n_tr_p2_tot}")
    print(f"P2 best_on_train detector = {p2_best_on_train} (Train AUPRC = {p2_tr_aps[p2_best_on_train]:.4f})")
    
    p2_hindsight_best = {}
    for T in p2_eval_targets:
        eval_mask = (df_works['typology'] == T) | (
            (df_works['split'] == 'test') & (~df_works['is_synthetic_fraud'])
        )
        overlap = int((p2_train_mask & eval_mask).sum())
        assert overlap == 0, f"P2 Eval {T}: Overlap detected!"
        
        n_ev_pos = int((eval_mask & (df_works['is_synthetic_fraud'] == True)).sum())
        n_ev_neg = int((eval_mask & (df_works['is_synthetic_fraud'] == False)).sum())
        n_ev_tot = int(eval_mask.sum())
        n_ev_ep = int(df_works.loc[eval_mask & (df_works['is_synthetic_fraud'] == True), 'episode_id'].nunique())
        
        y_ev = y_all[eval_mask]
        ev_aps = {d: calc_auprc(y_ev, raw_detectors[d][eval_mask]) for d in detector_list}
        best_hin = max(ev_aps, key=ev_aps.get)
        p2_hindsight_best[T] = best_hin
        
        p2_fold_info.append({
            'Target': T,
            'Eval_Pos': n_ev_pos,
            'Eval_Neg': n_ev_neg,
            'Eval_Total': n_ev_tot,
            'Eval_Episodes': n_ev_ep,
            'Hindsight_Best': best_hin,
            'Overlap': overlap
        })
        
    df_p2_folds = pd.DataFrame(p2_fold_info)
    print(df_p2_folds.to_string(index=False))
    print("  -> P2 train/eval sets strictly disjoint: PASSED.")
    
    # =========================================================================
    # STEP 3: RUN PROTOCOLS P1 & P2 EVALUATION & BOOTSTRAP
    # =========================================================================
    print("\n--- STEP 3: EVALUATION & 500-EPISODE BOOTSTRAP (P1 & P2) ---")
    feat_ablation_defs = {
        'full': all_feats,
        'detectors': det_ranks,
        'tabular': tab_feats
    }
    
    # --- PROTOCOL P1 EXECUTION ---
    print("\nExecuting Protocol P1 (Leave-Family-Out)...")
    p1_metrics_records = []
    p1_boot_records = []
    p1_preds = {}
    p1_eval_info = {}
    
    N_BOOT = 500
    rng_boot = np.random.RandomState(42)
    
    for fam_name, fam_members in p1_families:
        train_mask = (df_works['split'].isin(['train', 'val'])) & (
            (~df_works['is_synthetic_fraud']) | (~df_works['typology'].isin(fam_members))
        )
        eval_mask = (df_works['typology'].isin(fam_members)) | (
            (df_works['split'] == 'test') & (~df_works['is_synthetic_fraud'])
        )
        
        y_tr = y_all[train_mask]
        y_ev = y_all[eval_mask]
        n_ev = len(y_ev)
        k_5pct = int(round(0.05 * n_ev))
        k_10pct = int(round(0.10 * n_ev))
        
        eval_df = df_works[eval_mask].reset_index(drop=True)
        eval_syn = eval_df[eval_df['is_synthetic_fraud'] == True]
        real_eval_idx = np.where(eval_df['is_synthetic_fraud'] == False)[0]
        ep_dict = {ep: grp.index.values for ep, grp in eval_syn.groupby('episode_id')}
        ep_ids = sorted(list(ep_dict.keys()))
        
        p1_eval_info[fam_name] = {'ep_dict': ep_dict, 'ep_ids': ep_ids, 'real_idx': real_eval_idx, 'y_ev': y_ev}
        
        bot_d = p1_best_on_train[fam_name]
        hin_d = p1_hindsight_best[fam_name]
        
        preds = {
            'best_on_train': raw_detectors[bot_d][eval_mask],
            'hindsight_best': raw_detectors[hin_d][eval_mask],
            'ensemble': raw_detectors['ensemble'][eval_mask]
        }
        for d in detector_list:
            preds[d] = raw_detectors[d][eval_mask]
            
        for s_name, feats in feat_ablation_defs.items():
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(feats[train_mask])
            X_ev_s = scaler.transform(feats[eval_mask])
            
            lr = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=1000, random_state=0)
            lr.fit(X_tr_s, y_tr)
            preds[f'logistic_{s_name}'] = lr.predict_proba(X_ev_s)[:, 1]
            
            hgbc = HistGradientBoostingClassifier(
                max_depth=3, max_iter=100, learning_rate=0.1, class_weight='balanced', random_state=0
            )
            hgbc.fit(X_tr_s, y_tr)
            preds[f'hgbc_{s_name}'] = hgbc.predict_proba(X_ev_s)[:, 1]
            
        p1_preds[fam_name] = preds
        
        for m_name in model_order:
            p = preds[m_name]
            ap = float(calc_auprc(y_ev, p))
            auc = float(roc_auc_score(y_ev, p))
            r5, r10 = calc_recalls(y_ev, p, k_5pct, k_10pct)
            p1_metrics_records.append({
                'protocol': 'P1',
                'family': fam_name,
                'model': m_name,
                'auprc': ap,
                'auroc': auc,
                'recall_5pct': r5,
                'recall_10pct': r10,
                'best_on_train_name': bot_d,
                'hindsight_best_name': hin_d
            })
            
        # Bootstrap per family (500 resamples of family episodes)
        boot_draws = [rng_boot.choice(ep_ids, size=len(ep_ids), replace=True) for _ in range(N_BOOT)]
        
        s_bot_eval = preds['best_on_train']
        s_hin_eval = preds['hindsight_best']
        base_bot_ap = calc_auprc(y_ev, s_bot_eval)
        base_hin_ap = calc_auprc(y_ev, s_hin_eval)
        
        for m_name in learned_model_names:
            s_m_eval = preds[m_name]
            m_ap = calc_auprc(y_ev, s_m_eval)
            diffs_bot = []
            diffs_hin = []
            
            for draw in boot_draws:
                pos_idx = np.concatenate([ep_dict[ep] for ep in draw])
                y_b = np.concatenate([np.ones(len(pos_idx), dtype=int), np.zeros(len(real_eval_idx), dtype=int)])
                
                ap_bot = calc_auprc(y_b, np.concatenate([s_bot_eval[pos_idx], s_bot_eval[real_eval_idx]]))
                ap_hin = calc_auprc(y_b, np.concatenate([s_hin_eval[pos_idx], s_hin_eval[real_eval_idx]]))
                ap_m = calc_auprc(y_b, np.concatenate([s_m_eval[pos_idx], s_m_eval[real_eval_idx]]))
                
                diffs_bot.append(ap_m - ap_bot)
                diffs_hin.append(ap_m - ap_hin)
                
            m_diff_bot = float(np.mean(diffs_bot))
            ci_low_bot = float(np.percentile(diffs_bot, 2.5))
            ci_hi_bot = float(np.percentile(diffs_bot, 97.5))
            beats_bot = (m_diff_bot > 0 and ci_low_bot > 0)
            
            m_diff_hin = float(np.mean(diffs_hin))
            ci_low_hin = float(np.percentile(diffs_hin, 2.5))
            ci_hi_hin = float(np.percentile(diffs_hin, 97.5))
            beats_hin = (m_diff_hin > 0 and ci_low_hin > 0)
            
            p1_boot_records.append({
                'protocol': 'P1',
                'family': fam_name,
                'model': m_name,
                'model_auprc': m_ap,
                'best_on_train_auprc': base_bot_ap,
                'diff_vs_bot_mean': m_diff_bot,
                'diff_vs_bot_ci_95': f"[{ci_low_bot:+.4f}, {ci_hi_bot:+.4f}]",
                'beats_bot': 'YES' if beats_bot else 'NO',
                'hindsight_auprc': base_hin_ap,
                'diff_vs_hin_mean': m_diff_hin,
                'diff_vs_hin_ci_95': f"[{ci_low_hin:+.4f}, {ci_hi_hin:+.4f}]",
                'beats_hin': 'YES' if beats_hin else 'NO'
            })
            
    df_p1_metrics = pd.DataFrame(p1_metrics_records)
    df_p1_boot = pd.DataFrame(p1_boot_records)
    
    # --- P1 POOLED BOOTSTRAP OVER ALL 6 FAMILIES ---
    print("\nComputing P1 Pooled Bootstrap over all 6 families (500 resamples)...")
    boot_draws_p1 = {}
    for fam_name, _ in p1_families:
        ep_ids = p1_eval_info[fam_name]['ep_ids']
        boot_draws_p1[fam_name] = [rng_boot.choice(ep_ids, size=len(ep_ids), replace=True) for _ in range(N_BOOT)]
        
    bot_p1_boots = []
    hin_p1_boots = []
    for b_idx in range(N_BOOT):
        bot_aps = []
        hin_aps = []
        for fam_name, _ in p1_families:
            ep_dict = p1_eval_info[fam_name]['ep_dict']
            real_idx = p1_eval_info[fam_name]['real_idx']
            pos_idx = np.concatenate([ep_dict[ep] for ep in boot_draws_p1[fam_name][b_idx]])
            y_b = np.concatenate([np.ones(len(pos_idx), dtype=int), np.zeros(len(real_idx), dtype=int)])
            s_bot = p1_preds[fam_name]['best_on_train']
            s_hin = p1_preds[fam_name]['hindsight_best']
            bot_aps.append(calc_auprc(y_b, np.concatenate([s_bot[pos_idx], s_bot[real_idx]])))
            hin_aps.append(calc_auprc(y_b, np.concatenate([s_hin[pos_idx], s_hin[real_idx]])))
        bot_p1_boots.append(np.mean(bot_aps))
        hin_p1_boots.append(np.mean(hin_aps))
    bot_p1_boots = np.array(bot_p1_boots)
    hin_p1_boots = np.array(hin_p1_boots)
    
    p1_pooled_results = []
    for m_name in learned_model_names:
        m_p1_boots = []
        for b_idx in range(N_BOOT):
            m_aps = []
            for fam_name, _ in p1_families:
                ep_dict = p1_eval_info[fam_name]['ep_dict']
                real_idx = p1_eval_info[fam_name]['real_idx']
                pos_idx = np.concatenate([ep_dict[ep] for ep in boot_draws_p1[fam_name][b_idx]])
                y_b = np.concatenate([np.ones(len(pos_idx), dtype=int), np.zeros(len(real_idx), dtype=int)])
                s_m = p1_preds[fam_name][m_name]
                m_aps.append(calc_auprc(y_b, np.concatenate([s_m[pos_idx], s_m[real_idx]])))
            m_p1_boots.append(np.mean(m_aps))
        m_p1_boots = np.array(m_p1_boots)
        
        diff_bot = m_p1_boots - bot_p1_boots
        diff_hin = m_p1_boots - hin_p1_boots
        
        real_aps_m = {fam: calc_auprc(p1_eval_info[fam]['y_ev'], p1_preds[fam][m_name]) for fam, _ in p1_families}
        real_aps_bot = {fam: calc_auprc(p1_eval_info[fam]['y_ev'], p1_preds[fam]['best_on_train']) for fam, _ in p1_families}
        real_aps_hin = {fam: calc_auprc(p1_eval_info[fam]['y_ev'], p1_preds[fam]['hindsight_best']) for fam, _ in p1_families}
        
        pos_fam_bot = sum(real_aps_m[fam] > real_aps_bot[fam] for fam, _ in p1_families)
        pos_fam_hin = sum(real_aps_m[fam] > real_aps_hin[fam] for fam, _ in p1_families)
        mean_p1_point = np.mean([real_aps_m[fam] for fam, _ in p1_families])
        
        p1_pooled_results.append({
            'model': m_name,
            'mean_auprc_p1': mean_p1_point,
            'diff_vs_bot_mean': float(np.mean(diff_bot)),
            'diff_vs_bot_ci_95': f"[{np.percentile(diff_bot, 2.5):+.4f}, {np.percentile(diff_bot, 97.5):+.4f}]",
            'pos_families_vs_bot': f"{pos_fam_bot}/6",
            'beats_bot_pooled': 'YES' if (np.mean(diff_bot) > 0 and np.percentile(diff_bot, 2.5) > 0) else 'NO',
            'diff_vs_hin_mean': float(np.mean(diff_hin)),
            'diff_vs_hin_ci_95': f"[{np.percentile(diff_hin, 2.5):+.4f}, {np.percentile(diff_hin, 97.5):+.4f}]",
            'pos_families_vs_hin': f"{pos_fam_hin}/6",
            'beats_hin_pooled': 'YES' if (np.mean(diff_hin) > 0 and np.percentile(diff_hin, 2.5) > 0) else 'NO'
        })
    df_p1_pooled_boot = pd.DataFrame(p1_pooled_results)
    
    # --- PROTOCOL P2 EXECUTION ---
    print("\nExecuting Protocol P2 (Strict Unseen: Train ACD, Eval B, E, F, G, H)...")
    p2_metrics_records = []
    p2_boot_records = []
    p2_preds = {}
    p2_eval_info = {}
    
    # Train P2 models once on train ACD
    p2_fitted_models = {}
    y_tr_p2 = y_all[p2_train_mask]
    for s_name, feats in feat_ablation_defs.items():
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(feats[p2_train_mask])
        
        lr = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=1000, random_state=0)
        lr.fit(X_tr_s, y_tr_p2)
        
        hgbc = HistGradientBoostingClassifier(
            max_depth=3, max_iter=100, learning_rate=0.1, class_weight='balanced', random_state=0
        )
        hgbc.fit(X_tr_s, y_tr_p2)
        p2_fitted_models[s_name] = {'scaler': scaler, 'lr': lr, 'hgbc': hgbc}
        
    for T in p2_eval_targets:
        eval_mask = (df_works['typology'] == T) | (
            (df_works['split'] == 'test') & (~df_works['is_synthetic_fraud'])
        )
        y_ev = y_all[eval_mask]
        n_ev = len(y_ev)
        k_5pct = int(round(0.05 * n_ev))
        k_10pct = int(round(0.10 * n_ev))
        
        eval_df = df_works[eval_mask].reset_index(drop=True)
        eval_syn = eval_df[eval_df['is_synthetic_fraud'] == True]
        real_eval_idx = np.where(eval_df['is_synthetic_fraud'] == False)[0]
        ep_dict = {ep: grp.index.values for ep, grp in eval_syn.groupby('episode_id')}
        ep_ids = sorted(list(ep_dict.keys()))
        
        p2_eval_info[T] = {'ep_dict': ep_dict, 'ep_ids': ep_ids, 'real_idx': real_eval_idx, 'y_ev': y_ev}
        
        hin_d = p2_hindsight_best[T]
        preds = {
            'best_on_train': raw_detectors[p2_best_on_train][eval_mask],
            'hindsight_best': raw_detectors[hin_d][eval_mask],
            'ensemble': raw_detectors['ensemble'][eval_mask]
        }
        for d in detector_list:
            preds[d] = raw_detectors[d][eval_mask]
            
        for s_name, feats in feat_ablation_defs.items():
            sc = p2_fitted_models[s_name]['scaler']
            X_ev_s = sc.transform(feats[eval_mask])
            preds[f'logistic_{s_name}'] = p2_fitted_models[s_name]['lr'].predict_proba(X_ev_s)[:, 1]
            preds[f'hgbc_{s_name}'] = p2_fitted_models[s_name]['hgbc'].predict_proba(X_ev_s)[:, 1]
            
        p2_preds[T] = preds
        
        for m_name in model_order:
            p = preds[m_name]
            ap = float(calc_auprc(y_ev, p))
            auc = float(roc_auc_score(y_ev, p))
            r5, r10 = calc_recalls(y_ev, p, k_5pct, k_10pct)
            p2_metrics_records.append({
                'protocol': 'P2',
                'typology': T,
                'model': m_name,
                'auprc': ap,
                'auroc': auc,
                'recall_5pct': r5,
                'recall_10pct': r10,
                'best_on_train_name': p2_best_on_train,
                'hindsight_best_name': hin_d
            })
            
        # Bootstrap per typology in P2
        boot_draws = [rng_boot.choice(ep_ids, size=len(ep_ids), replace=True) for _ in range(N_BOOT)]
        s_bot_eval = preds['best_on_train']
        s_hin_eval = preds['hindsight_best']
        base_bot_ap = calc_auprc(y_ev, s_bot_eval)
        base_hin_ap = calc_auprc(y_ev, s_hin_eval)
        
        for m_name in learned_model_names:
            s_m_eval = preds[m_name]
            m_ap = calc_auprc(y_ev, s_m_eval)
            diffs_bot = []
            diffs_hin = []
            
            for draw in boot_draws:
                pos_idx = np.concatenate([ep_dict[ep] for ep in draw])
                y_b = np.concatenate([np.ones(len(pos_idx), dtype=int), np.zeros(len(real_eval_idx), dtype=int)])
                
                ap_bot = calc_auprc(y_b, np.concatenate([s_bot_eval[pos_idx], s_bot_eval[real_eval_idx]]))
                ap_hin = calc_auprc(y_b, np.concatenate([s_hin_eval[pos_idx], s_hin_eval[real_eval_idx]]))
                ap_m = calc_auprc(y_b, np.concatenate([s_m_eval[pos_idx], s_m_eval[real_eval_idx]]))
                
                diffs_bot.append(ap_m - ap_bot)
                diffs_hin.append(ap_m - ap_hin)
                
            m_diff_bot = float(np.mean(diffs_bot))
            ci_low_bot = float(np.percentile(diffs_bot, 2.5))
            ci_hi_bot = float(np.percentile(diffs_bot, 97.5))
            beats_bot = (m_diff_bot > 0 and ci_low_bot > 0)
            
            m_diff_hin = float(np.mean(diffs_hin))
            ci_low_hin = float(np.percentile(diffs_hin, 2.5))
            ci_hi_hin = float(np.percentile(diffs_hin, 97.5))
            beats_hin = (m_diff_hin > 0 and ci_low_hin > 0)
            
            p2_boot_records.append({
                'protocol': 'P2',
                'typology': T,
                'model': m_name,
                'model_auprc': m_ap,
                'best_on_train_auprc': base_bot_ap,
                'diff_vs_bot_mean': m_diff_bot,
                'diff_vs_bot_ci_95': f"[{ci_low_bot:+.4f}, {ci_hi_bot:+.4f}]",
                'beats_bot': 'YES' if beats_bot else 'NO',
                'hindsight_auprc': base_hin_ap,
                'diff_vs_hin_mean': m_diff_hin,
                'diff_vs_hin_ci_95': f"[{ci_low_hin:+.4f}, {ci_hi_hin:+.4f}]",
                'beats_hin': 'YES' if beats_hin else 'NO'
            })
            
    df_p2_metrics = pd.DataFrame(p2_metrics_records)
    df_p2_boot = pd.DataFrame(p2_boot_records)
    
    # --- P2 POOLED PRIMARY CLAIM BOOTSTRAP OVER F, G, H ---
    print("\nComputing P2 Pooled Primary Claim over F, G, H (500 resamples)...")
    boot_draws_p2_fgh = {}
    for T in ['F', 'G', 'H']:
        ep_ids = p2_eval_info[T]['ep_ids']
        boot_draws_p2_fgh[T] = [rng_boot.choice(ep_ids, size=len(ep_ids), replace=True) for _ in range(N_BOOT)]
        
    p2_fgh_boot_results = []
    
    # Baselines across F, G, H in P2
    bot_fgh_boots = []
    hin_fgh_boots = []
    for b_idx in range(N_BOOT):
        bot_aps = []
        hin_aps = []
        for T in ['F', 'G', 'H']:
            ep_dict = p2_eval_info[T]['ep_dict']
            real_idx = p2_eval_info[T]['real_idx']
            pos_idx = np.concatenate([ep_dict[ep] for ep in boot_draws_p2_fgh[T][b_idx]])
            y_b = np.concatenate([np.ones(len(pos_idx), dtype=int), np.zeros(len(real_idx), dtype=int)])
            
            s_bot = p2_preds[T]['best_on_train']
            s_hin = p2_preds[T]['hindsight_best']
            bot_aps.append(calc_auprc(y_b, np.concatenate([s_bot[pos_idx], s_bot[real_idx]])))
            hin_aps.append(calc_auprc(y_b, np.concatenate([s_hin[pos_idx], s_hin[real_idx]])))
        bot_fgh_boots.append(np.mean(bot_aps))
        hin_fgh_boots.append(np.mean(hin_aps))
    bot_fgh_boots = np.array(bot_fgh_boots)
    hin_fgh_boots = np.array(hin_fgh_boots)
    
    for m_name in learned_model_names:
        m_fgh_boots = []
        for b_idx in range(N_BOOT):
            m_aps = []
            for T in ['F', 'G', 'H']:
                ep_dict = p2_eval_info[T]['ep_dict']
                real_idx = p2_eval_info[T]['real_idx']
                pos_idx = np.concatenate([ep_dict[ep] for ep in boot_draws_p2_fgh[T][b_idx]])
                y_b = np.concatenate([np.ones(len(pos_idx), dtype=int), np.zeros(len(real_idx), dtype=int)])
                
                s_m = p2_preds[T][m_name]
                m_aps.append(calc_auprc(y_b, np.concatenate([s_m[pos_idx], s_m[real_idx]])))
            m_fgh_boots.append(np.mean(m_aps))
        m_fgh_boots = np.array(m_fgh_boots)
        
        diff_bot = m_fgh_boots - bot_fgh_boots
        diff_hin = m_fgh_boots - hin_fgh_boots
        
        # Point estimates across F, G, H and across all 5 targets (B, E, F, G, H)
        real_aps_m = {T: calc_auprc(p2_eval_info[T]['y_ev'], p2_preds[T][m_name]) for T in p2_eval_targets}
        real_aps_bot = {T: calc_auprc(p2_eval_info[T]['y_ev'], p2_preds[T]['best_on_train']) for T in p2_eval_targets}
        real_aps_hin = {T: calc_auprc(p2_eval_info[T]['y_ev'], p2_preds[T]['hindsight_best']) for T in p2_eval_targets}
        
        pos_fgh_bot = sum(real_aps_m[T] > real_aps_bot[T] for T in ['F', 'G', 'H'])
        pos_fgh_hin = sum(real_aps_m[T] > real_aps_hin[T] for T in ['F', 'G', 'H'])
        pos_all_bot = sum(real_aps_m[T] > real_aps_bot[T] for T in p2_eval_targets)
        pos_all_hin = sum(real_aps_m[T] > real_aps_hin[T] for T in p2_eval_targets)
        
        mean_fgh_point = np.mean([real_aps_m[T] for T in ['F', 'G', 'H']])
        
        p2_fgh_boot_results.append({
            'model': m_name,
            'mean_auprc_fgh': mean_fgh_point,
            'diff_vs_bot_mean': float(np.mean(diff_bot)),
            'diff_vs_bot_ci_95': f"[{np.percentile(diff_bot, 2.5):+.4f}, {np.percentile(diff_bot, 97.5):+.4f}]",
            'pos_typ_fgh_vs_bot': f"{pos_fgh_bot}/3",
            'pos_typ_all_vs_bot': f"{pos_all_bot}/5",
            'beats_bot_fgh': 'YES' if (np.mean(diff_bot) > 0 and np.percentile(diff_bot, 2.5) > 0) else 'NO',
            'diff_vs_hin_mean': float(np.mean(diff_hin)),
            'diff_vs_hin_ci_95': f"[{np.percentile(diff_hin, 2.5):+.4f}, {np.percentile(diff_hin, 97.5):+.4f}]",
            'pos_typ_fgh_vs_hin': f"{pos_fgh_hin}/3",
            'pos_typ_all_vs_hin': f"{pos_all_hin}/5",
            'beats_hin_fgh': 'YES' if (np.mean(diff_hin) > 0 and np.percentile(diff_hin, 2.5) > 0) else 'NO'
        })
        
    df_p2_fgh_boot = pd.DataFrame(p2_fgh_boot_results)
    
    # -------------------------------------------------------------------------
    # PRINT RESULTS TABLES FOR P1 AND P2
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("TABLE P1-A1: PROTOCOL P1 (LEAVE-FAMILY-OUT) AUPRC BY FAMILY")
    print("=" * 90)
    piv_p1_auprc = df_p1_metrics.pivot(index='model', columns='family', values='auprc')
    piv_p1_auprc['Mean_P1'] = piv_p1_auprc.mean(axis=1)
    piv_p1_auprc = piv_p1_auprc.reindex(model_order)
    print(piv_p1_auprc[['AD', 'BE', 'C', 'F', 'G', 'H', 'Mean_P1']].round(4).to_string())
    
    print("\n" + "=" * 90)
    print("TABLE P1-A2: PROTOCOL P1 (LEAVE-FAMILY-OUT) AUROC BY FAMILY")
    print("=" * 90)
    piv_p1_auroc = df_p1_metrics.pivot(index='model', columns='family', values='auroc')
    piv_p1_auroc['Mean_P1'] = piv_p1_auroc.mean(axis=1)
    piv_p1_auroc = piv_p1_auroc.reindex(model_order)
    print(piv_p1_auroc[['AD', 'BE', 'C', 'F', 'G', 'H', 'Mean_P1']].round(4).to_string())
    
    print("\n" + "=" * 90)
    print("TABLE P1-A3: PROTOCOL P1 (LEAVE-FAMILY-OUT) RECALL@5% BY FAMILY")
    print("=" * 90)
    piv_p1_r5 = df_p1_metrics.pivot(index='model', columns='family', values='recall_5pct')
    piv_p1_r5['Mean_P1'] = piv_p1_r5.mean(axis=1)
    piv_p1_r5 = piv_p1_r5.reindex(model_order)
    print(piv_p1_r5[['AD', 'BE', 'C', 'F', 'G', 'H', 'Mean_P1']].round(4).to_string())
    
    print("\n" + "=" * 90)
    print("TABLE P1-A4: PROTOCOL P1 (LEAVE-FAMILY-OUT) RECALL@10% BY FAMILY")
    print("=" * 90)
    piv_p1_r10 = df_p1_metrics.pivot(index='model', columns='family', values='recall_10pct')
    piv_p1_r10['Mean_P1'] = piv_p1_r10.mean(axis=1)
    piv_p1_r10 = piv_p1_r10.reindex(model_order)
    print(piv_p1_r10[['AD', 'BE', 'C', 'F', 'G', 'H', 'Mean_P1']].round(4).to_string())
    
    print("\n" + "=" * 90)
    print("TABLE P1-B: PROTOCOL P1 CELL-BY-CELL BOOTSTRAP PAIRED DIFFERENCE (500 RESAMPLES, YES/NO FOR EVERY CELL)")
    print("=" * 90)
    disp_p1_boot_cols = [
        'family', 'model', 'model_auprc',
        'best_on_train_auprc', 'diff_vs_bot_mean', 'diff_vs_bot_ci_95', 'beats_bot',
        'hindsight_auprc', 'diff_vs_hin_mean', 'diff_vs_hin_ci_95', 'beats_hin'
    ]
    print(df_p1_boot[disp_p1_boot_cols].to_string(index=False))
    
    print("\n" + "=" * 90)
    print("TABLE P1-C: PROTOCOL P1 POOLED SUMMARY ACROSS 6 FAMILIES (NUMBER OF POSITIVE FAMILIES)")
    print("=" * 90)
    base_bot_mean_p1 = np.mean([calc_auprc(p1_eval_info[fam]['y_ev'], p1_preds[fam]['best_on_train']) for fam, _ in p1_families])
    base_hin_mean_p1 = np.mean([calc_auprc(p1_eval_info[fam]['y_ev'], p1_preds[fam]['hindsight_best']) for fam, _ in p1_families])
    print(f"P1 Baseline Mean AUPRC across 6 families: best_on_train = {base_bot_mean_p1:.4f} | hindsight_best = {base_hin_mean_p1:.4f}\n")
    print(df_p1_pooled_boot.to_string(index=False))
    
    print("\n" + "=" * 90)
    print("TABLE P2-A1: PROTOCOL P2 (STRICT UNSEEN: TRAIN ACD) AUPRC BY TARGET")
    print("=" * 90)
    piv_p2_auprc = df_p2_metrics.pivot(index='model', columns='typology', values='auprc')
    piv_p2_auprc['Mean_FGH (Primary)'] = piv_p2_auprc[['F', 'G', 'H']].mean(axis=1)
    piv_p2_auprc['Mean_All_P2'] = piv_p2_auprc[['B', 'E', 'F', 'G', 'H']].mean(axis=1)
    piv_p2_auprc = piv_p2_auprc.reindex(model_order)
    print(piv_p2_auprc[['B', 'E', 'F', 'G', 'H', 'Mean_FGH (Primary)', 'Mean_All_P2']].round(4).to_string())
    
    print("\n" + "=" * 90)
    print("TABLE P2-A2: PROTOCOL P2 (STRICT UNSEEN: TRAIN ACD) AUROC BY TARGET")
    print("=" * 90)
    piv_p2_auroc = df_p2_metrics.pivot(index='model', columns='typology', values='auroc')
    piv_p2_auroc['Mean_FGH (Primary)'] = piv_p2_auroc[['F', 'G', 'H']].mean(axis=1)
    piv_p2_auroc['Mean_All_P2'] = piv_p2_auroc[['B', 'E', 'F', 'G', 'H']].mean(axis=1)
    piv_p2_auroc = piv_p2_auroc.reindex(model_order)
    print(piv_p2_auroc[['B', 'E', 'F', 'G', 'H', 'Mean_FGH (Primary)', 'Mean_All_P2']].round(4).to_string())
    
    print("\n" + "=" * 90)
    print("TABLE P2-A3: PROTOCOL P2 (STRICT UNSEEN: TRAIN ACD) RECALL@5% BY TARGET")
    print("=" * 90)
    piv_p2_r5 = df_p2_metrics.pivot(index='model', columns='typology', values='recall_5pct')
    piv_p2_r5['Mean_FGH (Primary)'] = piv_p2_r5[['F', 'G', 'H']].mean(axis=1)
    piv_p2_r5['Mean_All_P2'] = piv_p2_r5[['B', 'E', 'F', 'G', 'H']].mean(axis=1)
    piv_p2_r5 = piv_p2_r5.reindex(model_order)
    print(piv_p2_r5[['B', 'E', 'F', 'G', 'H', 'Mean_FGH (Primary)', 'Mean_All_P2']].round(4).to_string())
    
    print("\n" + "=" * 90)
    print("TABLE P2-A4: PROTOCOL P2 (STRICT UNSEEN: TRAIN ACD) RECALL@10% BY TARGET")
    print("=" * 90)
    piv_p2_r10 = df_p2_metrics.pivot(index='model', columns='typology', values='recall_10pct')
    piv_p2_r10['Mean_FGH (Primary)'] = piv_p2_r10[['F', 'G', 'H']].mean(axis=1)
    piv_p2_r10['Mean_All_P2'] = piv_p2_r10[['B', 'E', 'F', 'G', 'H']].mean(axis=1)
    piv_p2_r10 = piv_p2_r10.reindex(model_order)
    print(piv_p2_r10[['B', 'E', 'F', 'G', 'H', 'Mean_FGH (Primary)', 'Mean_All_P2']].round(4).to_string())
    
    print("\n" + "=" * 90)
    print("TABLE P2-B: PROTOCOL P2 CELL-BY-CELL BOOTSTRAP PAIRED DIFFERENCE (500 RESAMPLES, YES/NO FOR EVERY CELL)")
    print("=" * 90)
    disp_p2_boot_cols = [
        'typology', 'model', 'model_auprc',
        'best_on_train_auprc', 'diff_vs_bot_mean', 'diff_vs_bot_ci_95', 'beats_bot',
        'hindsight_auprc', 'diff_vs_hin_mean', 'diff_vs_hin_ci_95', 'beats_hin'
    ]
    print(df_p2_boot[disp_p2_boot_cols].to_string(index=False))
    
    print("\n" + "=" * 90)
    print("TABLE P2-C: PROTOCOL P2 PRIMARY CLAIM OVER F, G, H (500 RESAMPLES BOOTSTRAP & POSITIVE TYPOLOGIES COUNT)")
    print("=" * 90)
    base_bot_fgh = np.mean([calc_auprc(p2_eval_info[T]['y_ev'], p2_preds[T]['best_on_train']) for T in ['F', 'G', 'H']])
    base_hin_fgh = np.mean([calc_auprc(p2_eval_info[T]['y_ev'], p2_preds[T]['hindsight_best']) for T in ['F', 'G', 'H']])
    print(f"P2 Baseline Mean AUPRC over F, G, H: best_on_train = {base_bot_fgh:.4f} | hindsight_best = {base_hin_fgh:.4f}\n")
    print(df_p2_fgh_boot.to_string(index=False))
    
    # =========================================================================
    # STEP 4: SHUFFLED CONTROL (20 PERMUTATIONS PER FOLD)
    # =========================================================================
    print("\n--- STEP 4: SHUFFLED CONTROL (20 PERMUTATIONS PER FOLD) ---")
    shuffled_records = []
    
    # 4A. Shuffled control for P1
    print("Running P1 shuffled control (6 families x 20 perms x 2 learners)...")
    for fam_name, fam_members in p1_families:
        train_mask = (df_works['split'].isin(['train', 'val'])) & (
            (~df_works['is_synthetic_fraud']) | (~df_works['typology'].isin(fam_members))
        )
        eval_mask = (df_works['typology'].isin(fam_members)) | (
            (df_works['split'] == 'test') & (~df_works['is_synthetic_fraud'])
        )
        
        y_tr = y_all[train_mask]
        y_ev = y_all[eval_mask]
        eval_prev = float(np.sum(y_ev) / len(y_ev))
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(all_feats[train_mask])
        X_ev_s = scaler.transform(all_feats[eval_mask])
        
        real_lr = float(calc_auprc(y_ev, p1_preds[fam_name]['logistic_full']))
        real_hgbc = float(calc_auprc(y_ev, p1_preds[fam_name]['hgbc_full']))
        
        lr_shufs = []
        hgbc_shufs = []
        for p_idx in range(20):
            rng_p = np.random.RandomState(2000 + ord(fam_name[0]) * 50 + p_idx)
            y_tr_p = rng_p.permutation(y_tr)
            
            lr = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=1000, random_state=0)
            lr.fit(X_tr_s, y_tr_p)
            lr_shufs.append(calc_auprc(y_ev, lr.predict_proba(X_ev_s)[:, 1]))
            
            hgbc = HistGradientBoostingClassifier(
                max_depth=3, max_iter=100, learning_rate=0.1, class_weight='balanced', random_state=0
            )
            hgbc.fit(X_tr_s, y_tr_p)
            hgbc_shufs.append(calc_auprc(y_ev, hgbc.predict_proba(X_ev_s)[:, 1]))
            
        shuffled_records.append({
            'protocol': 'P1',
            'target': fam_name,
            'eval_prevalence': eval_prev,
            'lr_real': real_lr,
            'lr_shuf_mean': float(np.mean(lr_shufs)),
            'lr_shuf_p95': float(np.percentile(lr_shufs, 95)),
            'lr_exceeds': 'YES' if real_lr > np.percentile(lr_shufs, 95) else 'NO',
            'hgbc_real': real_hgbc,
            'hgbc_shuf_mean': float(np.mean(hgbc_shufs)),
            'hgbc_shuf_p95': float(np.percentile(hgbc_shufs, 95)),
            'hgbc_exceeds': 'YES' if real_hgbc > np.percentile(hgbc_shufs, 95) else 'NO'
        })
        
    # 4B. Shuffled control for P2
    print("Running P2 shuffled control (1 train set x 20 perms x 2 learners x 5 eval sets)...")
    scaler_p2 = StandardScaler()
    X_tr_p2_s = scaler_p2.fit_transform(all_feats[p2_train_mask])
    
    # Pre-train 20 shuffled models on P2 train
    lr_shuf_p2_models = []
    hgbc_shuf_p2_models = []
    for p_idx in range(20):
        rng_p = np.random.RandomState(3000 + p_idx)
        y_tr_p = rng_p.permutation(y_tr_p2)
        
        lr = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=1000, random_state=0)
        lr.fit(X_tr_p2_s, y_tr_p)
        lr_shuf_p2_models.append(lr)
        
        hgbc = HistGradientBoostingClassifier(
            max_depth=3, max_iter=100, learning_rate=0.1, class_weight='balanced', random_state=0
        )
        hgbc.fit(X_tr_p2_s, y_tr_p)
        hgbc_shuf_p2_models.append(hgbc)
        
    for T in p2_eval_targets:
        eval_mask = (df_works['typology'] == T) | (
            (df_works['split'] == 'test') & (~df_works['is_synthetic_fraud'])
        )
        y_ev = y_all[eval_mask]
        eval_prev = float(np.sum(y_ev) / len(y_ev))
        X_ev_s = scaler_p2.transform(all_feats[eval_mask])
        
        real_lr = float(calc_auprc(y_ev, p2_preds[T]['logistic_full']))
        real_hgbc = float(calc_auprc(y_ev, p2_preds[T]['hgbc_full']))
        
        lr_shufs = [calc_auprc(y_ev, lr_m.predict_proba(X_ev_s)[:, 1]) for lr_m in lr_shuf_p2_models]
        hgbc_shufs = [calc_auprc(y_ev, hgbc_m.predict_proba(X_ev_s)[:, 1]) for hgbc_m in hgbc_shuf_p2_models]
        
        shuffled_records.append({
            'protocol': 'P2',
            'target': T,
            'eval_prevalence': eval_prev,
            'lr_real': real_lr,
            'lr_shuf_mean': float(np.mean(lr_shufs)),
            'lr_shuf_p95': float(np.percentile(lr_shufs, 95)),
            'lr_exceeds': 'YES' if real_lr > np.percentile(lr_shufs, 95) else 'NO',
            'hgbc_real': real_hgbc,
            'hgbc_shuf_mean': float(np.mean(hgbc_shufs)),
            'hgbc_shuf_p95': float(np.percentile(hgbc_shufs, 95)),
            'hgbc_exceeds': 'YES' if real_hgbc > np.percentile(hgbc_shufs, 95) else 'NO'
        })
        
    df_shuffled = pd.DataFrame(shuffled_records)
    print("\nTable 4: Shuffled Control with 20 Permutations per Fold (P1 & P2):")
    print(df_shuffled.round(4).to_string(index=False))
    
    # =========================================================================
    # STEP 5: CAT_1-ONLY RESTRICTION (DROP CATEGORY FEATURES ON P1)
    # =========================================================================
    print("\n--- STEP 5: CAT_1-ONLY RESTRICTION (DROP CATEGORY FEATURES ON P1) ---")
    no_cat_records = []
    
    for fam_name, fam_members in p1_families:
        train_mask = (df_works['split'].isin(['train', 'val'])) & (
            (~df_works['is_synthetic_fraud']) | (~df_works['typology'].isin(fam_members))
        )
        eval_mask = (df_works['typology'].isin(fam_members)) | (
            (df_works['split'] == 'test') & (~df_works['is_synthetic_fraud'])
        )
        
        y_tr = y_all[train_mask]
        y_ev = y_all[eval_mask]
        
        # 12-feature no-category matrix
        scaler_no_cat = StandardScaler()
        X_tr_nc = scaler_no_cat.fit_transform(no_cat_all_feats[train_mask])
        X_ev_nc = scaler_no_cat.transform(no_cat_all_feats[eval_mask])
        
        # Logistic
        lr_nc = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=1000, random_state=0)
        lr_nc.fit(X_tr_nc, y_tr)
        p_lr_nc = lr_nc.predict_proba(X_ev_nc)[:, 1]
        ap_lr_nc = float(calc_auprc(y_ev, p_lr_nc))
        ap_lr_full = float(calc_auprc(y_ev, p1_preds[fam_name]['logistic_full']))
        
        # HGBC
        hgbc_nc = HistGradientBoostingClassifier(
            max_depth=3, max_iter=100, learning_rate=0.1, class_weight='balanced', random_state=0
        )
        hgbc_nc.fit(X_tr_nc, y_tr)
        p_hgbc_nc = hgbc_nc.predict_proba(X_ev_nc)[:, 1]
        ap_hgbc_nc = float(calc_auprc(y_ev, p_hgbc_nc))
        ap_hgbc_full = float(calc_auprc(y_ev, p1_preds[fam_name]['hgbc_full']))
        
        # Also Cat_1 works subset evaluation (all injected are cat 1; restrict real to cat 1)
        train_cat1_mask = train_mask & (df_works['work_category_code'] == 1)
        eval_cat1_mask = eval_mask & (df_works['work_category_code'] == 1)
        y_tr_c1 = y_all[train_cat1_mask]
        y_ev_c1 = y_all[eval_cat1_mask]
        
        scaler_c1 = StandardScaler()
        X_tr_c1 = scaler_c1.fit_transform(no_cat_all_feats[train_cat1_mask])
        X_ev_c1 = scaler_c1.transform(no_cat_all_feats[eval_cat1_mask])
        
        lr_c1 = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', max_iter=1000, random_state=0)
        lr_c1.fit(X_tr_c1, y_tr_c1)
        ap_lr_c1 = float(calc_auprc(y_ev_c1, lr_c1.predict_proba(X_ev_c1)[:, 1]))
        
        hgbc_c1 = HistGradientBoostingClassifier(
            max_depth=3, max_iter=100, learning_rate=0.1, class_weight='balanced', random_state=0
        )
        hgbc_c1.fit(X_tr_c1, y_tr_c1)
        ap_hgbc_c1 = float(calc_auprc(y_ev_c1, hgbc_c1.predict_proba(X_ev_c1)[:, 1]))
        
        no_cat_records.append({
            'family': fam_name,
            'lr_full_16feat': ap_lr_full,
            'lr_drop_cat_12feat': ap_lr_nc,
            'lr_drop_cat_delta': ap_lr_nc - ap_lr_full,
            'lr_cat1_subset': ap_lr_c1,
            'lr_cat1_subset_delta': ap_lr_c1 - ap_lr_full,
            'hgbc_full_16feat': ap_hgbc_full,
            'hgbc_drop_cat_12feat': ap_hgbc_nc,
            'hgbc_drop_cat_delta': ap_hgbc_nc - ap_hgbc_full,
            'hgbc_cat1_subset': ap_hgbc_c1,
            'hgbc_cat1_subset_delta': ap_hgbc_c1 - ap_hgbc_full
        })
        
    df_no_cat = pd.DataFrame(no_cat_records)
    print("\nTable 5A: Change in AUPRC when Dropping the 4 Category Features on P1:")
    disp_drop_cols = ['family', 'lr_full_16feat', 'lr_drop_cat_12feat', 'lr_drop_cat_delta',
                      'hgbc_full_16feat', 'hgbc_drop_cat_12feat', 'hgbc_drop_cat_delta']
    print(df_no_cat[disp_drop_cols].round(4).to_string(index=False))
    
    mean_lr_drop = df_no_cat['lr_drop_cat_delta'].mean()
    mean_hgbc_drop = df_no_cat['hgbc_drop_cat_delta'].mean()
    print(f"\nMean change from dropping category features: Logistic delta = {mean_lr_drop:+.4f} | HGBC delta = {mean_hgbc_drop:+.4f}")
    
    print("\nTable 5B: Change in AUPRC when Restricting to Category-1 Works Only on P1:")
    disp_subset_cols = ['family', 'lr_full_16feat', 'lr_cat1_subset', 'lr_cat1_subset_delta',
                        'hgbc_full_16feat', 'hgbc_cat1_subset', 'hgbc_cat1_subset_delta']
    print(df_no_cat[disp_subset_cols].round(4).to_string(index=False))
    
    mean_lr_sub = df_no_cat['lr_cat1_subset_delta'].mean()
    mean_hgbc_sub = df_no_cat['hgbc_cat1_subset_delta'].mean()
    print(f"\nMean change from cat_1 subset restriction: Logistic delta = {mean_lr_sub:+.4f} | HGBC delta = {mean_hgbc_sub:+.4f}")
    
    # =========================================================================
    # STEP 6: SAVE ARTIFACTS
    # =========================================================================
    print("\n--- STEP 6: SAVING CSV ARTIFACTS ---")
    
    f_p1_m = os.path.join(out_dir, 'p1_metrics.csv')
    df_p1_metrics.to_csv(f_p1_m, index=False)
    print(f"Saved: {f_p1_m}")
    
    f_p1_b = os.path.join(out_dir, 'p1_bootstrap.csv')
    df_p1_boot.to_csv(f_p1_b, index=False)
    print(f"Saved: {f_p1_b}")
    
    f_p1_pool = os.path.join(out_dir, 'p1_pooled_summary.csv')
    df_p1_pooled_boot.to_csv(f_p1_pool, index=False)
    print(f"Saved: {f_p1_pool}")
    
    f_p2_m = os.path.join(out_dir, 'p2_metrics.csv')
    df_p2_metrics.to_csv(f_p2_m, index=False)
    print(f"Saved: {f_p2_m}")
    
    f_p2_b = os.path.join(out_dir, 'p2_bootstrap_primary_fgh.csv')
    df_p2_fgh_boot.to_csv(f_p2_b, index=False)
    print(f"Saved: {f_p2_b}")
    
    f_p2_ind = os.path.join(out_dir, 'p2_bootstrap_individual.csv')
    df_p2_boot.to_csv(f_p2_ind, index=False)
    print(f"Saved: {f_p2_ind}")
    
    f_shuf = os.path.join(out_dir, 'shuffled_control_p1_p2.csv')
    df_shuffled.to_csv(f_shuf, index=False)
    print(f"Saved: {f_shuf}")
    
    f_nc = os.path.join(out_dir, 'no_category_restriction_p1.csv')
    df_no_cat.to_csv(f_nc, index=False)
    print(f"Saved: {f_nc}")
    
    total_time = time.time() - t_start
    print(f"\nPipeline finished successfully in {total_time:.2f}s.")
    print("=" * 80)


if __name__ == '__main__':
    main()
