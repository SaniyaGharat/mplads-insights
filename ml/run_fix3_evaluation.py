"""
ml/run_fix3_evaluation.py
========================
Execution script for Fix3 Benchmark Evaluation:
1. Sensitivity evaluation counts reconciliation (primary 15,000 vs sensitivity 13,431 eligible negatives).
2. Top-1% hits (k=154) recomputation across all detectors and typologies under 3 tie rules (injected-first, real-first, random 100-shuffle mean) with rank-154 tied count.
3. Real-only community scoring (15,000 works), matched-host share > 0 & mean comparison, and Spearman correlations.
4. Matched-host empirical null (1,000 draws) with episode skips, episode-level bootstrap 95% CIs, Holm-adjusted p-values, and explicit host IDA definition for Typology H.
5. Saves all resultant tables to data/benchmark/fix3/*.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.abspath('ml'))
from community_anomaly_scorer import compute_community_scores

def fast_auroc(pos_scores, sorted_neg_scores):
    n_pos = len(pos_scores)
    n_neg = len(sorted_neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    idx_left = np.searchsorted(sorted_neg_scores, pos_scores, side='left')
    idx_right = np.searchsorted(sorted_neg_scores, pos_scores, side='right')
    ties = idx_right - idx_left
    greater = idx_left
    return float(np.sum(greater + 0.5 * ties)) / (n_pos * n_neg)

def holm_bonferroni(p_values):
    m = len(p_values)
    order = np.argsort(p_values)
    adj_p = np.empty(m)
    current_max = 0.0
    for rank, idx in enumerate(order):
        p_val = p_values[idx]
        p_adj = min(1.0, p_val * (m - rank))
        current_max = max(current_max, p_adj)
        adj_p[idx] = current_max
    return adj_p

def main():
    out_dir = os.path.join("data", "benchmark", "fix3")
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 100)
    print("FIX3 EVALUATION: SENSITIVITY COUNTS, TIE-BREAKING, REAL-ONLY SCORING & MATCHED-HOST NULL")
    print("=" * 100)

    # Load datasets
    df_works = pd.read_csv(os.path.join("data", "benchmark", "benchmark_works.csv"))
    df_scores = pd.read_csv(os.path.join("data", "benchmark", "detector_scores_v2.csv"))
    df_works['sanc_dt'] = pd.to_datetime(df_works['Sanction Date'])
    
    df_real = df_works[df_works['is_synthetic_fraud'] == False].copy()
    df_syn = df_works[df_works['is_synthetic_fraud'] == True].copy()
    
    typologies = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    # =========================================================================
    # STEP 1: SENSITIVITY EVALUATION COUNTS RECONCILIATION
    # =========================================================================
    print("\n" + "=" * 100)
    print("STEP 1: SENSITIVITY EVALUATION COUNTS RECONCILIATION")
    print("=" * 100)
    
    mp_counts = df_real['MP'].value_counts()
    ida_counts = df_real['IDA'].value_counts()
    e_mps = set(mp_counts[mp_counts >= 15].index)
    e_idas = set(ida_counts[ida_counts >= 15].index)
    
    elig_mask = df_real['MP'].isin(e_mps) & df_real['IDA'].isin(e_idas)
    df_real_elig = df_real[elig_mask].copy().reset_index(drop=True)
    
    sens_counts_rows = []
    for typ in typologies:
        sub_syn = df_syn[df_syn['typology'] == typ]
        sens_counts_rows.append({
            'Typology': typ,
            'Episodes': sub_syn['episode_id'].nunique(),
            'Positives (Injected Works)': len(sub_syn),
            'Primary Negatives (All Real Works)': len(df_real),
            'Sensitivity Negatives (Eligible Real Works)': len(df_real_elig),
            'Sensitivity Ratio (Pos/Neg)': f"{len(sub_syn) / len(df_real_elig):.6f}",
        })
    sens_counts_rows.append({
        'Typology': 'ALL (Pooled)',
        'Episodes': df_syn['episode_id'].nunique(),
        'Positives (Injected Works)': len(df_syn),
        'Primary Negatives (All Real Works)': len(df_real),
        'Sensitivity Negatives (Eligible Real Works)': len(df_real_elig),
        'Sensitivity Ratio (Pos/Neg)': f"{len(df_syn) / len(df_real_elig):.6f}",
    })
    
    df_sens_counts = pd.DataFrame(sens_counts_rows)
    print(df_sens_counts.to_string(index=False))
    df_sens_counts.to_csv(os.path.join(out_dir, "sensitivity_counts.csv"), index=False)
    
    # =========================================================================
    # STEP 2: TOP-1% TIE RULES EVALUATION (k=154)
    # =========================================================================
    print("\n" + "=" * 100)
    print("STEP 2: TOP-1% TIE RULES EVALUATION (k=154, ALL DETECTORS x TYPOLOGIES A-H)")
    print("=" * 100)
    
    detectors = {
        'gat_frozen': 'GAT (Frozen)',
        'gat_augmented': 'GAT (Augmented)',
        'community_score': 'Community Scorer',
        'iforest_score': 'Isolation Forest',
        'robust_z_score': 'Robust Z-score',
    }
    
    k_1pct = 154
    N = len(df_scores)
    y_syn = (df_scores['is_synthetic_fraud'] == True).astype(int).values
    
    tie_summary_rows = []
    
    for det_col, det_name in detectors.items():
        s = df_scores[det_col].values
        
        # Sort descending to find score at 154th position
        s_sorted_desc = np.sort(s)[::-1]
        score_at_154 = s_sorted_desc[k_1pct - 1]
        n_tied_at_154 = int(np.sum(s == score_at_154))
        
        df_temp = df_scores[['work_index', 'is_synthetic_fraud', 'typology', det_col]].copy()
        
        # Injected-first sort (major: -s, minor: -y)
        idx_inj_first = np.lexsort((-y_syn, -s))
        top_inj_first = df_temp.iloc[idx_inj_first[:k_1pct]]
        vc_inj = top_inj_first[top_inj_first['is_synthetic_fraud'] == True]['typology'].value_counts()
        
        # Real-first sort (major: -s, minor: y)
        idx_real_first = np.lexsort((y_syn, -s))
        top_real_first = df_temp.iloc[idx_real_first[:k_1pct]]
        vc_real = top_real_first[top_real_first['is_synthetic_fraud'] == True]['typology'].value_counts()
        
        # Random shuffles (100 shuffles)
        rng = np.random.RandomState(42)
        rand_counts = {typ: [] for typ in typologies}
        rand_totals = []
        for _ in range(100):
            r = rng.rand(N)
            idx_rand = np.lexsort((r, -s))
            top_rand = df_temp.iloc[idx_rand[:k_1pct]]
            syn_rand = top_rand[top_rand['is_synthetic_fraud'] == True]
            rand_totals.append(len(syn_rand))
            vc_rand = syn_rand['typology'].value_counts()
            for typ in typologies:
                rand_counts[typ].append(vc_rand.get(typ, 0))
                
        for typ in typologies:
            n_tot_typ = (df_scores['typology'] == typ).sum()
            tie_summary_rows.append({
                'Detector': det_name,
                'Detector_Column': det_col,
                'Typology': typ,
                'Total_Injected_Works': n_tot_typ,
                'Score_At_Rank_154': score_at_154,
                'Tied_Works_At_Rank_154': n_tied_at_154,
                'Hits_Injected_First': vc_inj.get(typ, 0),
                'Hits_Real_First': vc_real.get(typ, 0),
                'Hits_Random_Mean_100': np.mean(rand_counts[typ]),
            })
            
        # Summary row for total injected works
        tie_summary_rows.append({
            'Detector': det_name,
            'Detector_Column': det_col,
            'Typology': 'ALL (Total)',
            'Total_Injected_Works': len(df_syn),
            'Score_At_Rank_154': score_at_154,
            'Tied_Works_At_Rank_154': n_tied_at_154,
            'Hits_Injected_First': len(top_inj_first[top_inj_first['is_synthetic_fraud'] == True]),
            'Hits_Real_First': len(top_real_first[top_real_first['is_synthetic_fraud'] == True]),
            'Hits_Random_Mean_100': np.mean(rand_totals),
        })

    df_ties = pd.DataFrame(tie_summary_rows)
    print(df_ties.to_string(index=False))
    df_ties.to_csv(os.path.join(out_dir, "top_1pct_ties_recomputed.csv"), index=False)
    
    # =========================================================================
    # STEP 3: COMMUNITY SCORER REAL-ONLY SCORING & MATCHED-HOST COMPARISON
    # =========================================================================
    print("\n" + "=" * 100)
    print("STEP 3: COMMUNITY SCORER REAL-ONLY RUN (15,000 REAL WORKS) & MATCHED HOST COMPARISON")
    print("=" * 100)
    
    comm_scores_real, comm_meta_real = compute_community_scores(df_real, window_days=30, eps=0.001)
    
    # Save new file with real-only scores
    df_comm_real_out = pd.DataFrame({
        'work_index': df_real['work_index'],
        'community_score': comm_scores_real,
        'cluster_distinct_mps': comm_meta_real['cluster_distinct_mps'],
        'cluster_works_count': comm_meta_real['cluster_works_count'],
        'cluster_total_amount': comm_meta_real['cluster_total_amount'],
        'cluster_amount_cv': comm_meta_real['cluster_amount_cv'],
        'cluster_cv_ratio': comm_meta_real['cluster_cv_ratio']
    })
    real_out_path = os.path.join(out_dir, "community_scores_real_only.csv")
    df_comm_real_out.to_csv(real_out_path, index=False)
    print(f"Saved real-only community scores to: {real_out_path} ({len(df_comm_real_out)} rows)")
    
    comm_aug_real = df_scores[df_scores['is_synthetic_fraud'] == False]['community_score'].values
    sp_all, p_all = spearmanr(comm_scores_real, comm_aug_real)
    print(f"\nSpearman Correlation between Real-Only and Augmented runs on ALL 15,000 Real Works: {sp_all:.6f} (p = {p_all:.4e})")
    
    window_delta = np.timedelta64(30, 'D')
    comm_comp_rows = []
    
    for typ in typologies:
        sub_syn = df_works[(df_works['is_synthetic_fraud'] == True) & (df_works['typology'] == typ)]
        
        matched_real_indices = set()
        for ep_id, grp in sub_syn.groupby('episode_id'):
            ep_idas = grp['IDA'].unique()
            min_date = grp['sanc_dt'].min()
            max_date = grp['sanc_dt'].max()
            
            ida_real = df_works[
                (df_works['is_synthetic_fraud'] == False) &
                (df_works['IDA'].isin(ep_idas)) &
                (df_works['sanc_dt'] >= min_date - window_delta) &
                (df_works['sanc_dt'] <= max_date + window_delta)
            ]
            matched_real_indices.update(ida_real.index.tolist())
            
        m_idx = sorted(list(matched_real_indices))
        n_m = len(m_idx)
        
        s_r = comm_scores_real[m_idx]
        s_a = comm_aug_real[m_idx]
        
        sp_typ, p_typ = spearmanr(s_r, s_a) if n_m > 1 else (np.nan, np.nan)
        
        comm_comp_rows.append({
            'Typology': typ,
            'Matched_Real_Works_N': n_m,
            'Real_Only_Share_GT_0': np.mean(s_r > 0),
            'Real_Only_Mean_Score': np.mean(s_r),
            'Augmented_Share_GT_0': np.mean(s_a > 0),
            'Augmented_Mean_Score': np.mean(s_a),
            'Spearman_Correlation': sp_typ,
            'Spearman_p_value': p_typ,
            'Host_IDA_Definition': 'All unique IDAs in episode span +- 30d' if typ == 'H' else 'Host IDA in episode span +- 30d'
        })
        
    # Also evaluate H under first-IDA only definition
    matched_real_Hf = set()
    for ep_id, grp in df_works[(df_works['is_synthetic_fraud'] == True) & (df_works['typology'] == 'H')].groupby('episode_id'):
        ep_ida = grp['IDA'].iloc[0]
        min_date = grp['sanc_dt'].min()
        max_date = grp['sanc_dt'].max()
        ida_real = df_works[
            (df_works['is_synthetic_fraud'] == False) &
            (df_works['IDA'] == ep_ida) &
            (df_works['sanc_dt'] >= min_date - window_delta) &
            (df_works['sanc_dt'] <= max_date + window_delta)
        ]
        matched_real_Hf.update(ida_real.index.tolist())
    m_idx_Hf = sorted(list(matched_real_Hf))
    s_r_Hf = comm_scores_real[m_idx_Hf]
    s_a_Hf = comm_aug_real[m_idx_Hf]
    sp_Hf, p_Hf = spearmanr(s_r_Hf, s_a_Hf)
    comm_comp_rows.append({
        'Typology': 'H (First-IDA Only)',
        'Matched_Real_Works_N': len(m_idx_Hf),
        'Real_Only_Share_GT_0': np.mean(s_r_Hf > 0),
        'Real_Only_Mean_Score': np.mean(s_r_Hf),
        'Augmented_Share_GT_0': np.mean(s_a_Hf > 0),
        'Augmented_Mean_Score': np.mean(s_a_Hf),
        'Spearman_Correlation': sp_Hf,
        'Spearman_p_value': p_Hf,
        'Host_IDA_Definition': 'First work IDA in episode span +- 30d'
    })
    
    df_comm_comp = pd.DataFrame(comm_comp_rows)
    print("\nMatched-Host Real Works Community Scorer Comparison Table:")
    print(df_comm_comp.to_string(index=False))
    df_comm_comp.to_csv(os.path.join(out_dir, "community_matched_host_comparison.csv"), index=False)
    
    # =========================================================================
    # STEP 4: MATCHED-HOST NULL EVALUATION (1,000 DRAWS & BOOTSTRAP 95% CI)
    # =========================================================================
    print("\n" + "=" * 100)
    print("STEP 4: MATCHED-HOST NULL EVALUATION (1,000 DRAWS, BOOTSTRAP 95% CI, HOLM ADJUSTMENT)")
    print("=" * 100)
    print("Definition Note for Typology H:")
    print("  Typology H (One-to-Many Fan-Out) injects works across 1 to 6 distinct IDAs per episode.")
    print("  Primary Host Definition: 'all_idas' — host IDAs are the set of unique IDAs touched by the episode,")
    print("  and candidates are real works at any of those IDAs within episode span +- 30 days.")
    print("  Sensitivity Definition: 'first_ida' — host IDA is restricted strictly to the first work's IDA.")
    
    def run_matched_null_experiment(score_mode='augmented', h_mode='all_idas'):
        results = []
        raw_p_list = []
        
        for typ in typologies:
            sub_syn = df_syn[df_syn['typology'] == typ]
            
            ep_dict = {}
            valid_ep_ids = []
            skipped_ep_ids = []
            
            for ep_id, grp in sub_syn.groupby('episode_id'):
                k = len(grp)
                min_dt = grp['sanc_dt'].min()
                max_dt = grp['sanc_dt'].max()
                
                if typ == 'H':
                    ep_idas = grp['IDA'].unique() if h_mode == 'all_idas' else [grp['IDA'].iloc[0]]
                else:
                    ep_idas = [grp['IDA'].iloc[0]]
                    
                cand = df_real[
                    (df_real['IDA'].isin(ep_idas)) &
                    (df_real['sanc_dt'] >= min_dt - window_delta) &
                    (df_real['sanc_dt'] <= max_dt + window_delta)
                ]
                cand_indices = cand.index.tolist()
                
                if len(cand_indices) >= k:
                    valid_ep_ids.append(ep_id)
                    ep_dict[ep_id] = {
                        'k': k,
                        'inj_indices': grp.index.tolist(),
                        'cand_indices': np.array(cand_indices)
                    }
                else:
                    skipped_ep_ids.append(f"{ep_id} (k={k}, avail={len(cand_indices)})")
                    
            n_valid = len(valid_ep_ids)
            
            # Deduplicated matched real works across valid episodes
            all_cand_indices = set()
            for ep_id in valid_ep_ids:
                all_cand_indices.update(ep_dict[ep_id]['cand_indices'].tolist())
            matched_real_arr = np.array(sorted(list(all_cand_indices)))
            
            # Injected works for valid episodes
            retained_inj_arr = np.array([idx for ep_id in valid_ep_ids for idx in ep_dict[ep_id]['inj_indices']])
            
            # Get scores via direct numpy indexing
            comm_scores_aug = df_scores['community_score'].values
            s_inj = comm_scores_aug[retained_inj_arr]
            if score_mode == 'augmented':
                s_matched = comm_scores_aug[matched_real_arr]
            else:
                s_matched = comm_scores_real[matched_real_arr]
                
            s_matched_sorted = np.sort(s_matched)
            actual_auroc = fast_auroc(s_inj, s_matched_sorted)
            
            # 1,000 empirical null draws
            rng = np.random.RandomState(42)
            N_DRAWS = 1000
            null_aurocs = []
            
            for _ in range(N_DRAWS):
                pseudo_inj = []
                for ep_id in valid_ep_ids:
                    c_idx = ep_dict[ep_id]['cand_indices']
                    k_ep = ep_dict[ep_id]['k']
                    chosen = rng.choice(c_idx, size=k_ep, replace=False)
                    pseudo_inj.extend(chosen)
                pseudo_inj = np.array(pseudo_inj)
                
                s_p = comm_scores_aug[pseudo_inj] if score_mode == 'augmented' else comm_scores_real[pseudo_inj]
                null_aurocs.append(fast_auroc(s_p, s_matched_sorted))
                
            null_aurocs = np.array(null_aurocs)
            null_mean = np.mean(null_aurocs)
            null_std = np.std(null_aurocs)
            raw_p = (np.sum(null_aurocs >= actual_auroc) + 1) / (N_DRAWS + 1)
            raw_p_list.append(raw_p)
            
            # Episode-level bootstrap 95% CI
            N_BOOT = 1000
            boot_aurocs = []
            for _ in range(N_BOOT):
                boot_eps = rng.choice(valid_ep_ids, size=n_valid, replace=True)
                b_inj_idx = []
                b_cand_idx = set()
                for ep_id in boot_eps:
                    b_inj_idx.extend(ep_dict[ep_id]['inj_indices'])
                    b_cand_idx.update(ep_dict[ep_id]['cand_indices'].tolist())
                b_cand_arr = np.array(sorted(list(b_cand_idx)))
                
                s_b_inj = comm_scores_aug[b_inj_idx]
                s_b_neg = comm_scores_aug[b_cand_arr] if score_mode == 'augmented' else comm_scores_real[b_cand_arr]
                
                boot_aurocs.append(fast_auroc(s_b_inj, np.sort(s_b_neg)))
                
            ci_lower = np.percentile(boot_aurocs, 2.5)
            ci_upper = np.percentile(boot_aurocs, 97.5)
            
            results.append({
                'Typology': typ,
                'Valid_Episodes': f"{n_valid}/{sub_syn['episode_id'].nunique()}",
                'Skipped_Episodes_Count': len(skipped_ep_ids),
                'Skipped_Episodes': "; ".join(skipped_ep_ids) if skipped_ep_ids else "None",
                'Retained_Injected_Works': len(retained_inj_arr),
                'Matched_Real_Works_N': len(matched_real_arr),
                'Actual_AUROC': actual_auroc,
                'Bootstrap_95_CI': f"[{ci_lower:.4f}, {ci_upper:.4f}]",
                'Bootstrap_CI_Lower': ci_lower,
                'Bootstrap_CI_Upper': ci_upper,
                'Null_Mean': null_mean,
                'Null_Std': null_std,
                'Null_AUROC_Mean_Std': f"{null_mean:.4f} +- {null_std:.4f}",
                'Raw_p': raw_p,
            })
            
        adj_p = holm_bonferroni(raw_p_list)
        for i, res in enumerate(results):
            res['Holm_p'] = adj_p[i]
            res['Detected_Adj_p_lt_005'] = "YES" if adj_p[i] < 0.05 else "NO"
            
        return pd.DataFrame(results)

    # Run primary matched-host null (Augmented scores, H=all_idas)
    df_null_primary = run_matched_null_experiment(score_mode='augmented', h_mode='all_idas')
    print("\n[A] PRIMARY MATCHED-HOST NULL (Augmented Scores, H = All Episode IDAs):")
    print(df_null_primary[['Typology', 'Valid_Episodes', 'Retained_Injected_Works', 'Matched_Real_Works_N', 'Actual_AUROC', 'Bootstrap_95_CI', 'Null_AUROC_Mean_Std', 'Raw_p', 'Holm_p', 'Detected_Adj_p_lt_005']].to_string(index=False))
    df_null_primary.to_csv(os.path.join(out_dir, "matched_host_null_results.csv"), index=False)

    # Run real-only matched-host null
    df_null_real = run_matched_null_experiment(score_mode='real_only', h_mode='all_idas')
    print("\n[B] REAL-ONLY MATCHED-HOST NULL (Real-Only Baseline Scores, H = All Episode IDAs):")
    print(df_null_real[['Typology', 'Valid_Episodes', 'Retained_Injected_Works', 'Matched_Real_Works_N', 'Actual_AUROC', 'Bootstrap_95_CI', 'Null_AUROC_Mean_Std', 'Raw_p', 'Holm_p', 'Detected_Adj_p_lt_005']].to_string(index=False))
    df_null_real.to_csv(os.path.join(out_dir, "matched_host_null_real_only.csv"), index=False)

    # Run first-IDA matched-host null for H sensitivity
    df_null_first_ida = run_matched_null_experiment(score_mode='augmented', h_mode='first_ida')
    print("\n[C] SENSITIVITY MATCHED-HOST NULL (Augmented Scores, H = First IDA Only):")
    print(df_null_first_ida[['Typology', 'Valid_Episodes', 'Retained_Injected_Works', 'Matched_Real_Works_N', 'Actual_AUROC', 'Bootstrap_95_CI', 'Null_AUROC_Mean_Std', 'Raw_p', 'Holm_p', 'Detected_Adj_p_lt_005']].to_string(index=False))
    df_null_first_ida.to_csv(os.path.join(out_dir, "matched_host_null_h_first_ida.csv"), index=False)

    print(f"\nAll Fix3 tables successfully generated and saved to {out_dir}/")

if __name__ == '__main__':
    main()
