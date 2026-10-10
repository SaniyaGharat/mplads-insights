"""
ml/run_fix5_evaluation.py

Fix 5 Evaluation Pipeline:
1. Recompute matched-host null with disjoint negatives (pool minus drawn positives),
   1,000 draws, seed=42; report null mean/std, raw p, Holm p, and re-apply pre-registered rule.
2. Like-for-like placebo: for A, D, F, score TRUE episodes in the same reduced set used
   for the placebo (15,000 real works + that typology's retained episodes only).
   Report share>0, mean score, AUROC vs matched real, beside existing placebo numbers.
   Also report fraction of placebo windows with CV == 0 and how many works hit the 1000 cap
   in true vs placebo runs.
3. Save tables to data/benchmark/fix5/.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Ensure ml/ is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

OUT_DIR = os.path.join('data', 'benchmark', 'fix5')
os.makedirs(OUT_DIR, exist_ok=True)


def fast_auroc(pos_scores, sorted_neg_scores):
    """Compute AUROC efficiently using searchsorted on pre-sorted negative scores."""
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
    """Apply Holm-Bonferroni step-down adjustment."""
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


def compute_community_scores_tracked(df, window_days=30, eps=0.001):
    """
    Computes community scores while tracking rolling multi-MP windows,
    zero-CV windows, and cap hits.
    """
    df = df.copy()
    df['Sanction_Date_dt'] = pd.to_datetime(df['Sanction Date'])
    n_total = len(df)
    work_scores = np.zeros(n_total, dtype=np.float64)
    window_delta = np.timedelta64(window_days, 'D')

    total_multi_mp_windows = 0
    zero_cv_windows = 0

    for ida, g in df.groupby('IDA'):
        if len(g) < 2:
            continue
        g_sorted = g.sort_values('Sanction_Date_dt')
        indices = g_sorted.index.values
        dates = g_sorted['Sanction_Date_dt'].values
        amounts = g_sorted['sanction_amount'].values
        mps = g_sorted['MP'].values
        n_group = len(g_sorted)

        for i in range(n_group):
            d_start = dates[i]
            d_end = d_start + window_delta
            right_idx = np.searchsorted(dates, d_end, side='right')
            w_mps = mps[i:right_idx]
            if len(set(w_mps)) >= 2:
                w_amts = amounts[i:right_idx]
                total_multi_mp_windows += 1
                w_std = np.std(w_amts)
                w_mean = np.mean(w_amts)
                w_cv = float(w_std / (w_mean + 1e-6))
                if w_cv < 1e-9:
                    zero_cv_windows += 1
                score = 1.0 / (w_cv + eps)
                w_idx = indices[i:right_idx]
                for r in w_idx:
                    if score > work_scores[r]:
                        work_scores[r] = score

    return work_scores, total_multi_mp_windows, zero_cv_windows


def run_step3_disjoint_null(df_works, df_scores):
    print("=" * 80)
    print("STEP 3: MATCHED-HOST NULL WITH DISJOINT NEGATIVES (POOL MINUS DRAWN POSITIVES)")
    print("=" * 80)

    df_works = df_works.copy()
    df_works['sanc_dt'] = pd.to_datetime(df_works['Sanction Date'])
    window_delta = np.timedelta64(30, 'D')

    syn = df_works[df_works['is_synthetic_fraud'] == True]
    real = df_works[df_works['is_synthetic_fraud'] == False]
    comm_scores_aug = df_scores['community_score'].values

    typologies = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    rows = []
    raw_p_list = []

    for typ in typologies:
        sub_syn = syn[syn['typology'] == typ]
        valid_ep_ids = []
        ep_dict = {}

        for ep_id, grp in sub_syn.groupby('episode_id'):
            k = len(grp)
            min_dt = grp['sanc_dt'].min()
            max_dt = grp['sanc_dt'].max()
            ep_idas = grp['IDA'].unique() if typ == 'H' else [grp['IDA'].iloc[0]]

            cand = real[
                (real['IDA'].isin(ep_idas)) &
                (real['sanc_dt'] >= min_dt - window_delta) &
                (real['sanc_dt'] <= max_dt + window_delta)
            ]
            cand_idx = cand.index.tolist()
            if len(cand_idx) >= k:
                valid_ep_ids.append(ep_id)
                ep_dict[ep_id] = {'k': k, 'cand': cand_idx, 'inj': grp.index.tolist()}

        all_matched = set()
        total_k = 0
        for ep_id in valid_ep_ids:
            all_matched.update(ep_dict[ep_id]['cand'])
            total_k += ep_dict[ep_id]['k']

        matched_pool = np.array(sorted(list(all_matched)))
        n_pool = len(matched_pool)

        retained_inj = [idx for ep in valid_ep_ids for idx in ep_dict[ep]['inj']]
        s_inj = comm_scores_aug[retained_inj]
        s_matched_sorted = np.sort(comm_scores_aug[matched_pool])
        act_auroc = fast_auroc(s_inj, s_matched_sorted)

        # 1,000 null draws with disjoint negatives (pool minus drawn positives)
        rng = np.random.RandomState(42)
        N_DRAWS = 1000
        null_aurocs = []

        for _ in range(N_DRAWS):
            pseudo_inj = []
            for ep_id in valid_ep_ids:
                c_idx = ep_dict[ep_id]['cand']
                k_ep = ep_dict[ep_id]['k']
                pseudo_inj.extend(rng.choice(c_idx, size=k_ep, replace=False))
            p_set = set(pseudo_inj)
            neg_idx = np.array([idx for idx in matched_pool if idx not in p_set])
            s_p = comm_scores_aug[pseudo_inj]
            s_n = np.sort(comm_scores_aug[neg_idx])
            null_aurocs.append(fast_auroc(s_p, s_n))

        null_aurocs = np.array(null_aurocs)
        null_mean = float(np.mean(null_aurocs))
        null_std = float(np.std(null_aurocs))
        raw_p = float((np.sum(null_aurocs >= act_auroc) + 1) / (N_DRAWS + 1))
        raw_p_list.append(raw_p)

        # Bootstrap 95% CI over valid episodes
        N_BOOT = 1000
        boot_aurocs = []
        n_valid = len(valid_ep_ids)
        if n_valid > 1:
            for _ in range(N_BOOT):
                boot_ep_ids = rng.choice(valid_ep_ids, size=n_valid, replace=True)
                b_inj = [idx for ep in boot_ep_ids for idx in ep_dict[ep]['inj']]
                boot_aurocs.append(fast_auroc(comm_scores_aug[b_inj], s_matched_sorted))
            ci_lower = float(np.percentile(boot_aurocs, 2.5))
            ci_upper = float(np.percentile(boot_aurocs, 97.5))
        elif n_valid == 1:
            ci_lower = act_auroc
            ci_upper = act_auroc
        else:
            ci_lower = np.nan
            ci_upper = np.nan

        rows.append({
            'Typology': typ,
            'Valid_Episodes': f"{n_valid}/{sub_syn['episode_id'].nunique()}",
            'Valid_Eps_Count': n_valid,
            'Retained_Inj_Works': len(retained_inj),
            'Matched_Pool_Size': n_pool,
            'Actual_AUROC': act_auroc,
            'Bootstrap_CI_Lower': ci_lower,
            'Bootstrap_CI_Upper': ci_upper,
            'Disjoint_Null_Mean': null_mean,
            'Disjoint_Null_Std': null_std,
            'Disjoint_Null_Mean_Std': f"{null_mean:.4f} +- {null_std:.4f}",
            'Raw_p': raw_p,
        })

    # Apply Holm adjustment
    holm_p_list = holm_bonferroni(raw_p_list)

    for i, r in enumerate(rows):
        h_p = float(holm_p_list[i])
        r['Holm_p'] = h_p

        # Re-apply pre-registered rule:
        # detected = AUROC > 0.5 AND bootstrap CI lower bound > 0.5 AND Holm p < 0.05 AND at least 8 valid episodes
        c_auroc = r['Actual_AUROC'] > 0.5
        c_ci = r['Bootstrap_CI_Lower'] > 0.5
        c_holm = h_p < 0.05
        c_eps = r['Valid_Eps_Count'] >= 8
        detected = c_auroc and c_ci and c_holm and c_eps

        r['Rule_AUROC_gt_05'] = c_auroc
        r['Rule_CI_Lower_gt_05'] = c_ci
        r['Rule_Holm_p_lt_005'] = c_holm
        r['Rule_Valid_Eps_gte_8'] = c_eps
        r['Detected_Preregistered'] = 'YES' if detected else 'NO'

    df_step3 = pd.DataFrame(rows)
    cols_display = [
        'Typology', 'Valid_Episodes', 'Retained_Inj_Works', 'Matched_Pool_Size',
        'Actual_AUROC', 'Bootstrap_CI_Lower', 'Disjoint_Null_Mean_Std',
        'Raw_p', 'Holm_p', 'Rule_CI_Lower_gt_05', 'Rule_Holm_p_lt_005',
        'Rule_Valid_Eps_gte_8', 'Detected_Preregistered'
    ]
    print(df_step3[cols_display].to_string(index=False))

    out_file = os.path.join(OUT_DIR, 'matched_host_null_disjoint.csv')
    df_step3.to_csv(out_file, index=False)
    print(f"\nSaved disjoint matched-host null table to {out_file}")
    return df_step3


def run_step4_like_for_like_placebo(df_works):
    print("\n" + "=" * 80)
    print("STEP 4: LIKE-FOR-LIKE PLACEBO CONTROL (TYPOLOGIES A, D, F)")
    print("Scores TRUE episodes in the same reduced set used for placebo (15,000 real + that typology only)")
    print("=" * 80)

    df_works = df_works.copy()
    df_works['sanc_dt'] = pd.to_datetime(df_works['Sanction Date'])
    real = df_works[df_works['is_synthetic_fraud'] == False].copy().reset_index(drop=True)
    syn = df_works[df_works['is_synthetic_fraud'] == True].copy()
    cols = ['IDA', 'MP', 'sanction_amount', 'Sanction Date']
    window_delta = np.timedelta64(30, 'D')

    real_amts_by_ida = {ida: g['sanction_amount'].values for ida, g in real.groupby('IDA')}
    global_real_amts = real['sanction_amount'].values

    rows = []
    N_PLACEBO = 20

    for typ in ['A', 'D', 'F']:
        sub_syn = syn[syn['typology'] == typ]
        valid_ep_ids = []
        ep_data = {}

        for ep_id, grp in sub_syn.groupby('episode_id'):
            k = len(grp)
            min_dt = grp['sanc_dt'].min()
            max_dt = grp['sanc_dt'].max()
            ep_idas = [grp['IDA'].iloc[0]]
            cand = real[
                (real['IDA'].isin(ep_idas)) &
                (real['sanc_dt'] >= min_dt - window_delta) &
                (real['sanc_dt'] <= max_dt + window_delta)
            ]
            if len(cand) >= k:
                valid_ep_ids.append(ep_id)
                ep_data[ep_id] = {'df': grp.copy(), 'cand_idx': cand.index.tolist()}

        retained_inj_df = pd.concat([ep_data[ep]['df'] for ep in valid_ep_ids], ignore_index=True)
        all_matched = set()
        for ep in valid_ep_ids:
            all_matched.update(ep_data[ep]['cand_idx'])
        matched_real_idx = sorted(list(all_matched))

        # 1. True reduced run: 15,000 real works + that typology's retained true episodes only
        df_true = pd.concat([real[cols], retained_inj_df[cols]], ignore_index=True)
        scores_true, tot_w_true, zero_w_true = compute_community_scores_tracked(df_true)
        n_real = len(real)
        scores_inj_true = scores_true[n_real:]
        scores_matched_true = scores_true[matched_real_idx]

        true_share_gt0 = float(np.mean(scores_inj_true > 0))
        true_mean = float(np.mean(scores_inj_true))
        y_true = np.array([1] * len(scores_inj_true) + [0] * len(scores_matched_true))
        s_true = np.concatenate([scores_inj_true, scores_matched_true])
        true_auc = float(roc_auc_score(y_true, s_true))

        true_cap_hits_inj = int(np.sum(scores_inj_true >= 999.99))
        true_cap_hits_all = int(np.sum(scores_true >= 999.99))
        true_zero_cv_frac = float(zero_w_true / tot_w_true) if tot_w_true > 0 else 0.0

        # 2. Placebo runs: 20 draws of empirical amounts
        rng = np.random.RandomState(42)
        p_shares, p_means, p_aucs = [], [], []
        p_zero_cv_fracs = []
        p_cap_inj, p_cap_all = [], []

        for _ in range(N_PLACEBO):
            placebo_ep_dfs = []
            for ep in valid_ep_ids:
                df_ep = ep_data[ep]['df'].copy()
                ep_ida = df_ep['IDA'].iloc[0]
                pool = real_amts_by_ida.get(ep_ida, global_real_amts)
                if len(pool) < 15:
                    pool = global_real_amts
                df_ep['sanction_amount'] = rng.choice(pool, size=len(df_ep), replace=True)
                placebo_ep_dfs.append(df_ep[cols])

            df_p = pd.concat([real[cols]] + placebo_ep_dfs, ignore_index=True)
            scores_p, tot_w_p, zero_w_p = compute_community_scores_tracked(df_p)
            scores_inj_p = scores_p[n_real:]
            scores_matched_p = scores_p[matched_real_idx]

            p_shares.append(float(np.mean(scores_inj_p > 0)))
            p_means.append(float(np.mean(scores_inj_p)))
            y_p = np.array([1] * len(scores_inj_p) + [0] * len(scores_matched_p))
            s_p = np.concatenate([scores_inj_p, scores_matched_p])
            p_aucs.append(float(roc_auc_score(y_p, s_p)))

            p_zero_cv_fracs.append(float(zero_w_p / tot_w_p) if tot_w_p > 0 else 0.0)
            p_cap_inj.append(int(np.sum(scores_inj_p >= 999.99)))
            p_cap_all.append(int(np.sum(scores_p >= 999.99)))

        rows.append({
            'Typology': typ,
            'Valid_Episodes': f"{len(valid_ep_ids)}/{sub_syn['episode_id'].nunique()}",
            'Retained_Works': len(scores_inj_true),
            'Matched_Real_Works': len(matched_real_idx),
            # True Reduced Run Metrics
            'True_Reduced_Share_GT_0': true_share_gt0,
            'True_Reduced_Mean_Score': true_mean,
            'True_Reduced_AUROC_vs_Matched': true_auc,
            'True_Reduced_Cap_Hits_Injected': true_cap_hits_inj,
            'True_Reduced_Cap_Hits_Total': true_cap_hits_all,
            'True_Reduced_Multi_MP_Windows': tot_w_true,
            'True_Reduced_Zero_CV_Windows': zero_w_true,
            'True_Reduced_Zero_CV_Frac': true_zero_cv_frac,
            # Placebo Control Metrics (20 draws)
            'Placebo_Share_GT_0_Mean': float(np.mean(p_shares)),
            'Placebo_Share_GT_0_Std': float(np.std(p_shares)),
            'Placebo_Mean_Score_Mean': float(np.mean(p_means)),
            'Placebo_Mean_Score_Std': float(np.std(p_means)),
            'Placebo_AUROC_Mean': float(np.mean(p_aucs)),
            'Placebo_AUROC_Std': float(np.std(p_aucs)),
            'Placebo_Cap_Hits_Injected_Mean': float(np.mean(p_cap_inj)),
            'Placebo_Cap_Hits_Injected_Std': float(np.std(p_cap_inj)),
            'Placebo_Cap_Hits_Total_Mean': float(np.mean(p_cap_all)),
            'Placebo_Cap_Hits_Total_Std': float(np.std(p_cap_all)),
            'Placebo_Zero_CV_Window_Frac_Mean': float(np.mean(p_zero_cv_fracs)),
            'Placebo_Zero_CV_Window_Frac_Std': float(np.std(p_zero_cv_fracs)),
        })

        print(f"\nTypology {typ}:")
        print(f"  TRUE (Reduced Run):")
        print(f"    Share > 0:            {true_share_gt0*100:.2f}%")
        print(f"    Mean Score:           {true_mean:.4f}")
        print(f"    AUROC vs Matched Real:{true_auc:.4f}")
        print(f"    Works hitting 1000 cap: Injected={true_cap_hits_inj}/{len(scores_inj_true)}, Total={true_cap_hits_all}")
        print(f"    Zero-CV Windows:      {true_zero_cv_frac*100:.4f}% ({zero_w_true}/{tot_w_true} windows)")
        print(f"  PLACEBO CONTROL (20 Draws):")
        print(f"    Share > 0:            {np.mean(p_shares)*100:.2f}% +- {np.std(p_shares)*100:.2f}%")
        print(f"    Mean Score:           {np.mean(p_means):.4f} +- {np.std(p_means):.4f}")
        print(f"    AUROC vs Matched Real:{np.mean(p_aucs):.4f} +- {np.std(p_aucs):.4f}")
        print(f"    Works hitting 1000 cap: Injected={np.mean(p_cap_inj):.2f} +- {np.std(p_cap_inj):.2f}, Total={np.mean(p_cap_all):.2f} +- {np.std(p_cap_all):.2f}")
        print(f"    Zero-CV Windows:      {np.mean(p_zero_cv_fracs)*100:.4f}% +- {np.std(p_zero_cv_fracs)*100:.4f}%")

    df_step4 = pd.DataFrame(rows)
    out_file = os.path.join(OUT_DIR, 'like_for_like_placebo_adf.csv')
    df_step4.to_csv(out_file, index=False)
    print(f"\nSaved like-for-like placebo table to {out_file}")
    return df_step4


def main():
    print("Loading benchmark data...")
    df_works = pd.read_csv('data/benchmark/benchmark_works.csv')
    df_scores = pd.read_csv('data/benchmark/detector_scores_v2.csv')

    # Step 3
    run_step3_disjoint_null(df_works, df_scores)

    # Step 4
    run_step4_like_for_like_placebo(df_works)

    print("\n" + "=" * 80)
    print("ALL FIX5 EVALUATION STEPS COMPLETED SUCCESSFULLY!")
    print(f"Artifacts saved in {OUT_DIR}/")
    print("=" * 80)


if __name__ == '__main__':
    main()
