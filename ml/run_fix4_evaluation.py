"""
ml/run_fix4_evaluation.py

Fix 4 Evaluation Pipeline:
1. Housekeeping checks (reported via git/grep)
2. Matched-host null sanity check (1,000 random draws per typology: positives = k random works from matched pool, negatives = remaining pool)
   and matched real works vs all 15,000 real works (B and F).
3. Pre-registered rule applied to existing matched-host results:
   detected = AUROC > 0.5 AND bootstrap CI lower bound > 0.5 AND Holm p < 0.05 AND at least 8 valid episodes.
4. Score decomposition: share > 0 for injected vs matched real, and AUROC among active works (score > 0).
5. Placebo control for typologies A, D, F (community scorer only):
   amounts replaced by empirical draws from host IDA (or global if < 15), 20 draws per episode.
6. Review-budget recall table from detector_scores_v2.csv at top 1%, 2%, 5%, 10% under kind='stable' (real-first ties).
7. Save all tables as CSV in data/benchmark/fix4/.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Ensure ml/ is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from community_anomaly_scorer import compute_community_scores

OUT_DIR = os.path.join('data', 'benchmark', 'fix4')
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


def run_step2_null_sanity(df_works, df_scores):
    print("=" * 80)
    print("STEP 2: NULL SANITY CHECK")
    print("=" * 80)

    df_works = df_works.copy()
    df_works['sanc_dt'] = pd.to_datetime(df_works['Sanction Date'])
    window_delta = np.timedelta64(30, 'D')

    syn = df_works[df_works['is_synthetic_fraud'] == True]
    real = df_works[df_works['is_synthetic_fraud'] == False]
    comm_scores_aug = df_scores['community_score'].values

    typologies = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    # 2b: Positives = k random works from matched pool, Negatives = remaining pool works
    print("\n--- Step 2b: 1,000 Null Draws from Matched Pool (Expect 0.50 +- 0.02) ---")
    results_2b = []
    rng = np.random.RandomState(42)

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
            cand_indices = cand.index.tolist()
            if len(cand_indices) >= k:
                valid_ep_ids.append(ep_id)
                ep_dict[ep_id] = {'k': k, 'cand': cand_indices}

        all_matched = set()
        total_k = 0
        for ep_id in valid_ep_ids:
            all_matched.update(ep_dict[ep_id]['cand'])
            total_k += ep_dict[ep_id]['k']

        matched_pool = np.array(sorted(list(all_matched)))
        n_pool = len(matched_pool)
        n_neg = n_pool - total_k

        aurocs = []
        for _ in range(1000):
            pos_idx = rng.choice(matched_pool, size=total_k, replace=False)
            pos_set = set(pos_idx)
            neg_idx = np.array([idx for idx in matched_pool if idx not in pos_set])

            s_pos = comm_scores_aug[pos_idx]
            s_neg = np.sort(comm_scores_aug[neg_idx])
            aurocs.append(fast_auroc(s_pos, s_neg))

        m_auc = float(np.mean(aurocs))
        s_auc = float(np.std(aurocs))
        min_auc = float(np.min(aurocs))
        max_auc = float(np.max(aurocs))
        sanity_pass = abs(m_auc - 0.50) <= 0.02

        print(f"Typology {typ}: Valid Eps={len(valid_ep_ids)}/{sub_syn['episode_id'].nunique()}, "
              f"Pool={n_pool}, k={total_k}, Neg={n_neg} | Mean AUROC = {m_auc:.4f} +- {s_auc:.4f} "
              f"[{min_auc:.4f}, {max_auc:.4f}] | Pass: {sanity_pass}")

        results_2b.append({
            'Typology': typ,
            'Valid_Episodes': f"{len(valid_ep_ids)}/{sub_syn['episode_id'].nunique()}",
            'Matched_Pool_Size': n_pool,
            'Positives_k': total_k,
            'Negatives_N_minus_k': n_neg,
            'Mean_AUROC': m_auc,
            'Std_AUROC': s_auc,
            'Min_AUROC': min_auc,
            'Max_AUROC': max_auc,
            'Sanity_Pass': sanity_pass
        })

    df_2b = pd.DataFrame(results_2b)
    df_2b.to_csv(os.path.join(OUT_DIR, 'null_sanity_2b.csv'), index=False)

    # Sanity assertion
    for r in results_2b:
        if not r['Sanity_Pass']:
            raise ValueError(f"Step 2b SANITY FAILURE: Typology {r['Typology']} mean AUROC {r['Mean_AUROC']:.4f} outside [0.48, 0.52]!")

    # 2c: B and F matched real works vs all 15,000 real works
    print("\n--- Step 2c: B and F Matched Real Works vs All 15,000 Real Works ---")
    results_2c = []
    real_indices = np.arange(15000)

    for typ in ['B', 'F']:
        sub_syn = syn[syn['typology'] == typ]
        all_matched = set()
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
                all_matched.update(cand.index.tolist())

        matched_indices = np.array(sorted(list(all_matched)))
        s_matched = comm_scores_aug[matched_indices]
        s_all_real = np.sort(comm_scores_aug[real_indices])

        non_matched_indices = np.array([i for i in range(15000) if i not in set(matched_indices)])
        s_non_matched = np.sort(comm_scores_aug[non_matched_indices])

        auc_vs_all = fast_auroc(s_matched, s_all_real)
        auc_vs_non_matched = fast_auroc(s_matched, s_non_matched)

        print(f"Typology {typ} (Matched Real n={len(matched_indices)}):")
        print(f"  AUROC vs all 15,000 real works:        {auc_vs_all:.4f}")
        print(f"  AUROC vs non-matched real works (n={len(non_matched_indices)}): {auc_vs_non_matched:.4f}")

        results_2c.append({
            'Typology': typ,
            'Matched_Real_Works_N': len(matched_indices),
            'AUROC_vs_All_15k_Real': auc_vs_all,
            'Non_Matched_Real_Works_N': len(non_matched_indices),
            'AUROC_vs_NonMatched_Real': auc_vs_non_matched
        })

    df_2c = pd.DataFrame(results_2c)
    df_2c.to_csv(os.path.join(OUT_DIR, 'matched_real_vs_all_2c.csv'), index=False)
    return df_2b, df_2c


def run_step3_preregistered_rule():
    print("=" * 80)
    print("STEP 3: PRE-REGISTERED RULE APPLICATION")
    print("Rule: detected = AUROC > 0.5 AND bootstrap CI lower bound > 0.5 AND Holm p < 0.05 AND valid_episodes >= 8")
    print("=" * 80)

    in_file = os.path.join('data', 'benchmark', 'fix3', 'matched_host_null_results.csv')
    df = pd.read_csv(in_file)

    def apply_rule(row):
        valid_eps_count = int(str(row['Valid_Episodes']).split('/')[0])
        cond_auroc = float(row['Actual_AUROC']) > 0.5
        cond_ci = float(row['Bootstrap_CI_Lower']) > 0.5
        cond_holm = float(row['Holm_p']) < 0.05
        cond_eps = valid_eps_count >= 8
        detected = cond_auroc and cond_ci and cond_holm and cond_eps
        return pd.Series({
            'Rule_AUROC_gt_05': cond_auroc,
            'Rule_CI_Lower_gt_05': cond_ci,
            'Rule_Holm_p_lt_005': cond_holm,
            'Rule_Valid_Eps_gte_8': cond_eps,
            'Detected_Preregistered': 'YES' if detected else 'NO'
        })

    eval_df = df.apply(apply_rule, axis=1)
    df_relabeled = pd.concat([df, eval_df], axis=1)

    print("\nRelabeled Results (Augmented Run):")
    cols_show = ['Typology', 'Valid_Episodes', 'Actual_AUROC', 'Bootstrap_CI_Lower', 'Holm_p',
                 'Rule_AUROC_gt_05', 'Rule_CI_Lower_gt_05', 'Rule_Holm_p_lt_005', 'Rule_Valid_Eps_gte_8',
                 'Detected_Adj_p_lt_005', 'Detected_Preregistered']
    print(df_relabeled[cols_show].to_string(index=False))

    out_file = os.path.join(OUT_DIR, 'matched_host_preregistered_rule.csv')
    df_relabeled.to_csv(out_file, index=False)
    print(f"Saved relabeled table to {out_file}")

    # Also apply to real-only run for complete reference
    in_real_only = os.path.join('data', 'benchmark', 'fix3', 'matched_host_null_real_only.csv')
    if os.path.exists(in_real_only):
        df_ro = pd.read_csv(in_real_only)
        eval_ro = df_ro.apply(apply_rule, axis=1)
        df_ro_relabeled = pd.concat([df_ro, eval_ro], axis=1)
        out_ro = os.path.join(OUT_DIR, 'matched_host_preregistered_rule_real_only.csv')
        df_ro_relabeled.to_csv(out_ro, index=False)
        print(f"Saved real-only relabeled table to {out_ro}")

    return df_relabeled


def run_step4_decomposition(df_works, df_scores):
    print("=" * 80)
    print("STEP 4: SCORE DECOMPOSITION (AUGMENTED SCORES, RETAINED EPISODES)")
    print("=" * 80)

    df_works = df_works.copy()
    df_works['sanc_dt'] = pd.to_datetime(df_works['Sanction Date'])
    window_delta = np.timedelta64(30, 'D')

    syn = df_works[df_works['is_synthetic_fraud'] == True]
    real = df_works[df_works['is_synthetic_fraud'] == False]
    comm_scores = df_scores['community_score'].values

    typologies = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    rows = []

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
            if len(cand) >= k:
                valid_ep_ids.append(ep_id)
                ep_dict[ep_id] = {'inj_idx': grp.index.tolist(), 'cand_idx': cand.index.tolist()}

        retained_inj_idx = [idx for ep in valid_ep_ids for idx in ep_dict[ep]['inj_idx']]
        all_matched = set()
        for ep in valid_ep_ids:
            all_matched.update(ep_dict[ep]['cand_idx'])
        matched_real_idx = sorted(list(all_matched))

        s_inj = comm_scores[retained_inj_idx]
        s_matched = comm_scores[matched_real_idx]

        inj_gt0 = (s_inj > 0)
        matched_gt0 = (s_matched > 0)

        share_inj_gt0 = float(np.mean(inj_gt0)) if len(inj_gt0) > 0 else 0.0
        share_matched_gt0 = float(np.mean(matched_gt0)) if len(matched_gt0) > 0 else 0.0

        s_inj_pos = s_inj[inj_gt0]
        s_matched_pos = s_matched[matched_gt0]

        n_inj_pos = len(s_inj_pos)
        n_matched_pos = len(s_matched_pos)

        if n_inj_pos > 0 and n_matched_pos > 0:
            y_pos = np.array([1] * n_inj_pos + [0] * n_matched_pos)
            scores_pos = np.concatenate([s_inj_pos, s_matched_pos])
            if len(np.unique(scores_pos)) > 1:
                auc_gt0 = float(roc_auc_score(y_pos, scores_pos))
                auc_str = f"{auc_gt0:.4f}"
            else:
                auc_gt0 = 0.5
                auc_str = "0.5000"
        else:
            auc_gt0 = np.nan
            auc_str = "N/A"

        rows.append({
            'Typology': typ,
            'Valid_Episodes': f"{len(valid_ep_ids)}/{sub_syn['episode_id'].nunique()}",
            'Retained_Inj_Works': len(retained_inj_idx),
            'Inj_Share_GT_0': share_inj_gt0,
            'Inj_Count_GT_0': n_inj_pos,
            'Matched_Real_Works': len(matched_real_idx),
            'Matched_Real_Share_GT_0': share_matched_gt0,
            'Matched_Real_Count_GT_0': n_matched_pos,
            'AUROC_Among_Score_GT_0': auc_str
        })

    df_decomp = pd.DataFrame(rows)
    print(df_decomp.to_string(index=False))

    out_file = os.path.join(OUT_DIR, 'community_score_decomposition.csv')
    df_decomp.to_csv(out_file, index=False)
    print(f"\nSaved decomposition table to {out_file}")
    return df_decomp


def run_step5_placebo_control(df_works, df_scores):
    print("=" * 80)
    print("STEP 5: PLACEBO CONTROL (TYPOLOGIES A, D, F, COMMUNITY SCORER ONLY)")
    print("Per retained episode: keep MP, IDA, sanction dates, lags, count;")
    print("replace amounts with draw from host IDA real amounts (global if < 15). 20 placebo draws.")
    print("=" * 80)

    df_works = df_works.copy()
    df_works['sanc_dt'] = pd.to_datetime(df_works['Sanction Date'])
    window_delta = np.timedelta64(30, 'D')

    syn = df_works[df_works['is_synthetic_fraud'] == True]
    real = df_works[df_works['is_synthetic_fraud'] == False]
    comm_scores_aug = df_scores['community_score'].values

    # Pre-extract real amounts per IDA and global
    real_amts_by_ida = {}
    for ida, g in real.groupby('IDA'):
        real_amts_by_ida[ida] = g['sanction_amount'].values
    global_real_amts = real['sanction_amount'].values

    cols_needed = ['IDA', 'MP', 'sanction_amount', 'Sanction Date']
    base_real_df = real[cols_needed].copy().reset_index(drop=True)
    n_real = len(base_real_df)

    rows = []
    rng = np.random.RandomState(42)
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
                ep_data[ep_id] = {
                    'df': grp.copy(),
                    'cand_idx': cand.index.tolist()
                }

        retained_inj_orig_idx = [idx for ep in valid_ep_ids for idx in ep_data[ep]['df'].index.tolist()]
        all_matched = set()
        for ep in valid_ep_ids:
            all_matched.update(ep_data[ep]['cand_idx'])
        matched_real_idx = sorted(list(all_matched))

        # True metrics
        true_inj_scores = comm_scores_aug[retained_inj_orig_idx]
        true_matched_scores = comm_scores_aug[matched_real_idx]

        true_share_gt0 = float(np.mean(true_inj_scores > 0))
        true_mean_score = float(np.mean(true_inj_scores))
        y_true = np.array([1] * len(true_inj_scores) + [0] * len(true_matched_scores))
        s_true = np.concatenate([true_inj_scores, true_matched_scores])
        true_auroc = float(roc_auc_score(y_true, s_true))

        # Placebo draws
        placebo_shares = []
        placebo_means = []
        placebo_aurocs = []

        for draw in range(N_PLACEBO):
            placebo_ep_dfs = []
            for ep in valid_ep_ids:
                df_ep = ep_data[ep]['df'].copy()
                ep_ida = df_ep['IDA'].iloc[0]
                pool = real_amts_by_ida.get(ep_ida, global_real_amts)
                if len(pool) < 15:
                    pool = global_real_amts
                sampled_amts = rng.choice(pool, size=len(df_ep), replace=True)
                df_ep['sanction_amount'] = sampled_amts
                placebo_ep_dfs.append(df_ep[cols_needed])

            df_placebo_eps = pd.concat(placebo_ep_dfs, ignore_index=True)
            df_combined = pd.concat([base_real_df, df_placebo_eps], ignore_index=True)

            scores_comb, _ = compute_community_scores(df_combined, window_days=30, eps=0.001)

            scores_placebo = scores_comb[n_real:]
            scores_matched_real = scores_comb[matched_real_idx]

            p_share = float(np.mean(scores_placebo > 0))
            p_mean = float(np.mean(scores_placebo))

            y_p = np.array([1] * len(scores_placebo) + [0] * len(scores_matched_real))
            s_p = np.concatenate([scores_placebo, scores_matched_real])
            p_auc = float(roc_auc_score(y_p, s_p))

            placebo_shares.append(p_share)
            placebo_means.append(p_mean)
            placebo_aurocs.append(p_auc)

        row = {
            'Typology': typ,
            'Valid_Episodes': f"{len(valid_ep_ids)}/{sub_syn['episode_id'].nunique()}",
            'Retained_Works': len(retained_inj_orig_idx),
            'Matched_Real_Works': len(matched_real_idx),
            'True_Share_GT_0': true_share_gt0,
            'True_Mean_Score': true_mean_score,
            'True_AUROC_vs_Matched': true_auroc,
            'Placebo_Share_GT_0_Mean': float(np.mean(placebo_shares)),
            'Placebo_Share_GT_0_Std': float(np.std(placebo_shares)),
            'Placebo_Mean_Score_Mean': float(np.mean(placebo_means)),
            'Placebo_Mean_Score_Std': float(np.std(placebo_means)),
            'Placebo_AUROC_Mean': float(np.mean(placebo_aurocs)),
            'Placebo_AUROC_Std': float(np.std(placebo_aurocs))
        }
        rows.append(row)

        print(f"\nTypology {typ}:")
        print(f"  True:    Share>0 = {row['True_Share_GT_0']*100:.2f}%, Mean Score = {row['True_Mean_Score']:.4f}, AUROC = {row['True_AUROC_vs_Matched']:.4f}")
        print(f"  Placebo: Share>0 = {row['Placebo_Share_GT_0_Mean']*100:.2f}% +- {row['Placebo_Share_GT_0_Std']*100:.2f}%, "
              f"Mean Score = {row['Placebo_Mean_Score_Mean']:.4f} +- {row['Placebo_Mean_Score_Std']:.4f}, "
              f"AUROC = {row['Placebo_AUROC_Mean']:.4f} +- {row['Placebo_AUROC_Std']:.4f}")

    df_placebo = pd.DataFrame(rows)
    out_file = os.path.join(OUT_DIR, 'placebo_control_adf.csv')
    df_placebo.to_csv(out_file, index=False)
    print(f"\nSaved placebo control table to {out_file}")
    return df_placebo


def run_step6_review_budget(df_scores):
    print("=" * 80)
    print("STEP 6: REVIEW BUDGET TABLE (DETECTOR_SCORES_V2.CSV)")
    print("Recall of injected works at top 1%, 2%, 5%, 10% of 15,430 works (kind='stable', real-first ties)")
    print("=" * 80)

    detectors = {
        'gat_frozen': 'GAT (Frozen)',
        'gat_augmented': 'GAT (Augmented)',
        'community_score': 'Community Scorer',
        'iforest_score': 'Isolation Forest',
        'robust_z_score': 'Robust Z-score',
    }

    typologies = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    N = len(df_scores)
    y = (df_scores['is_synthetic_fraud'] == True).astype(int).values

    cutoffs = {
        'Top 1%': int(round(0.01 * N)),   # 154
        'Top 2%': int(round(0.02 * N)),   # 309
        'Top 5%': int(round(0.05 * N)),   # 772
        'Top 10%': int(round(0.10 * N)),  # 1543
    }

    rows = []

    for det_col, det_name in detectors.items():
        s = df_scores[det_col].values

        # Real-first tie breaking with stable sort:
        # np.lexsort((y, -s)) sorts by -s (descending), breaking ties by y (0 before 1)
        order = np.lexsort((y, -s))
        top_indices = {c_name: set(order[:k_val]) for c_name, k_val in cutoffs.items()}

        for typ in typologies:
            sub_syn = df_scores[df_scores['typology'] == typ]
            n_inj = len(sub_syn)
            syn_indices = set(sub_syn.index)

            row = {
                'Detector': det_name,
                'Detector_Column': det_col,
                'Typology': typ,
                'Total_Injected': n_inj
            }
            for c_name, k_val in cutoffs.items():
                hits = len(syn_indices.intersection(top_indices[c_name]))
                rec = hits / n_inj if n_inj > 0 else 0.0
                row[f'Hits_{c_name}'] = hits
                row[f'Recall_{c_name}'] = rec
                row[f'Recall_Str_{c_name}'] = f"{rec*100:.2f}% ({hits}/{n_inj})"
            rows.append(row)

        # Total across all typologies
        syn_all = set(df_scores[df_scores['is_synthetic_fraud'] == True].index)
        n_tot_inj = len(syn_all)
        tot_row = {
            'Detector': det_name,
            'Detector_Column': det_col,
            'Typology': 'ALL (Total)',
            'Total_Injected': n_tot_inj
        }
        for c_name, k_val in cutoffs.items():
            hits = len(syn_all.intersection(top_indices[c_name]))
            rec = hits / n_tot_inj if n_tot_inj > 0 else 0.0
            tot_row[f'Hits_{c_name}'] = hits
            tot_row[f'Recall_{c_name}'] = rec
            tot_row[f'Recall_Str_{c_name}'] = f"{rec*100:.2f}% ({hits}/{n_tot_inj})"
        rows.append(tot_row)

    df_budget = pd.DataFrame(rows)
    cols_display = ['Detector', 'Typology', 'Total_Injected', 'Recall_Str_Top 1%', 'Recall_Str_Top 2%', 'Recall_Str_Top 5%', 'Recall_Str_Top 10%']
    print(df_budget[cols_display].to_string(index=False))

    out_file = os.path.join(OUT_DIR, 'review_budget_recall.csv')
    df_budget.to_csv(out_file, index=False)
    print(f"\nSaved review budget table to {out_file}")
    return df_budget


def main():
    print("Loading benchmark data...")
    df_works = pd.read_csv('data/benchmark/benchmark_works.csv')
    df_scores = pd.read_csv('data/benchmark/detector_scores_v2.csv')

    # Step 2
    run_step2_null_sanity(df_works, df_scores)

    # Step 3
    run_step3_preregistered_rule()

    # Step 4
    run_step4_decomposition(df_works, df_scores)

    # Step 5
    run_step5_placebo_control(df_works, df_scores)

    # Step 6
    run_step6_review_budget(df_scores)

    print("\n" + "=" * 80)
    print("ALL FIX4 EVALUATION STEPS COMPLETED SUCCESSFULLY!")
    print(f"Artifacts saved in {OUT_DIR}/")
    print("=" * 80)


if __name__ == '__main__':
    main()
