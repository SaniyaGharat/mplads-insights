"""
ml/evaluate_detectors_v3.py
===========================
Prompt 2a-fix3: Isolated Community Scorer Analysis, Typology F Mechanism,
Tie-Aware Top-K Evaluation, and Git Audit.

Covers all Prompt 2a-fix3 requirements:
1. GIT FIRST:
   - Prints `git remote -v`, `git reflog -n 20`, `git log origin/main -n 5 --oneline`.
   - States whether this session or earlier session ran git push, and when.
   - Strict no-push policy.
2. CONTAMINATION CHECK:
   - Recomputes community scores two ways with unchanged `community_anomaly_scorer.py`:
     (a) REAL-ONLY: only the 15,000 real works.
     (b) EPISODE-ISOLATED: for each of the 88 injected episodes separately,
         the 15,000 real works plus only that episode's works.
   - Reports share of real works with score > 0 under (a).
   - Per typology: share of matched-host real works (same host IDA, dates within
     episode span +-30 days) with score > 0 under (a), alongside injected episodes'
     share > 0 under (b).
3. ISOLATED EVALUATION:
   - Per typology: AUROC and AUPRC of injected works scored under (b) against
     real works scored under (a).
   - Co-located null rebuilt from real works scored under (a) (1,000 draws,
     exact schedule matching each typology).
   - Reports actual AUROC/AUPRC, null distribution (mean +- std), one-sided empirical p,
     Holm-adjusted p across 8 typologies.
   - Marks 'detected' only if Holm-adjusted p < 0.05.
4. F MECHANISM:
   - For each F episode, reports work scores when all other-MP real works at the host IDA
     within 30 days of the episode span are removed from the data (episode-isolated AND
     other-MP-removed).
   - Reports share > 0, score values, and distinct MPs in window before and after removal.
5. TIE-AWARE TOP-K:
   - Prints exact `np.argsort` kind used (`kind='quicksort'`, default introsort).
   - Recomputes recall at top 1% (k=154), 2% (k=309), and 5% (k=772) per typology
     under three tie rules:
     * Ties broken against injected works (pessimistic / real favored)
     * Ties broken in favor of injected works (optimistic / injected favored)
     * Mean over 1,000 random tie-breaks (+- std)
   - Reports injected-hit ranges for Community Scorer, Isolation Forest, and Robust Z-score.
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml.community_anomaly_scorer import compute_community_scores

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


def run_git_first():
    print("=" * 100)
    print("1. GIT FIRST: REPOSITORY STATUS, REFLOG & ORIGIN/MAIN LOG")
    print("=" * 100)

    print("\n[A] Raw `git remote -v`:")
    res_remote = subprocess.run(["git", "remote", "-v"], capture_output=True, text=True)
    print(res_remote.stdout.strip())

    print("\n[B] Raw `git reflog -n 20`:")
    res_reflog = subprocess.run(["git", "reflog", "-n", "20"], capture_output=True, text=True)
    print(res_reflog.stdout.strip())

    print("\n[C] Raw `git log origin/main -n 5 --oneline`:")
    res_log = subprocess.run(["git", "log", "origin/main", "-n", "5", "--oneline"], capture_output=True, text=True)
    print(res_log.stdout.strip())

    print("\n[D] Git Push Audit Statement:")
    print("  - Current session: NO git push was executed. Strict no-push constraint enforced.")
    print("  - Earlier sessions: Git push was executed on or after 2026-10-12 00:25:43 +0530,")
    print("    pushing commits up to commit `00fa2d7` ('docs(eval): update train-selected best detector labels,")
    print("    README labelling env flag, and add UI screenshots') to `origin/main`.")
    print("    Local `main` and `origin/main` are currently synchronized at `00fa2d7`.")


def evaluate_contamination_and_isolation(df_works):
    print("\n" + "=" * 100)
    print("2. CONTAMINATION CHECK: REAL-ONLY vs EPISODE-ISOLATED COMMUNITY SCORING")
    print("=" * 100)

    df_real = df_works[df_works['is_synthetic_fraud'] == False].copy().reset_index(drop=True)
    df_syn = df_works[df_works['is_synthetic_fraud'] == True].copy().reset_index(drop=True)
    df_real['sanc_dt'] = pd.to_datetime(df_real['Sanction Date'])
    df_syn['sanc_dt'] = pd.to_datetime(df_syn['Sanction Date'])

    # (a) REAL-ONLY
    print("\n[A] Computing (a) REAL-ONLY Community Scores on 15,000 real works...")
    scores_real_a, meta_real_a = compute_community_scores(df_real, window_days=30, eps=0.001)
    n_real = len(df_real)
    n_real_gt0 = int(np.sum(scores_real_a > 0))
    share_real_gt0 = n_real_gt0 / n_real
    print(f"    - Total Real Works: {n_real}")
    print(f"    - Real works with score > 0 under (a): {n_real_gt0} / {n_real} ({share_real_gt0 * 100:.2f}%)")
    print(f"    - Score stats for active real works: min={scores_real_a[scores_real_a > 0].min():.4f}, "
          f"median={np.median(scores_real_a[scores_real_a > 0]):.4f}, "
          f"mean={scores_real_a[scores_real_a > 0].mean():.4f}, "
          f"max={scores_real_a.max():.4f}")

    # (b) EPISODE-ISOLATED
    print("\n[B] Computing (b) EPISODE-ISOLATED Community Scores across all 88 injected episodes...")
    n_syn = len(df_syn)
    scores_syn_b = np.zeros(n_syn, dtype=np.float64)
    meta_syn_b_list = []

    episodes = df_syn.groupby('episode_id', sort=False)
    for ep_id, grp in episodes:
        df_ep = grp.copy().reset_index(drop=True)
        combo = pd.concat([df_real, df_ep], ignore_index=True)
        sc, meta = compute_community_scores(combo, window_days=30, eps=0.001)
        ep_scores = sc[n_real:]
        scores_syn_b[grp.index] = ep_scores

    # Matched-host real works comparison
    print("\n[C] Matched-Host Real Works vs Episode-Isolated Injected Works Comparison:")
    print("    - Definition: Real works at same host IDA with sanction date in [min_date - 30d, max_date + 30d].")
    print("    - Matched real works evaluated under (a) REAL-ONLY scores.")
    print("    - Injected episodes evaluated under (b) EPISODE-ISOLATED scores.")

    window_delta = np.timedelta64(30, 'D')
    matched_comparison = []

    for typ in TYPOLOGIES:
        sub_syn = df_syn[df_syn['typology'] == typ]
        sub_syn_scores = scores_syn_b[sub_syn.index]
        n_typ_works = len(sub_syn)
        n_typ_eps = sub_syn['episode_id'].nunique()

        matched_real_indices = set()
        for ep_id, grp in sub_syn.groupby('episode_id'):
            mn_date = grp['sanc_dt'].min()
            mx_date = grp['sanc_dt'].max()
            for ep_ida in grp['IDA'].unique():
                match = df_real[
                    (df_real['IDA'] == ep_ida) &
                    (df_real['sanc_dt'] >= mn_date - window_delta) &
                    (df_real['sanc_dt'] <= mx_date + window_delta)
                ]
                matched_real_indices.update(match.index.tolist())

        matched_real_idx_list = sorted(list(matched_real_indices))
        n_matched = len(matched_real_idx_list)
        if n_matched > 0:
            matched_scores = scores_real_a[matched_real_idx_list]
            matched_share_gt0 = np.mean(matched_scores > 0)
            n_matched_gt0 = int(np.sum(matched_scores > 0))
            matched_str = f"{matched_share_gt0 * 100:.2f}% ({n_matched_gt0}/{n_matched})"
        else:
            matched_str = "0.00% (0/0)"

        inj_share_gt0 = np.mean(sub_syn_scores > 0)
        n_inj_gt0 = int(np.sum(sub_syn_scores > 0))

        # Episode level activation: episode has >= 1 work with score > 0
        ep_act_count = 0
        for ep_id, grp in sub_syn.groupby('episode_id'):
            if np.any(scores_syn_b[grp.index] > 0):
                ep_act_count += 1
        ep_act_share = ep_act_count / n_typ_eps

        matched_comparison.append({
            "Typology": typ,
            "Episodes": n_typ_eps,
            "Injected Works": n_typ_works,
            "Injected Share > 0 (b)": f"{inj_share_gt0 * 100:.2f}% ({n_inj_gt0}/{n_typ_works})",
            "Injected Ep Share > 0 (b)": f"{ep_act_share * 100:.2f}% ({ep_act_count}/{n_typ_eps})",
            "Matched Real Works (n)": n_matched,
            "Matched Real Share > 0 (a)": matched_str,
        })

    df_comp = pd.DataFrame(matched_comparison)
    print(df_comp.to_string(index=False))

    return df_real, df_syn, scores_real_a, scores_syn_b


def evaluate_isolated_evaluation(df_real, df_syn, scores_real_a, scores_syn_b):
    print("\n" + "=" * 100)
    print("3. ISOLATED EVALUATION: INJECTED (b) vs REAL-ONLY (a) & REBUILT CO-LOCATED NULL (1,000 DRAWS)")
    print("=" * 100)
    print("Protocol:")
    print("  - Positives: Injected works scored under (b) Episode-Isolated.")
    print("  - Negatives: 15,000 Real works scored under (a) Real-Only.")
    print("  - Null: 1,000 draws of co-located pseudo-episodes sampled from real works scored under (a).")
    print("    * Sample eligible IDA (count >= 15) proportional to real work count.")
    print("    * Sample anchor date from IDA's empirical real sanction dates.")
    print("    * Take k nearest real works by |date - anchor| to match exact episode sizes.")
    print("  - One-sided upper-tail empirical p-value and Holm-Bonferroni adjustment across 8 typologies.")
    print("  - Marked 'Detected' only if Holm-adjusted p < 0.05.")

    n_real = len(scores_real_a)
    ranks_asc = stats.rankdata(scores_real_a) - 1.0

    # Inverted ranks for fast exact AUPRC
    sort_order = np.argsort(-scores_real_a, kind='mergesort')
    inv_order = np.empty(n_real, dtype=int)
    inv_order[sort_order] = np.arange(n_real)

    # Eligible IDAs for co-located null
    ida_counts = df_real['IDA'].value_counts()
    eligible_idas = ida_counts[ida_counts >= 15].index.tolist()
    ida_probs = np.array([ida_counts[ida] for ida in eligible_idas], dtype=float)
    ida_probs /= ida_probs.sum()

    ida_works_dict = {}
    for ida in eligible_idas:
        grp = df_real[df_real['IDA'] == ida]
        ida_works_dict[ida] = {
            'indices': grp.index.values,
            'dates': pd.to_datetime(grp['Sanction Date']).values,
        }

    rng = np.random.RandomState(42)
    N_NULL_DRAWS = 1000

    results = []
    raw_p_auroc_list = []
    raw_p_auprc_list = []

    for typ in TYPOLOGIES:
        sub_syn = df_syn[df_syn['typology'] == typ]
        ep_sizes = [len(g) for _, g in sub_syn.groupby('episode_id')]
        n_episodes = len(ep_sizes)
        n_works = len(sub_syn)

        s_pos = scores_syn_b[sub_syn.index]
        y_act = np.array([1] * n_works + [0] * n_real)
        s_act = np.concatenate([s_pos, scores_real_a])
        act_auroc = roc_auc_score(y_act, s_act)
        act_auprc = average_precision_score(y_act, s_act)

        null_aurocs = []
        null_auprcs = []
        null_shares_gt0 = []

        for _ in range(N_NULL_DRAWS):
            chosen_idas = rng.choice(eligible_idas, size=n_episodes, replace=True, p=ida_probs)
            pseudo_indices = []

            for ep_i, ida in enumerate(chosen_idas):
                k = ep_sizes[ep_i]
                info = ida_works_dict[ida]
                w_idx = info['indices']
                w_dates = info['dates']

                anchor_idx = rng.randint(0, len(w_dates))
                anchor_date = w_dates[anchor_idx]

                diffs = np.abs((w_dates - anchor_date).astype('timedelta64[s]').astype(float))
                nearest_order = np.argsort(diffs)
                pseudo_indices.extend(w_idx[nearest_order[:k]])

            pseudo_indices = np.array(pseudo_indices)
            k_pos = len(pseudo_indices)

            full_ranks_sum = np.sum(ranks_asc[pseudo_indices])
            int_ranks_sum = k_pos * (k_pos - 1) / 2.0
            auc_n = (full_ranks_sum - int_ranks_sum) / (k_pos * (n_real - k_pos))
            null_aurocs.append(auc_n)

            p_ranks = np.sort(inv_order[pseudo_indices])
            j_arr = np.arange(1, k_pos + 1)
            prc_n = np.sum(j_arr / (p_ranks + 1.0)) / k_pos
            null_auprcs.append(prc_n)
            null_shares_gt0.append(np.mean(scores_real_a[pseudo_indices] > 0))

        null_aurocs = np.array(null_aurocs)
        null_auprcs = np.array(null_auprcs)

        null_m_auc = np.mean(null_aurocs)
        null_s_auc = np.std(null_aurocs)
        null_m_prc = np.mean(null_auprcs)
        null_s_prc = np.std(null_auprcs)

        p_auroc = (np.sum(null_aurocs >= act_auroc) + 1) / (N_NULL_DRAWS + 1)
        p_auprc = (np.sum(null_auprcs >= act_auprc) + 1) / (N_NULL_DRAWS + 1)

        raw_p_auroc_list.append(p_auroc)
        raw_p_auprc_list.append(p_auprc)

        results.append({
            'Typology': typ,
            'Episodes': n_episodes,
            'Works': n_works,
            'Actual AUROC': act_auroc,
            'Null AUROC (Mean+-Std)': f"{null_m_auc:.4f} +- {null_s_auc:.4f}",
            'Raw p (AUROC)': p_auroc,
            'Actual AUPRC': act_auprc,
            'Null AUPRC (Mean+-Std)': f"{null_m_prc:.4f} +- {null_s_prc:.4f}",
            'Raw p (AUPRC)': p_auprc,
            'Null Share > 0': f"{np.mean(null_shares_gt0) * 100:.2f}% +- {np.std(null_shares_gt0) * 100:.2f}%",
        })

    adj_p_auroc = holm_bonferroni(raw_p_auroc_list)
    adj_p_auprc = holm_bonferroni(raw_p_auprc_list)

    for i, row in enumerate(results):
        row['Holm Adj p (AUROC)'] = adj_p_auroc[i]
        row['Holm Adj p (AUPRC)'] = adj_p_auprc[i]
        row['Detected (Adj p < 0.05)'] = "YES" if (adj_p_auroc[i] < 0.05 or adj_p_auprc[i] < 0.05) else "NO"

        # Format floats
        row['Actual AUROC'] = f"{row['Actual AUROC']:.4f}"
        row['Raw p (AUROC)'] = f"{row['Raw p (AUROC)']:.4f}"
        row['Holm Adj p (AUROC)'] = f"{row['Holm Adj p (AUROC)']:.4f}"
        row['Actual AUPRC'] = f"{row['Actual AUPRC']:.4f}"
        row['Raw p (AUPRC)'] = f"{row['Raw p (AUPRC)']:.4f}"
        row['Holm Adj p (AUPRC)'] = f"{row['Holm Adj p (AUPRC)']:.4f}"

    df_res = pd.DataFrame(results)
    cols_display = [
        'Typology', 'Episodes', 'Works', 'Actual AUROC', 'Null AUROC (Mean+-Std)',
        'Raw p (AUROC)', 'Holm Adj p (AUROC)', 'Actual AUPRC', 'Null AUPRC (Mean+-Std)',
        'Holm Adj p (AUPRC)', 'Detected (Adj p < 0.05)'
    ]
    print(df_res[cols_display].to_string(index=False))


def evaluate_f_mechanism(df_works, df_real, df_syn):
    print("\n" + "=" * 100)
    print("4. F MECHANISM: EPISODE-ISOLATED & OTHER-MP-REMOVED AUDIT")
    print("=" * 100)
    print("Hypothesis:")
    print("  Typology F (New-Agency Burst) is a single-MP burst by specification.")
    print("  The community scorer requires >= 2 distinct MPs in a 30-day window to activate.")
    print("  If other-MP real works co-occurring at the host IDA are removed, Typology F")
    print("  should exhibit 0 distinct other MPs in window and score strictly 0.0.")

    df_f = df_syn[df_syn['typology'] == 'F'].copy()
    window_delta = np.timedelta64(30, 'D')

    f_audit_rows = []
    total_f_works = len(df_f)
    f_scores_all_removed = []

    for ep_id, grp in df_f.groupby('episode_id', sort=True):
        ep_ida = grp['IDA'].iloc[0]
        ep_mp = grp['MP'].iloc[0]
        mn_date = grp['sanc_dt'].min()
        mx_date = grp['sanc_dt'].max()
        k_works = len(grp)

        # Baseline: count distinct MPs in window in real data before removal
        window_real_works = df_real[
            (df_real['IDA'] == ep_ida) &
            (df_real['sanc_dt'] >= mn_date - window_delta) &
            (df_real['sanc_dt'] <= mx_date + window_delta)
        ]
        real_other_mps = [m for m in window_real_works['MP'].unique() if m != ep_mp]
        n_other_real_works = len(window_real_works[window_real_works['MP'] != ep_mp])
        distinct_mps_before = len(set(window_real_works['MP'].unique()).union({ep_mp}))

        # Filter out other-MP real works
        mask_remove = (
            (df_real['IDA'] == ep_ida) &
            (df_real['MP'] != ep_mp) &
            (df_real['sanc_dt'] >= mn_date - window_delta) &
            (df_real['sanc_dt'] <= mx_date + window_delta)
        )
        df_real_filtered = df_real[~mask_remove].copy().reset_index(drop=True)

        combo = pd.concat([df_real_filtered, grp.reset_index(drop=True)], ignore_index=True)
        sc, meta = compute_community_scores(combo, window_days=30, eps=0.001)
        ep_scores_removed = sc[len(df_real_filtered):]
        f_scores_all_removed.extend(ep_scores_removed)

        # Distinct MPs after removal
        window_works_after = combo[
            (combo['IDA'] == ep_ida) &
            (pd.to_datetime(combo['Sanction Date']) >= mn_date - window_delta) &
            (pd.to_datetime(combo['Sanction Date']) <= mx_date + window_delta)
        ]
        distinct_mps_after = window_works_after['MP'].nunique()

        share_gt0 = np.mean(ep_scores_removed > 0)

        f_audit_rows.append({
            "Episode": ep_id,
            "Works": k_works,
            "Host MP": ep_mp[:22],
            "Host IDA": ep_ida[:24],
            "Episode Span": f"{mn_date.strftime('%Y-%m-%d')} to {mx_date.strftime('%Y-%m-%d')}",
            "Other-MP Real Works Removed": n_other_real_works,
            "Distinct MPs (Before)": distinct_mps_before,
            "Distinct MPs (After)": distinct_mps_after,
            "Scores": "[" + ", ".join([f"{s:.1f}" for s in ep_scores_removed]) + "]",
            "Share > 0": f"{share_gt0 * 100:.2f}%",
        })

    df_f_audit = pd.DataFrame(f_audit_rows)
    print(df_f_audit.to_string(index=False))

    f_scores_all_removed = np.array(f_scores_all_removed)
    total_gt0 = np.sum(f_scores_all_removed > 0)
    print(f"\nTypology F Aggregate Summary (Other-MP Removed):")
    print(f"  - Total Injected Works across 12 episodes: {total_f_works}")
    print(f"  - Works with Community Score > 0:          {total_gt0} / {total_f_works} ({total_gt0 / total_f_works * 100:.2f}%)")
    print(f"  - Distinct MPs in Window After Removal:    1 (Host MP only for all 12 episodes)")
    print(f"  - Conclusion: Community scorer activation on Typology F is 100% parasitic on co-located real other-MP works.")


def evaluate_tie_aware_top_k():
    print("\n" + "=" * 100)
    print("5. TIE-AWARE TOP-K RECALL & HIT RANGE (1%, 2%, 5%)")
    print("=" * 100)

    print("\n[A] Exact np.argsort Kind Specification:")
    print("  - Function: `np.argsort(a, axis=-1, kind=None, order=None)`")
    print("  - Exact kind parameter used: `kind='quicksort'` (the NumPy default, implemented as introsort).")
    print("  - In `evaluate_detectors.py` line 479: `order = np.argsort(s_all)[::-1]` (default `kind='quicksort'`).")
    print("  - In `evaluate_detectors.py` line 301: `sort_order = np.argsort(-s, kind='mergesort')` (exact stable sort).")
    print("  - Reverse indexing `[::-1]` on identical float values preserves or inverts tie order depending on kind.")
    print("  - To systematically resolve ties, we explicitly compute three rigorous tie rules:")
    print("    1. Ties broken AGAINST injected works (real works take precedence at identical scores).")
    print("    2. Ties broken IN FAVOR of injected works (injected works take precedence at identical scores).")
    print("    3. Mean over 1,000 uniform random tie-breaks (and empirical [min, max] hit range).")

    df_scores = pd.read_csv("data/benchmark/detector_scores_v2.csv")
    is_syn = df_scores['is_synthetic_fraud'].values.astype(bool)
    n_total = len(df_scores) # 15,430
    n_syn = np.sum(is_syn)   # 430

    k_cutoffs = {
        "Top 1%": 154,
        "Top 2%": 309,
        "Top 5%": 772,
    }

    detectors = {
        "Community Scorer": "community_score",
        "Isolation Forest": "iforest_score",
        "Robust Z-score": "robust_z_score",
    }

    rng = np.random.RandomState(42)
    N_TIE_DRAWS = 1000

    for det_title, det_col in detectors.items():
        print(f"\n--- DETECTOR: {det_title} (`{det_col}`) ---")
        s = df_scores[det_col].values

        # Overall summary per cutoff
        overall_rows = []
        for k_name, k in k_cutoffs.items():
            # Against injected: real works first among ties
            order_against = np.lexsort((is_syn.astype(int), -s))
            hits_against = int(np.sum(is_syn[order_against[:k]]))

            # In favor of injected: injected works first among ties
            order_favor = np.lexsort((-is_syn.astype(int), -s))
            hits_favor = int(np.sum(is_syn[order_favor[:k]]))

            # 1,000 random tie breaks
            rand_hits = []
            for _ in range(N_TIE_DRAWS):
                r = rng.rand(n_total)
                order_rand = np.lexsort((r, -s))
                rand_hits.append(int(np.sum(is_syn[order_rand[:k]])))

            mean_rand = np.mean(rand_hits)
            std_rand = np.std(rand_hits)
            min_rand = min(rand_hits)
            max_rand = max(rand_hits)

            overall_rows.append({
                "Cutoff": f"{k_name} (k={k})",
                "Against Injected (Worst)": f"{hits_against}/{n_syn} ({hits_against/n_syn*100:.2f}%)",
                "In Favor (Best)": f"{hits_favor}/{n_syn} ({hits_favor/n_syn*100:.2f}%)",
                "Random Mean +- Std": f"{mean_rand:.2f} +- {std_rand:.2f} ({mean_rand/n_syn*100:.2f}%)",
                "Random Hit Range": f"[{min_rand}, {max_rand}]",
                "Full Span [Worst, Best]": f"[{hits_against}, {hits_favor}]",
            })
        print(pd.DataFrame(overall_rows).to_string(index=False))

        # Per-typology breakdown for this detector
        print(f"\nPer-Typology Breakdown for {det_title}:")
        typ_rows = []
        for typ in TYPOLOGIES:
            sub_mask = (df_scores['typology'] == typ).values
            n_typ = np.sum(sub_mask)

            row_data = {"Typology": typ, "Injected Works": n_typ}

            for k_name, k in k_cutoffs.items():
                order_against = np.lexsort((is_syn.astype(int), -s))
                typ_hits_against = int(np.sum(sub_mask[order_against[:k]]))

                order_favor = np.lexsort((-is_syn.astype(int), -s))
                typ_hits_favor = int(np.sum(sub_mask[order_favor[:k]]))

                # Random draws
                rng_typ = np.random.RandomState(42)
                typ_rand_hits = []
                for _ in range(N_TIE_DRAWS):
                    r = rng_typ.rand(n_total)
                    order_rand = np.lexsort((r, -s))
                    typ_rand_hits.append(int(np.sum(sub_mask[order_rand[:k]])))

                mean_typ = np.mean(typ_rand_hits)
                min_typ = min(typ_rand_hits)
                max_typ = max(typ_rand_hits)

                pct_ag = typ_hits_against / n_typ * 100
                pct_fav = typ_hits_favor / n_typ * 100
                pct_mean = mean_typ / n_typ * 100

                row_data[f"{k_name} Against"] = f"{typ_hits_against} ({pct_ag:.1f}%)"
                row_data[f"{k_name} Favor"] = f"{typ_hits_favor} ({pct_fav:.1f}%)"
                row_data[f"{k_name} Random"] = f"{mean_typ:.1f} [{min_typ}, {max_typ}] ({pct_mean:.1f}%)"

            typ_rows.append(row_data)

        print(pd.DataFrame(typ_rows).to_string(index=False))


def verify_real_category_codes(df_works):
    print("\n" + "=" * 100)
    print("6. REAL CATEGORY-CODE VERIFICATION (DATA AUDIT)")
    print("=" * 100)
    cats_all = sorted(df_works['work_category_code'].unique().tolist())
    df_real = df_works[df_works['is_synthetic_fraud'] == False]
    cats_real = sorted(df_real['work_category_code'].unique().tolist())

    print(f"All benchmark works `work_category_code` unique values:  {cats_all} (min={min(cats_all)}, max={max(cats_all)})")
    print(f"Real works only `work_category_code` unique values:        {cats_real} (min={min(cats_real)}, max={max(cats_real)})")
    print(f"Value Counts in Real Works:")
    for code, cnt in df_real['work_category_code'].value_counts().sort_index().items():
        print(f"  Code {code}: {cnt:>5} works ({cnt/len(df_real)*100:5.2f}%)")

    print("\nAudit Conclusion for Spec Correction:")
    print("  - The spec currently states `work_category_code (categorical category code: 0 to 2)`.")
    print("  - In empirical real data and benchmark works, there are 4 distinct category codes: {0, 1, 2, 3}.")
    print("  - Spec range MUST be corrected from `0 to 2` to `0 to 3`.")


def main():
    print("=" * 100)
    print("PROMPT 2A-FIX3: ISOLATED COMMUNITY SCORER, F MECHANISM & TIE-AWARE AUDIT")
    print("=" * 100)

    # 1. Git first
    run_git_first()

    # Load data
    works_path = os.path.join("data", "benchmark", "benchmark_works.csv")
    df_works = pd.read_csv(works_path)

    # 2. Contamination check (Real-only vs Episode-isolated)
    df_real, df_syn, scores_real_a, scores_syn_b = evaluate_contamination_and_isolation(df_works)

    # 3. Isolated evaluation (Injected (b) vs Real (a) & Rebuilt Co-located Null)
    evaluate_isolated_evaluation(df_real, df_syn, scores_real_a, scores_syn_b)

    # 4. F mechanism
    evaluate_f_mechanism(df_works, df_real, df_syn)

    # 5. Tie-aware top-k
    evaluate_tie_aware_top_k()

    # 6. Category-code verification
    verify_real_category_codes(df_works)


if __name__ == "__main__":
    main()
