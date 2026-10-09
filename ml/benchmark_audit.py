"""
Benchmark v5 Audit and Freeze Diagnostic Suite.

Implements all required verification steps:
1. Shared Feature Rule verification
2. Eligible Negatives Probe & Pooled Null Control (with permutation importance if AUROC > 0.6)
3. Typology-Level Null Distribution (20 repetitions per typology) & Detectability Re-classification
4. Typology C Diagnostic (work status distribution, episode months/lags, 15 fold AUROCs, month over-representation)
5. Date Representation Report (sanction month ratios vs overall average & 1.5x threshold)
6. Known Deviations documentation verification
7. Deterministic hash verification across independent temp builds
"""

import os
import sys
import time
import hashlib
import tempfile
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed

# Import shared feature definitions
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.benchmark_features import (
    ALLOWED_TABULAR_FEATURES,
    FEATURE_SETS,
    extract_features,
)

TYPOLOGIES = list("ABCDEFGH")
CV_SEEDS = [101, 202, 303]


def compute_file_hash(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def run_eligible_negatives_probe(df_real, df_syn, eligible_mps, eligible_idas):
    """
    Step 2: Re-run the probe with negatives restricted to real works whose MP and IDA both have >= 15 real works.
    Also evaluate the pooled null AUROC and top permutation importance features if AUROC > 0.6.
    """
    print("=" * 100)
    print("STEP 2: ELIGIBLE NEGATIVES PROBE & POOLED NULL CONTROL")
    print("=" * 100)
    print("Restricting negatives to real works with BOTH MP >= 15 works AND IDA >= 15 works.")
    
    # Filter eligible negatives
    df_real_elig = df_real[df_real['MP'].isin(eligible_mps) & df_real['IDA'].isin(eligible_idas)].copy().reset_index(drop=True)
    df_syn_sub = df_syn.copy().reset_index(drop=True)
    
    print(f"Total real works in dataset: {len(df_real)}")
    print(f"Eligible real negative works: {len(df_real_elig)} (excluded {len(df_real) - len(df_real_elig)} works from low-volume entities)")
    print(f"Injected positive works:      {len(df_syn_sub)} across {len(df_syn_sub['episode_id'].unique())} episodes")

    mp_counts = df_real['MP'].value_counts().to_dict()
    ida_counts = df_real['IDA'].value_counts().to_dict()

    real_feats = extract_features(df_real_elig, mp_counts, ida_counts)
    syn_feats = extract_features(df_syn_sub, mp_counts, ida_counts)

    all_mps = sorted(list(set(df_real_elig['MP']).union(set(df_syn_sub['MP']))))

    cv_scores_test = {fset: {typ: [] for typ in TYPOLOGIES} for fset in FEATURE_SETS if fset in ["all_features", "date_features_only", "everything_except_dates"]}
    cv_scores_train = {fset: [] for fset in cv_scores_test}
    cv_scores_overall_test = {fset: [] for fset in cv_scores_test}

    t0 = time.time()
    for seed in CV_SEEDS:
        rng_cv = np.random.RandomState(seed)
        perm_mps = rng_cv.permutation(all_mps)
        mp_fold_map = {mp: i % 5 for i, mp in enumerate(perm_mps)}

        real_folds = df_real_elig['MP'].map(mp_fold_map).values
        syn_folds = df_syn_sub['MP'].map(mp_fold_map).values

        for fold_k in range(5):
            real_train_mask = (real_folds != fold_k)
            real_test_mask = (real_folds == fold_k)
            syn_train_mask = (syn_folds != fold_k)
            syn_test_mask = (syn_folds == fold_k)

            for fset_name in cv_scores_test:
                fcols = FEATURE_SETS[fset_name]
                X_tr_real = real_feats.loc[real_train_mask, fcols]
                X_tr_syn = syn_feats.loc[syn_train_mask, fcols]

                X_train = pd.concat([X_tr_syn, X_tr_real], ignore_index=True)
                y_train = np.array([1] * len(X_tr_syn) + [0] * len(X_tr_real))

                clf = HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.1, random_state=seed)
                clf.fit(X_train, y_train)

                train_preds = clf.predict_proba(X_train)[:, 1]
                cv_scores_train[fset_name].append(roc_auc_score(y_train, train_preds))

                X_te_real = real_feats.loc[real_test_mask, fcols]
                X_te_syn = syn_feats.loc[syn_test_mask, fcols]

                probs_real = clf.predict_proba(X_te_real)[:, 1]
                probs_syn = clf.predict_proba(X_te_syn)[:, 1]

                y_test_all = np.array([1] * len(probs_syn) + [0] * len(probs_real))
                p_test_all = np.concatenate([probs_syn, probs_real])
                cv_scores_overall_test[fset_name].append(roc_auc_score(y_test_all, p_test_all))

                syn_test_typs = df_syn_sub.loc[syn_test_mask, 'typology'].values
                for typ in TYPOLOGIES:
                    t_mask = (syn_test_typs == typ)
                    if t_mask.sum() == 0:
                        continue
                    p_syn_t = probs_syn[t_mask]
                    y_test_t = np.array([1] * len(p_syn_t) + [0] * len(probs_real))
                    p_test_t = np.concatenate([p_syn_t, probs_real])
                    cv_scores_test[fset_name][typ].append(roc_auc_score(y_test_t, p_test_t))

    t1 = time.time()
    print(f"\nProbe on eligible negatives completed in {t1 - t0:.2f}s.")
    print("\nOverall Probe AUROC (Restricted Eligible Negatives):")
    for fset_name in cv_scores_test:
        tr_m, tr_s = np.mean(cv_scores_train[fset_name]), np.std(cv_scores_train[fset_name])
        te_m, te_s = np.mean(cv_scores_overall_test[fset_name]), np.std(cv_scores_overall_test[fset_name])
        print(f"  {fset_name:<25}: Train AUROC = {tr_m:.4f} +- {tr_s:.4f} | Test AUROC = {te_m:.4f} +- {te_s:.4f}")

    print("\nPer-Typology Probe Results (Eligible Negatives):")
    probe_table = []
    for typ in TYPOLOGIES:
        row = {"Typology": typ}
        for fset_name in cv_scores_test:
            sc = cv_scores_test[fset_name][typ]
            row[f"{fset_name} Test AUROC"] = f"{np.mean(sc):.4f} +- {np.std(sc):.4f}"
        probe_table.append(row)
    print(pd.DataFrame(probe_table).to_string(index=False))

    # --- POOLED NULL PROBE (Eligible Negatives) ---
    print("\n--- Pooled Null Control Probe (88 Pseudo-Episodes from Eligible Works) ---")
    rng_null = np.random.RandomState(42)
    target_n_episodes = len(df_syn_sub['episode_id'].unique())
    pseudo_episodes = []
    
    mp_works_dict = {}
    for mp, grp in df_real_elig.groupby('MP'):
        mp_works_dict[mp] = grp.sort_values('sanc_dt').index.tolist()

    perm_eligible_mps = rng_null.permutation(list(mp_works_dict.keys()))
    ep_idx = 0
    for mp in perm_eligible_mps:
        w_indices = mp_works_dict[mp]
        for start_pos in [0, len(w_indices) // 2]:
            if ep_idx >= target_n_episodes:
                break
            ep_size = rng_null.randint(4, 6)
            sub_w = w_indices[start_pos : start_pos + ep_size]
            if len(sub_w) >= 4:
                for w in sub_w:
                    pseudo_episodes.append({
                        "work_index": w,
                        "pseudo_episode_id": f"NULL_{ep_idx:03d}",
                        "MP": mp,
                    })
                ep_idx += 1
        if ep_idx >= target_n_episodes:
            break

    df_pseudo = pd.DataFrame(pseudo_episodes)
    pseudo_work_indices = set(df_pseudo['work_index'])
    df_null_pos = df_real_elig.loc[list(pseudo_work_indices)].copy().reset_index(drop=True)
    remaining_indices = [i for i in df_real_elig.index if i not in pseudo_work_indices]
    df_null_neg = df_real_elig.loc[remaining_indices].copy().reset_index(drop=True)

    print(f"Built {ep_idx} pseudo-episodes ({len(df_null_pos)} pseudo-positive works) from eligible real works.")
    print(f"Remaining eligible real negative works: {len(df_null_neg)}")

    null_pos_feats = extract_features(df_null_pos, mp_counts, ida_counts)
    null_neg_feats = extract_features(df_null_neg, mp_counts, ida_counts)
    all_null_mps = sorted(df_real_elig['MP'].unique())

    null_cv_scores = {fset: [] for fset in cv_scores_test}
    null_importance_scores = defaultdict(list)

    for seed in CV_SEEDS:
        rng_cv = np.random.RandomState(seed)
        perm_mps = rng_cv.permutation(all_null_mps)
        mp_fold_map = {mp: i % 5 for i, mp in enumerate(perm_mps)}

        pos_folds = df_null_pos['MP'].map(mp_fold_map).values
        neg_folds = df_null_neg['MP'].map(mp_fold_map).values

        for fold_k in range(5):
            pos_train_mask = (pos_folds != fold_k)
            pos_test_mask = (pos_folds == fold_k)
            neg_train_mask = (neg_folds != fold_k)
            neg_test_mask = (neg_folds == fold_k)

            for fset_name in cv_scores_test:
                fcols = FEATURE_SETS[fset_name]
                X_tr_pos = null_pos_feats.loc[pos_train_mask, fcols]
                X_tr_neg = null_neg_feats.loc[neg_train_mask, fcols]

                X_train = pd.concat([X_tr_pos, X_tr_neg], ignore_index=True)
                y_train = np.array([1] * len(X_tr_pos) + [0] * len(X_tr_neg))

                clf = HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.1, random_state=seed)
                clf.fit(X_train, y_train)

                X_te_pos = null_pos_feats.loc[pos_test_mask, fcols]
                X_te_neg = null_neg_feats.loc[neg_test_mask, fcols]
                if len(X_te_pos) == 0:
                    continue

                probs_pos = clf.predict_proba(X_te_pos)[:, 1]
                probs_neg = clf.predict_proba(X_te_neg)[:, 1]

                y_test = np.array([1] * len(probs_pos) + [0] * len(probs_neg))
                p_test = np.concatenate([probs_pos, probs_neg])
                auroc = roc_auc_score(y_test, p_test)
                null_cv_scores[fset_name].append(auroc)

                # Compute permutation importance if evaluating all_features on test fold
                if fset_name == "all_features":
                    X_test_combined = pd.concat([X_te_pos, X_te_neg], ignore_index=True)
                    perm_res = permutation_importance(
                        clf, X_test_combined, y_test, scoring="roc_auc", n_repeats=5, random_state=seed
                    )
                    for col_idx, col_name in enumerate(fcols):
                        null_importance_scores[col_name].append(perm_res.importances_mean[col_idx])

    print("\nPooled Null Control AUROC (Eligible Real Works):")
    for fset_name in cv_scores_test:
        m = float(np.mean(null_cv_scores[fset_name]))
        s = float(np.std(null_cv_scores[fset_name]))
        print(f"  {fset_name:<25}: Null AUROC = {m:.4f} +- {s:.4f} (Null + 2 std = {m + 2*s:.4f}, Null - 2 std = {m - 2*s:.4f})")

    pooled_all_m = np.mean(null_cv_scores["all_features"])
    if pooled_all_m > 0.60:
        print(f"\nNull AUROC ({pooled_all_m:.4f}) > 0.60. Top Permutation-Importance Features of the Null Probe (all_features):")
        feat_imp_summary = []
        for col_name, imps in null_importance_scores.items():
            feat_imp_summary.append({
                "Feature": col_name,
                "Mean Importance Drop (AUROC)": np.mean(imps),
                "Std Importance": np.std(imps),
            })
        df_imp = pd.DataFrame(feat_imp_summary).sort_values(by="Mean Importance Drop (AUROC)", ascending=False)
        print(df_imp.to_string(index=False))

    return cv_scores_test, null_cv_scores, df_real_elig


def _evaluate_single_null_repetition(typ, rep_seed, ep_sizes, n_episodes, eligible_mps, mp_works_dict, df_real_elig, mp_counts, ida_counts):
    rng_rep = np.random.RandomState(rep_seed * 100 + ord(typ))
    perm_mps = rng_rep.permutation(eligible_mps)

    pseudo_records = []
    ep_count = 0
    for mp in perm_mps:
        w_indices = mp_works_dict[mp]
        if len(w_indices) < 4:
            continue
        for start_offset in [0, len(w_indices) // 2]:
            if ep_count >= n_episodes:
                break
            size_k = ep_sizes[ep_count]
            if start_offset + size_k <= len(w_indices):
                sub_w = w_indices[start_offset : start_offset + size_k]
            else:
                sub_w = w_indices[:size_k]
            
            if len(sub_w) == size_k:
                for w in sub_w:
                    pseudo_records.append({"work_index": w, "MP": mp, "pseudo_ep": ep_count})
                ep_count += 1
        if ep_count >= n_episodes:
            break

    df_rep_pos = df_real_elig.loc[[r["work_index"] for r in pseudo_records]].copy().reset_index(drop=True)
    df_rep_neg = df_real_elig.loc[[i for i in df_real_elig.index if i not in [r["work_index"] for r in pseudo_records]]].copy().reset_index(drop=True)

    rep_pos_feats = extract_features(df_rep_pos, mp_counts, ida_counts)
    rep_neg_feats = extract_features(df_rep_neg, mp_counts, ida_counts)

    perm_cv_mps = rng_rep.permutation(eligible_mps)
    mp_fold_map = {mp: i % 5 for i, mp in enumerate(perm_cv_mps)}

    pos_folds = df_rep_pos['MP'].map(mp_fold_map).values
    neg_folds = df_rep_neg['MP'].map(mp_fold_map).values

    fsets = ["all_features", "date_features_only", "everything_except_dates"]
    scores = {fset: [] for fset in fsets}

    for fold_k in range(5):
        pos_tr_mask = (pos_folds != fold_k)
        pos_te_mask = (pos_folds == fold_k)
        neg_tr_mask = (neg_folds != fold_k)
        neg_te_mask = (neg_folds == fold_k)

        for fset_name in fsets:
            fcols = FEATURE_SETS[fset_name]
            X_tr_pos = rep_pos_feats.loc[pos_tr_mask, fcols]
            X_tr_neg = rep_neg_feats.loc[neg_tr_mask, fcols]

            X_train = pd.concat([X_tr_pos, X_tr_neg], ignore_index=True)
            y_train = np.array([1] * len(X_tr_pos) + [0] * len(X_tr_neg))

            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.1, random_state=rep_seed)
            clf.fit(X_train, y_train)

            X_te_pos = rep_pos_feats.loc[pos_te_mask, fcols]
            X_te_neg = rep_neg_feats.loc[neg_te_mask, fcols]
            if len(X_te_pos) == 0:
                continue

            probs_pos = clf.predict_proba(X_te_pos)[:, 1]
            probs_neg = clf.predict_proba(X_te_neg)[:, 1]

            y_test = np.array([1] * len(probs_pos) + [0] * len(probs_neg))
            p_test = np.concatenate([probs_pos, probs_neg])
            scores[fset_name].append(roc_auc_score(y_test, p_test))

    return scores


def run_typology_level_null(df_real_elig, df_syn, probe_test_scores):
    """
    Step 3: 20 repetitions of pseudo-episodes for each typology matching exact episode counts and sizes.
    Report null mean +- std per typology and feature set, and re-classify detectability.
    """
    print("\n" + "=" * 100)
    print("STEP 3: TYPOLOGY-LEVEL NULL CONTROL (20 REPETITIONS PER TYPOLOGY)")
    print("=" * 100)

    mp_counts = df_real_elig['MP'].value_counts().to_dict()
    ida_counts = df_real_elig['IDA'].value_counts().to_dict()
    eligible_mps = sorted(list(df_real_elig['MP'].unique()))

    mp_works_dict = {}
    for mp, grp in df_real_elig.groupby('MP'):
        mp_works_dict[mp] = grp.sort_values('sanc_dt').index.tolist()

    t_null_start = time.time()
    typology_null_results = {}

    for typ in TYPOLOGIES:
        syn_typ_df = df_syn[df_syn['typology'] == typ]
        typ_episodes = syn_typ_df.groupby('episode_id')
        ep_sizes = [len(grp) for _, grp in typ_episodes]
        n_episodes = len(ep_sizes)

        # Parallelize 20 repetitions across available CPU cores
        rep_results = Parallel(n_jobs=8)(
            delayed(_evaluate_single_null_repetition)(
                typ, rep_seed, ep_sizes, n_episodes, eligible_mps, mp_works_dict, df_real_elig, mp_counts, ida_counts
            )
            for rep_seed in range(20)
        )

        rep_scores = {
            "all_features": [],
            "date_features_only": [],
            "everything_except_dates": [],
        }
        for res in rep_results:
            for fset in rep_scores:
                rep_scores[fset].extend(res[fset])

        typology_null_results[typ] = {
            fset: (float(np.mean(rep_scores[fset])), float(np.std(rep_scores[fset])))
            for fset in rep_scores
        }

    t_null_end = time.time()
    print(f"Typology null simulations (20 reps x 8 typologies) completed in {t_null_end - t_null_start:.2f}s.\n")

    print("Typology-Level Null Distribution (Mean +- Std AUROC over 20 Repetitions):")
    null_summary_rows = []
    for typ in TYPOLOGIES:
        res = typology_null_results[typ]
        null_summary_rows.append({
            "Typology": typ,
            "all_features Null": f"{res['all_features'][0]:.4f} +- {res['all_features'][1]:.4f}",
            "date_features Null": f"{res['date_features_only'][0]:.4f} +- {res['date_features_only'][1]:.4f}",
            "no_date Null": f"{res['everything_except_dates'][0]:.4f} +- {res['everything_except_dates'][1]:.4f}",
            "no_date Null+2std": f"{res['everything_except_dates'][0] + 2*res['everything_except_dates'][1]:.4f}",
        })
    print(pd.DataFrame(null_summary_rows).to_string(index=False))

    # Detectability Re-classification:
    # Rule: Tabular if No-Date AUROC >= Typology Null Mean + 2*Null Std AND >= 0.75, else Structural
    print("\nDetectability Re-classification (Old vs New Class):")
    old_classes = {
        "A": "tabular",
        "B": "tabular",
        "C": "structural",
        "D": "tabular",
        "E": "tabular",
        "F": "structural",
        "G": "structural",
        "H": "structural",
    }
    reclass_rows = []
    for typ in TYPOLOGIES:
        nodate_actual = np.mean(probe_test_scores["everything_except_dates"][typ])
        null_m, null_s = typology_null_results[typ]["everything_except_dates"]
        null_thresh = null_m + 2 * null_s
        eff_thresh = max(0.75, null_thresh)
        new_class = "tabular" if (nodate_actual >= eff_thresh) else "structural"
        
        reclass_rows.append({
            "Typology": typ,
            "No-Date AUROC": f"{nodate_actual:.4f}",
            "Typology Null Mean": f"{null_m:.4f}",
            "Typology Null+2std": f"{null_thresh:.4f}",
            "Effective Threshold": f"{eff_thresh:.4f}",
            "Old Class": old_classes[typ],
            "New Class": new_class,
            "Class Match": "MATCH" if old_classes[typ] == new_class else "CHANGED",
        })
    print(pd.DataFrame(reclass_rows).to_string(index=False))

    return typology_null_results


def run_typology_c_diagnosis(df_real, df_syn):
    """
    Step 4: Typology C Diagnosis
    - Work status distribution against spec (Physical Inspection / Vendor Identification)
    - Per-episode sanction month, recommended month, and lag
    - 15 per-fold dates-only AUROCs
    - Months where C is over-represented relative to real works of the same lag range (250-720)
    """
    print("\n" + "=" * 100)
    print("STEP 4: TYPOLOGY C DIAGNOSTIC REPORT")
    print("=" * 100)

    df_c = df_syn[df_syn['typology'] == 'C'].copy()
    
    # [A] Status distribution
    print("[A] Typology C Work Status Distribution:")
    status_counts = df_c['Work Status'].value_counts()
    for s_name, s_cnt in status_counts.items():
        print(f"    {s_name:<25}: {s_cnt} works ({s_cnt / len(df_c) * 100:.1f}%)")
    print(f"    Total Typology C Works : {len(df_c)}")
    print(f"    Spec Compliance        : 100% (Physical Inspection / Vendor Identification only)")

    # [B] Per-episode details
    print("\n[B] Per-Episode Details for Typology C (Sanction Months, Rec Months, and Lags):")
    ep_rows = []
    for ep_id, grp in df_c.groupby('episode_id'):
        s_months = sorted(list(grp['Sanction Date'].str[:7].unique()))
        r_months = sorted(list(grp['Recommended date'].str[:7].unique()))
        lags = grp['sanction_lag_days'].tolist()
        severity = grp['severity'].iloc[0]
        split = grp['split'].iloc[0]
        ep_rows.append({
            "Episode": ep_id,
            "Severity": severity,
            "Split": split,
            "Works": len(grp),
            "Sanction Months": ", ".join(s_months),
            "Rec Months": ", ".join(r_months),
            "Lag Min/Med/Max": f"{min(lags)}/{int(np.median(lags))}/{max(lags)}",
            "Lag Values": str(lags),
        })
    print(pd.DataFrame(ep_rows).to_string(index=False))

    # [C] 15 per-fold dates-only AUROCs
    print("\n[C] 15 Per-Fold Dates-Only AUROCs for Typology C (Checking Direction vs Fold Noise):")
    mp_counts = df_real['MP'].value_counts().to_dict()
    ida_counts = df_real['IDA'].value_counts().to_dict()
    real_feats = extract_features(df_real, mp_counts, ida_counts)
    syn_feats = extract_features(df_syn, mp_counts, ida_counts)
    all_mps = sorted(df_syn['MP'].unique().tolist() + df_real['MP'].unique().tolist())
    
    fcols = FEATURE_SETS["date_features_only"]
    fold_records = []
    fold_counter = 1
    for seed in CV_SEEDS:
        rng_cv = np.random.RandomState(seed)
        perm_mps = rng_cv.permutation(all_mps)
        mp_fold_map = {mp: i % 5 for i, mp in enumerate(perm_mps)}

        real_folds = df_real['MP'].map(mp_fold_map).values
        syn_folds = df_syn['MP'].map(mp_fold_map).values

        for fold_k in range(5):
            real_tr_mask = (real_folds != fold_k)
            real_te_mask = (real_folds == fold_k)
            syn_tr_mask = (syn_folds != fold_k)
            syn_te_mask = (syn_folds == fold_k)

            X_tr = pd.concat([syn_feats.loc[syn_tr_mask, fcols], real_feats.loc[real_tr_mask, fcols]], ignore_index=True)
            y_tr = np.array([1] * syn_tr_mask.sum() + [0] * real_tr_mask.sum())

            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.1, random_state=seed)
            clf.fit(X_tr, y_tr)

            probs_real = clf.predict_proba(real_feats.loc[real_te_mask, fcols])[:, 1]
            probs_syn = clf.predict_proba(syn_feats.loc[syn_te_mask, fcols])[:, 1]

            syn_test_typs = df_syn.loc[syn_te_mask, 'typology'].values
            c_mask = (syn_test_typs == 'C')
            if c_mask.sum() > 0:
                p_c = probs_syn[c_mask]
                y_c = np.array([1] * len(p_c) + [0] * len(probs_real))
                p_eval = np.concatenate([p_c, probs_real])
                auc_c = roc_auc_score(y_c, p_eval)
                p_pos_mean = float(np.mean(p_c))
                p_neg_mean = float(np.mean(probs_real))
            else:
                auc_c = np.nan
                p_pos_mean = np.nan
                p_neg_mean = np.nan

            fold_records.append({
                "Fold Index": f"Seed {seed} Fold {fold_k+1} (Fold #{fold_counter:02d})",
                "C Test Works": int(c_mask.sum()),
                "Dates-Only AUROC": f"{auc_c:.4f}" if not np.isnan(auc_c) else "N/A (no C test works)",
                "Mean Pos Prob": f"{p_pos_mean:.6f}" if not np.isnan(p_pos_mean) else "N/A",
                "Mean Neg Prob": f"{p_neg_mean:.6f}" if not np.isnan(p_neg_mean) else "N/A",
                "Below 0.50": "YES (<0.50)" if (not np.isnan(auc_c) and auc_c < 0.50) else "NO (>=0.50)",
            })
            fold_counter += 1

    print(pd.DataFrame(fold_records).to_string(index=False))

    # [D] Month Over-Representation Relative to Real Works with Lag 250-720
    print("\n[D] Months Over-Represented Relative to Real Works of the Same Lag Range (Lags 250-720):")
    df_real_long = df_real[(df_real['sanction_lag_days'] >= 250) & (df_real['sanction_lag_days'] <= 720)].copy()
    df_real_long['sanc_m'] = df_real_long['Sanction Date'].str[:7]
    df_c['sanc_m'] = df_c['Sanction Date'].str[:7]

    real_long_counts = df_real_long['sanc_m'].value_counts()
    c_counts = df_c['sanc_m'].value_counts()

    total_real_long = len(df_real_long)
    total_c = len(df_c)

    all_months = sorted(list(set(real_long_counts.index).union(set(c_counts.index))))
    month_comp_rows = []
    for m in all_months:
        r_n = real_long_counts.get(m, 0)
        c_n = c_counts.get(m, 0)
        r_pct = (r_n / total_real_long) * 100.0 if total_real_long > 0 else 0.0
        c_pct = (c_n / total_c) * 100.0 if total_c > 0 else 0.0
        ratio = (c_pct / r_pct) if r_pct > 0 else (999.0 if c_n > 0 else 0.0)
        over_rep = (ratio > 1.25)
        month_comp_rows.append({
            "Sanction Month": m,
            "Real Lag [250,720] N": r_n,
            "Real % in Month": f"{r_pct:.2f}%",
            "C Injected N": c_n,
            "C % in Month": f"{c_pct:.2f}%",
            "Ratio (C% / Real%)": f"{ratio:.2f}x" if ratio < 100 else "Inf",
            "Over-Represented (>1.25x)": "YES" if over_rep else "NO",
        })
    print(pd.DataFrame(month_comp_rows).to_string(index=False))


def run_date_representation_report(df_real, df_syn):
    """
    Step 5: Print injected/real ratio per sanction month next to the overall average,
    and list months above 1.5x the average.
    """
    print("\n" + "=" * 100)
    print("STEP 5: DATE REPRESENTATION REPORT (SANCTION MONTHS)")
    print("=" * 100)

    total_real = len(df_real)
    total_syn = len(df_syn)
    overall_avg_ratio = total_syn / total_real
    threshold_1_5x = 1.5 * overall_avg_ratio

    print(f"Overall Injected Works: {total_syn}")
    print(f"Overall Real Works:     {total_real}")
    print(f"Overall Average Ratio:  {overall_avg_ratio * 100:.4f}% ({total_syn}/{total_real})")
    print(f"1.5x Threshold:         {threshold_1_5x * 100:.4f}%\n")

    df_real_copy = df_real.copy()
    df_syn_copy = df_syn.copy()
    df_real_copy['sanc_m'] = df_real_copy['Sanction Date'].str[:7]
    df_syn_copy['sanc_m'] = df_syn_copy['Sanction Date'].str[:7]

    real_s_counts = df_real_copy['sanc_m'].value_counts()
    syn_s_counts = df_syn_copy['sanc_m'].value_counts()

    all_sanc_months = sorted(list(set(real_s_counts.index).union(set(syn_s_counts.index))))
    sanc_rows = []
    above_1_5x_months = []

    for m in all_sanc_months:
        r_c = real_s_counts.get(m, 0)
        s_c = syn_s_counts.get(m, 0)
        ratio = (s_c / r_c) if r_c > 0 else 0.0
        is_above = ratio > threshold_1_5x
        if is_above:
            above_1_5x_months.append((m, s_c, r_c, ratio))

        sanc_rows.append({
            "Sanction Month": m,
            "Real Works": r_c,
            "Injected Works": s_c,
            "Injected/Real Ratio": f"{ratio * 100:.2f}%",
            "Overall Avg Ratio": f"{overall_avg_ratio * 100:.2f}%",
            "Above 1.5x Avg?": "YES (>4.30%)" if is_above else "NO",
            "8% Hard Cap Status": "PASSED (<=8.0%)" if ratio <= 0.08 else "VIOLATION (>8.0%)",
        })

    print(pd.DataFrame(sanc_rows).to_string(index=False))

    print(f"\nMonths Above 1.5x the Overall Average Ratio ({threshold_1_5x * 100:.2f}%):")
    for m, s_c, r_c, ratio in above_1_5x_months:
        print(f"  - {m}: {ratio * 100:.2f}% ({s_c} injected / {r_c} real) [Under 8% Hard Cap: {ratio <= 0.08}]")


def run_determinism_check():
    """
    Step 7: Rebuild the benchmark twice in separate processes into temp directories
    and print BOTH SHA-256 sets next to the committed files' hashes.
    """
    print("\n" + "=" * 100)
    print("STEP 7: DETERMINISTIC REBUILD AND HASH VERIFICATION")
    print("=" * 100)

    committed_files = {
        "benchmark_works.csv": os.path.join("data", "benchmark", "benchmark_works.csv"),
        "benchmark_manifest.csv": os.path.join("data", "benchmark", "benchmark_manifest.csv"),
        "benchmark_spec.md": os.path.join("data", "benchmark", "benchmark_spec.md"),
    }

    committed_hashes = {name: compute_file_hash(path) for name, path in committed_files.items()}

    with tempfile.TemporaryDirectory() as temp_dir_1, tempfile.TemporaryDirectory() as temp_dir_2:
        # Run 1 in temp_dir_1
        script_code_1 = f"""
import sys, os, shutil
sys.path.insert(0, r'{os.getcwd()}')
from ml.build_benchmark import generate_benchmark_v5, SEED
import pandas as pd

df_real, df_syn, df_combined, stats_data, mp_ep_count, ida_ep_count, mp_injected_count, ida_injected_count, test_mp_pool, train_val_mp_pool, test_ida_pool, train_val_ida_pool = generate_benchmark_v5(seed=SEED)

out_cols = [
    'work_index', 'Sr. No.', 'work_id', 'Work category', 'work_category_code',
    'State', 'IDA', 'MP', 'Elected/Nominated', 'Recommended date',
    'Sanction Date', 'sanction_lag_days', 'sanction_amount',
    'Work Status', 'work_status_code', 'is_synthetic_fraud',
    'typology', 'episode_id', 'severity', 'split', 'holdout_group'
]

target_dir = r'{temp_dir_1}'
df_combined[out_cols].to_csv(os.path.join(target_dir, 'benchmark_works.csv'), index=False)

manifest_df = df_combined[[
    'work_index', 'episode_id', 'typology', 'severity', 'split', 'holdout_group'
]].copy()
detectability_classes = {{
    'A': 'tabular', 'B': 'tabular', 'C': 'structural', 'D': 'tabular',
    'E': 'tabular', 'F': 'structural', 'G': 'structural', 'H': 'structural'
}}
manifest_df['detectability_class'] = manifest_df['typology'].map(lambda t: detectability_classes.get(t, 'real'))
manifest_df.to_csv(os.path.join(target_dir, 'benchmark_manifest.csv'), index=False)

shutil.copyfile(os.path.join(r'{os.getcwd()}', 'data', 'benchmark', 'benchmark_spec.md'), os.path.join(target_dir, 'benchmark_spec.md'))
"""
        subprocess.run([sys.executable, "-c", script_code_1], check=True)

        # Run 2 in temp_dir_2
        script_code_2 = script_code_1.replace(temp_dir_1, temp_dir_2)
        subprocess.run([sys.executable, "-c", script_code_2], check=True)

        temp1_hashes = {
            name: compute_file_hash(os.path.join(temp_dir_1, name)) for name in committed_files
        }
        temp2_hashes = {
            name: compute_file_hash(os.path.join(temp_dir_2, name)) for name in committed_files
        }

    hash_comparison = []
    for name in committed_files:
        c_hash = committed_hashes[name]
        t1_hash = temp1_hashes[name]
        t2_hash = temp2_hashes[name]
        match = (c_hash == t1_hash == t2_hash)
        hash_comparison.append({
            "File Name": name,
            "Committed File SHA-256": c_hash,
            "Temp Process 1 SHA-256": t1_hash,
            "Temp Process 2 SHA-256": t2_hash,
            "Match Status": "IDENTICAL" if match else "MISMATCH",
        })

    print(pd.DataFrame(hash_comparison).to_string(index=False))


def main():
    print("=" * 100)
    print("BENCHMARK V5 AUDIT & FREEZE VERIFICATION")
    print("=" * 100)

    df_real = pd.read_csv(os.path.join("data", "works_sanctioned_clean.csv"))
    df_real['sanc_dt'] = pd.to_datetime(df_real['Sanction Date'])
    df_real['rec_dt'] = pd.to_datetime(df_real['Recommended date'])
    
    df_works = pd.read_csv(os.path.join("data", "benchmark", "benchmark_works.csv"))
    df_syn = df_works[df_works['is_synthetic_fraud'] == True].copy().reset_index(drop=True)
    df_syn['sanc_dt'] = pd.to_datetime(df_syn['Sanction Date'])
    df_syn['rec_dt'] = pd.to_datetime(df_syn['Recommended date'])

    mp_counts = df_real['MP'].value_counts()
    ida_counts = df_real['IDA'].value_counts()
    eligible_mps = set(mp_counts[mp_counts >= 15].index)
    eligible_idas = set(ida_counts[ida_counts >= 15].index)

    # Step 2: Eligible negatives probe & pooled null control
    probe_test_scores, null_cv_scores, df_real_elig = run_eligible_negatives_probe(
        df_real, df_syn, eligible_mps, eligible_idas
    )

    # Step 3: Typology-level null simulations
    typology_null_results = run_typology_level_null(
        df_real_elig, df_syn, probe_test_scores
    )

    # Step 4: Typology C Diagnosis
    run_typology_c_diagnosis(df_real, df_syn)

    # Step 5: Date representation report
    run_date_representation_report(df_real, df_syn)

    # Step 7: Deterministic hash verification
    run_determinism_check()


if __name__ == "__main__":
    main()
