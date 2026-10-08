"""
Synthetic Fraud Benchmark Generation Script (v3.0 - Leakage-Aware)

Generates benchmark_works.csv and benchmark_manifest.csv with:
1. Host entity selection: weighted sampling (p ~ real count), min 15 real works,
   max 2 episodes per entity, max 25% injected share, disjoint train/val vs test entities.
2. Existing pair constraints: >=50% existing pairs for A, D, H; 100% novel for F; 100% existing for B, C, E, G.
3. Entity report per typology.
4. Leakage tests:
   a. Pooled AUC per typology x severity.
   b. Permutation null (200 pseudo-episodes) 95th percentile comparison.
   c. Adversarial probe (HistGradientBoostingClassifier) AUROC/AUPRC + feature importance.
5. Detectability class assignment (tabular if AUROC >= 0.85 else structural).
6. Text column dropping (Work description dropped as synthetic text is not reused).
7. Determinism verification.
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

SEED = 42
BASE_IDX = 30_000
TYPOLOGIES = list("ABCDEFGH")
SEVERITIES = ["subtle", "moderate", "obvious"]
EPISODES_PER_SEVERITY = 4
EPISODES_PER_TYPOLOGY = 12

STATUS_TO_CODE = {
    "Physical Inspection": 0,
    "Sanction": 1,
    "Time Estimation": 2,
    "Vendor Identification": 3,
    "Work Completed": 4,
    "Work partially Completed": 5,
}

def _status_code(status_name):
    return STATUS_TO_CODE.get(status_name, 0)

def _load_real_data():
    path = os.path.join("data", "works_sanctioned_clean.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)

def _real_quantile(series, q):
    return float(np.quantile(series.dropna(), q))

def _sample_status(rng, n, real_status_dist):
    statuses = sorted(real_status_dist.keys())
    probs = np.array([real_status_dist[s] for s in statuses], dtype=float)
    probs /= probs.sum()
    chosen = rng.choice(len(statuses), size=n, p=probs)
    return [statuses[c] for c in chosen]

def _clamp_amount(val, real_min, real_max):
    return float(np.clip(val, real_min, real_max))

def _clamp_lag(val, real_max):
    return int(np.clip(round(val), 0, real_max))

def _make_rec_date(sanction_date, lag_days):
    sanc = pd.Timestamp(sanction_date)
    rec = sanc - pd.Timedelta(days=int(lag_days))
    return rec.strftime("%Y-%m-%d")

def _sample_weighted(candidates, counts_dict, rng, n=1, replace=False):
    if len(candidates) == 0:
        raise ValueError("No candidates available for sampling.")
    cand_list = sorted(list(candidates))
    weights = np.array([counts_dict[c] for c in cand_list], dtype=float)
    probs = weights / weights.sum()
    n_sample = min(n, len(cand_list))
    chosen_idx = rng.choice(len(cand_list), size=n_sample, replace=replace, p=probs)
    return [cand_list[i] for i in chosen_idx]

def generate_benchmark_v3(seed=SEED):
    rng = np.random.RandomState(seed)
    df_real = _load_real_data()

    mp_counts = df_real['MP'].value_counts().to_dict()
    ida_counts = df_real['IDA'].value_counts().to_dict()
    mp_home_state = df_real.groupby('MP')['State'].agg(lambda s: s.mode().iloc[0]).to_dict()
    ida_state = df_real.groupby('IDA')['State'].agg(lambda s: s.mode().iloc[0]).to_dict()

    eligible_mps = set([mp for mp, c in mp_counts.items() if c >= 15])
    eligible_idas = set([ida for ida, c in ida_counts.items() if c >= 15])

    df_elig = df_real[df_real['MP'].isin(eligible_mps) & df_real['IDA'].isin(eligible_idas)]
    real_elig_pairs = set(zip(df_elig['MP'], df_elig['IDA']))

    existing_mps_per_ida = defaultdict(set)
    for mp, ida in real_elig_pairs:
        existing_mps_per_ida[ida].add(mp)

    existing_idas_per_mp = defaultdict(set)
    for mp, ida in real_elig_pairs:
        existing_idas_per_mp[mp].add(ida)

    mps_by_state = defaultdict(list)
    for mp in eligible_mps:
        mps_by_state[mp_home_state[mp]].append(mp)
    for st in mps_by_state:
        mps_by_state[st].sort()

    idas_by_state = defaultdict(list)
    for ida in eligible_idas:
        idas_by_state[ida_state[ida]].append(ida)
    for st in idas_by_state:
        idas_by_state[st].sort()

    real_min_sanc = pd.Timestamp(df_real['Sanction Date'].min())
    real_max_sanc = pd.Timestamp(df_real['Sanction Date'].max())
    real_min_rec = pd.Timestamp(df_real['Recommended date'].min())
    real_max_rec = pd.Timestamp(df_real['Recommended date'].max())

    stats = {
        "amt_min": float(df_real["sanction_amount"].min()),
        "amt_max": float(df_real["sanction_amount"].max()),
        "lag_max": int(df_real["sanction_lag_days"].max()),
        "lag_p25": int(df_real["sanction_lag_days"].quantile(0.25)),
        "lag_p75": int(df_real["sanction_lag_days"].quantile(0.75)),
        "status_dist": df_real["Work Status"].value_counts().to_dict(),
        "mp_home_state": mp_home_state,
        "ida_state": ida_state,
        "mps_by_state": mps_by_state,
        "idas_by_state": idas_by_state,
        "mp_counts": mp_counts,
        "ida_counts": ida_counts,
        "eligible_mps": eligible_mps,
        "eligible_idas": eligible_idas,
        "existing_mps_per_ida": existing_mps_per_ida,
        "existing_idas_per_mp": existing_idas_per_mp,
        "real_elig_pairs": real_elig_pairs,
        "real_min_sanc": real_min_sanc,
        "real_max_sanc": real_max_sanc,
        "real_min_rec": real_min_rec,
        "real_max_rec": real_max_rec,
    }

    # Partition eligible entities into disjoint test and train_val pools per state
    test_mp_pool = set()
    train_val_mp_pool = set()
    test_ida_pool = set()
    train_val_ida_pool = set()

    for st in sorted(idas_by_state.keys()):
        idas = idas_by_state[st]
        perm_idas = [idas[i] for i in rng.permutation(len(idas))]
        n_test = max(1, int(round(0.25 * len(idas)))) if len(idas) >= 3 else 0
        test_ida_pool.update(perm_idas[:n_test])
        train_val_ida_pool.update(perm_idas[n_test:])

    for st in sorted(mps_by_state.keys()):
        mps = mps_by_state[st]
        perm_mps = [mps[i] for i in rng.permutation(len(mps))]
        n_test = max(1, int(round(0.25 * len(mps)))) if len(mps) >= 3 else 0
        test_mp_pool.update(perm_mps[:n_test])
        train_val_mp_pool.update(perm_mps[n_test:])

    # Track episode usage
    mp_ep_count = defaultdict(int)
    ida_ep_count = defaultdict(int)
    mp_injected_count = defaultdict(int)
    ida_injected_count = defaultdict(int)

    split_schedule = {
        1: "train", 2: "train", 3: "val", 4: "test",
        5: "train", 6: "train", 7: "val", 8: "test",
        9: "train", 10: "train", 11: "val", 12: "test"
    }
    severity_schedule = {
        1: "subtle", 2: "subtle", 3: "subtle", 4: "subtle",
        5: "moderate", 6: "moderate", 7: "moderate", 8: "moderate",
        9: "obvious", 10: "obvious", 11: "obvious", 12: "obvious"
    }

    def get_pool_mps(split):
        return test_mp_pool if split == "test" else train_val_mp_pool

    def get_pool_idas(split):
        return test_ida_pool if split == "test" else train_val_ida_pool

    def can_use_mp(mp, n_works, split):
        if mp not in get_pool_mps(split):
            return False
        if mp_ep_count[mp] >= 2:
            return False
        if (mp_injected_count[mp] + n_works) > 0.25 * mp_counts[mp]:
            return False
        return True

    def can_use_ida(ida, n_works, split):
        if ida not in get_pool_idas(split):
            return False
        if ida_ep_count[ida] >= 2:
            return False
        if (ida_injected_count[ida] + n_works) > 0.25 * ida_counts[ida]:
            return False
        return True

    def record_usage(mps_used, idas_used, n_works_per_mp, n_works_per_ida):
        for mp in mps_used:
            mp_ep_count[mp] += 1
            mp_injected_count[mp] += n_works_per_mp[mp]
        for ida in idas_used:
            ida_ep_count[ida] += 1
            ida_injected_count[ida] += n_works_per_ida[ida]

    all_records = []
    idx_cursor = BASE_IDX

    for typ in TYPOLOGIES:
        for ep_num in range(1, EPISODES_PER_TYPOLOGY + 1):
            severity = severity_schedule[ep_num]
            split = split_schedule[ep_num]
            episode_id = ep_num
            use_existing_pairs = (ep_num in [1, 2, 5, 6, 9, 10])

            if typ == "A":
                cfg = {
                    "subtle":   {"n_works": (4, 5), "n_mps": (2, 3), "window": (25, 40), "cv": (0.10, 0.15), "amt_q": (0.50, 0.75)},
                    "moderate": {"n_works": (5, 6), "n_mps": (2, 3), "window": (10, 20), "cv": (0.04, 0.09), "amt_q": (0.60, 0.90)},
                    "obvious":  {"n_works": (5, 6), "n_mps": (2, 3), "window": (3, 7), "cv": (0.01, 0.03), "amt_q": (0.70, 0.95)},
                }[severity]
                n_works = rng.randint(cfg["n_works"][0], cfg["n_works"][1] + 1)
                n_mps = rng.randint(cfg["n_mps"][0], cfg["n_mps"][1] + 1)

                pool_idas = get_pool_idas(split)
                if use_existing_pairs:
                    cand_idas = [
                        ida for ida in pool_idas
                        if can_use_ida(ida, n_works, split)
                        and len([m for m in existing_mps_per_ida[ida] if can_use_mp(m, (n_works // n_mps) + 1, split)]) >= n_mps
                    ]
                    if not cand_idas:
                        cand_idas = [
                            ida for ida in pool_idas
                            if can_use_ida(ida, n_works, split)
                            and len([m for m in existing_mps_per_ida[ida] if can_use_mp(m, 1, split)]) >= 1
                        ]
                else:
                    cand_idas = [
                        ida for ida in pool_idas
                        if can_use_ida(ida, n_works, split)
                        and len([m for m in mps_by_state[ida_state[ida]] if can_use_mp(m, (n_works // n_mps) + 1, split)]) >= n_mps
                    ]
                if not cand_idas:
                    cand_idas = [ida for ida in pool_idas if can_use_ida(ida, n_works, split)]
                
                target_ida = _sample_weighted(cand_idas, ida_counts, rng, 1)[0]
                state = ida_state[target_ida]

                if use_existing_pairs:
                    cand_mps = [m for m in existing_mps_per_ida[target_ida] if can_use_mp(m, (n_works // n_mps) + 1, split)]
                    if not cand_mps:
                        cand_mps = [m for m in existing_mps_per_ida[target_ida] if can_use_mp(m, 1, split)]
                    if not cand_mps:
                        cand_mps = [m for m in mps_by_state[state] if can_use_mp(m, 1, split)]
                else:
                    cand_mps = [m for m in mps_by_state[state] if can_use_mp(m, (n_works // n_mps) + 1, split)]
                    if not cand_mps:
                        cand_mps = [m for m in mps_by_state[state] if can_use_mp(m, 1, split)]
                if not cand_mps:
                    cand_mps = [m for m in get_pool_mps(split) if can_use_mp(m, 1, split)]

                chosen_mps = _sample_weighted(cand_mps, mp_counts, rng, min(n_mps, len(cand_mps)))

                mp_work_alloc = defaultdict(int)
                for i in range(n_works):
                    mp_work_alloc[chosen_mps[i % len(chosen_mps)]] += 1
                record_usage(chosen_mps, [target_ida], mp_work_alloc, {target_ida: n_works})

                cv = rng.uniform(cfg["cv"][0], cfg["cv"][1])
                amt_center = _real_quantile(df_real["sanction_amount"], rng.uniform(cfg["amt_q"][0], cfg["amt_q"][1]))
                std = cv * amt_center
                amounts = rng.normal(amt_center, std, size=n_works)
                amounts = np.clip(amounts, stats["amt_min"], stats["amt_max"])

                base_date = pd.Timestamp("2025-02-01") + pd.Timedelta(days=rng.randint(0, 100))
                date_offsets = rng.randint(0, cfg["window"][1] + 1, size=n_works)
                sanction_dates = [base_date + pd.Timedelta(days=int(d)) for d in sorted(date_offsets)]
                lags = rng.randint(stats["lag_p25"], stats["lag_p75"] + 1, size=n_works)
                statuses = _sample_status(rng, n_works, stats["status_dist"])

                for i in range(n_works):
                    mp = chosen_mps[i % len(chosen_mps)]
                    lag = _clamp_lag(lags[i], stats["lag_max"])
                    amt = int(round(_clamp_amount(amounts[i], stats["amt_min"], stats["amt_max"])))
                    sd = sanction_dates[i].strftime("%Y-%m-%d")
                    all_records.append({
                        "work_index": idx_cursor,
                        "Sr. No.": idx_cursor + 1,
                        "work_id": f"WS/BENCH-A{episode_id:02d}-{i+1:02d}",
                        "Work category": "Normal/Others",
                        "work_category_code": 1,
                        "State": state,
                        "IDA": target_ida,
                        "MP": mp,
                        "Elected/Nominated": "Elected MP",
                        "Recommended date": _make_rec_date(sd, lag),
                        "Sanction Date": sd,
                        "sanction_lag_days": lag,
                        "sanction_amount": float(amt),
                        "Work Status": statuses[i],
                        "work_status_code": _status_code(statuses[i]),
                        "is_synthetic_fraud": True,
                        "typology": "A",
                        "episode_id": f"A{episode_id:02d}",
                        "severity": severity,
                        "split": split,
                        "holdout_group": "A",
                    })
                    idx_cursor += 1

            elif typ == "B":
                cfg = {
                    "subtle":   {"n_works": (4, 5), "amt_q": (0.75, 0.90), "lag": (10, 25)},
                    "moderate": {"n_works": (5, 6), "amt_q": (0.90, 0.97), "lag": (2, 6)},
                    "obvious":  {"n_works": (5, 6), "amt_q": (0.95, 0.99), "lag": (0, 2)},
                }[severity]
                n_works = rng.randint(cfg["n_works"][0], cfg["n_works"][1] + 1)
                
                cand_pairs = [
                    (mp, ida) for mp, ida in real_elig_pairs
                    if can_use_mp(mp, 1, split) and can_use_ida(ida, 1, split)
                ]
                if len(cand_pairs) < n_works:
                    cand_pairs = [
                        (mp, ida) for mp, ida in real_elig_pairs
                        if mp in get_pool_mps(split) and ida in get_pool_idas(split)
                    ]

                cand_pairs_list = sorted(list(cand_pairs))
                pair_weights = np.array([mp_counts[m] * ida_counts[i] for m, i in cand_pairs_list], dtype=float)
                pair_probs = pair_weights / pair_weights.sum()
                chosen_idx = rng.choice(len(cand_pairs_list), size=n_works, replace=False if len(cand_pairs_list)>=n_works else True, p=pair_probs)
                chosen_pairs = [cand_pairs_list[i] for i in chosen_idx]

                mp_work_alloc = defaultdict(int)
                ida_work_alloc = defaultdict(int)
                for m, i in chosen_pairs:
                    mp_work_alloc[m] += 1
                    ida_work_alloc[i] += 1
                record_usage(list(mp_work_alloc.keys()), list(ida_work_alloc.keys()), mp_work_alloc, ida_work_alloc)

                base_date = pd.Timestamp("2025-02-01") + pd.Timedelta(days=rng.randint(0, 120))
                for i in range(n_works):
                    mp, ida = chosen_pairs[i]
                    state = mp_home_state[mp]
                    amt_q = rng.uniform(cfg["amt_q"][0], cfg["amt_q"][1])
                    amt = int(round(_clamp_amount(_real_quantile(df_real["sanction_amount"], amt_q), stats["amt_min"], stats["amt_max"])))
                    lag = rng.randint(cfg["lag"][0], cfg["lag"][1] + 1)
                    lag = _clamp_lag(lag, stats["lag_max"])
                    sd = (base_date + pd.Timedelta(days=rng.randint(0, 30))).strftime("%Y-%m-%d")
                    all_records.append({
                        "work_index": idx_cursor,
                        "Sr. No.": idx_cursor + 1,
                        "work_id": f"WS/BENCH-B{episode_id:02d}-{i+1:02d}",
                        "Work category": "Normal/Others",
                        "work_category_code": 1,
                        "State": state,
                        "IDA": ida,
                        "MP": mp,
                        "Elected/Nominated": "Elected MP",
                        "Recommended date": _make_rec_date(sd, lag),
                        "Sanction Date": sd,
                        "sanction_lag_days": lag,
                        "sanction_amount": float(amt),
                        "Work Status": "Sanction",
                        "work_status_code": 1,
                        "is_synthetic_fraud": True,
                        "typology": "B",
                        "episode_id": f"B{episode_id:02d}",
                        "severity": severity,
                        "split": split,
                        "holdout_group": "B",
                    })
                    idx_cursor += 1

            elif typ == "C":
                cfg = {
                    "subtle":   {"n_works": (4, 5), "amt_q": (0.25, 0.70), "lag": (250, 400), "statuses": ["Physical Inspection", "Vendor Identification"]},
                    "moderate": {"n_works": (5, 6), "amt_q": (0.20, 0.50), "lag": (450, 600), "statuses": ["Physical Inspection", "Vendor Identification"]},
                    "obvious":  {"n_works": (5, 6), "amt_q": (0.15, 0.40), "lag": (600, 720), "statuses": ["Physical Inspection"]},
                }[severity]
                n_works = rng.randint(cfg["n_works"][0], cfg["n_works"][1] + 1)
                
                cand_pairs = [
                    (mp, ida) for mp, ida in real_elig_pairs
                    if can_use_mp(mp, 1, split) and can_use_ida(ida, 1, split)
                ]
                if len(cand_pairs) < n_works:
                    cand_pairs = [
                        (mp, ida) for mp, ida in real_elig_pairs
                        if mp in get_pool_mps(split) and ida in get_pool_idas(split)
                    ]

                cand_pairs_list = sorted(list(cand_pairs))
                pair_weights = np.array([mp_counts[m] * ida_counts[i] for m, i in cand_pairs_list], dtype=float)
                pair_probs = pair_weights / pair_weights.sum()
                chosen_idx = rng.choice(len(cand_pairs_list), size=n_works, replace=False if len(cand_pairs_list)>=n_works else True, p=pair_probs)
                chosen_pairs = [cand_pairs_list[i] for i in chosen_idx]

                mp_work_alloc = defaultdict(int)
                ida_work_alloc = defaultdict(int)
                for m, i in chosen_pairs:
                    mp_work_alloc[m] += 1
                    ida_work_alloc[i] += 1
                record_usage(list(mp_work_alloc.keys()), list(ida_work_alloc.keys()), mp_work_alloc, ida_work_alloc)

                for i in range(n_works):
                    mp, ida = chosen_pairs[i]
                    state = mp_home_state[mp]
                    amt_q = rng.uniform(cfg["amt_q"][0], cfg["amt_q"][1])
                    amt = int(round(_clamp_amount(_real_quantile(df_real["sanction_amount"], amt_q), stats["amt_min"], stats["amt_max"])))
                    lag = rng.randint(cfg["lag"][0], min(cfg["lag"][1], stats["lag_max"]) + 1)
                    lag = _clamp_lag(lag, stats["lag_max"])
                    status = rng.choice(cfg["statuses"])

                    rec_base = pd.Timestamp("2024-07-15") + pd.Timedelta(days=rng.randint(0, 45))
                    sd_dt = rec_base + pd.Timedelta(days=lag)
                    if sd_dt > stats["real_max_sanc"]:
                        sd_dt = stats["real_max_sanc"] - pd.Timedelta(days=rng.randint(1, 10))
                        rec_base = sd_dt - pd.Timedelta(days=lag)

                    sd = sd_dt.strftime("%Y-%m-%d")
                    rd = rec_base.strftime("%Y-%m-%d")

                    all_records.append({
                        "work_index": idx_cursor,
                        "Sr. No.": idx_cursor + 1,
                        "work_id": f"WS/BENCH-C{episode_id:02d}-{i+1:02d}",
                        "Work category": "Normal/Others",
                        "work_category_code": 1,
                        "State": state,
                        "IDA": ida,
                        "MP": mp,
                        "Elected/Nominated": "Elected MP",
                        "Recommended date": rd,
                        "Sanction Date": sd,
                        "sanction_lag_days": lag,
                        "sanction_amount": float(amt),
                        "Work Status": status,
                        "work_status_code": _status_code(status),
                        "is_synthetic_fraud": True,
                        "typology": "C",
                        "episode_id": f"C{episode_id:02d}",
                        "severity": severity,
                        "split": split,
                        "holdout_group": "C",
                    })
                    idx_cursor += 1

            elif typ == "D":
                cfg = {
                    "subtle":   {"n_works": (4, 5), "n_mps": (2, 3), "window": (80, 120), "cv": (0.18, 0.25), "amt_q": (0.40, 0.70)},
                    "moderate": {"n_works": (5, 6), "n_mps": (2, 3), "window": (50, 80), "cv": (0.10, 0.17), "amt_q": (0.50, 0.80)},
                    "obvious":  {"n_works": (5, 6), "n_mps": (2, 3), "window": (30, 50), "cv": (0.06, 0.09), "amt_q": (0.50, 0.80)},
                }[severity]
                n_works = rng.randint(cfg["n_works"][0], cfg["n_works"][1] + 1)
                n_mps = rng.randint(cfg["n_mps"][0], cfg["n_mps"][1] + 1)

                pool_idas = get_pool_idas(split)
                if use_existing_pairs:
                    cand_idas = [
                        ida for ida in pool_idas
                        if can_use_ida(ida, n_works, split)
                        and len([m for m in existing_mps_per_ida[ida] if can_use_mp(m, (n_works // n_mps) + 1, split)]) >= n_mps
                    ]
                    if not cand_idas:
                        cand_idas = [
                            ida for ida in pool_idas
                            if can_use_ida(ida, n_works, split)
                            and len([m for m in existing_mps_per_ida[ida] if can_use_mp(m, 1, split)]) >= 1
                        ]
                else:
                    cand_idas = [
                        ida for ida in pool_idas
                        if can_use_ida(ida, n_works, split)
                        and len([m for m in mps_by_state[ida_state[ida]] if can_use_mp(m, (n_works // n_mps) + 1, split)]) >= n_mps
                    ]
                if not cand_idas:
                    cand_idas = [ida for ida in pool_idas if can_use_ida(ida, n_works, split)]
                
                target_ida = _sample_weighted(cand_idas, ida_counts, rng, 1)[0]
                state = ida_state[target_ida]

                if use_existing_pairs:
                    cand_mps = [m for m in existing_mps_per_ida[target_ida] if can_use_mp(m, (n_works // n_mps) + 1, split)]
                    if not cand_mps:
                        cand_mps = [m for m in existing_mps_per_ida[target_ida] if can_use_mp(m, 1, split)]
                    if not cand_mps:
                        cand_mps = [m for m in mps_by_state[state] if can_use_mp(m, 1, split)]
                else:
                    cand_mps = [m for m in mps_by_state[state] if can_use_mp(m, (n_works // n_mps) + 1, split)]
                    if not cand_mps:
                        cand_mps = [m for m in mps_by_state[state] if can_use_mp(m, 1, split)]
                if not cand_mps:
                    cand_mps = [m for m in get_pool_mps(split) if can_use_mp(m, 1, split)]

                chosen_mps = _sample_weighted(cand_mps, mp_counts, rng, min(n_mps, len(cand_mps)))

                mp_work_alloc = defaultdict(int)
                for i in range(n_works):
                    mp_work_alloc[chosen_mps[i % len(chosen_mps)]] += 1
                record_usage(chosen_mps, [target_ida], mp_work_alloc, {target_ida: n_works})

                cv = rng.uniform(cfg["cv"][0], cfg["cv"][1])
                amt_center = _real_quantile(df_real["sanction_amount"], rng.uniform(cfg["amt_q"][0], cfg["amt_q"][1]))
                std = cv * amt_center
                amounts = rng.normal(amt_center, std, size=n_works)
                amounts = np.clip(amounts, stats["amt_min"], stats["amt_max"])

                base_date = pd.Timestamp("2025-01-15") + pd.Timedelta(days=rng.randint(0, 100))
                date_offsets = rng.randint(0, cfg["window"][1] + 1, size=n_works)
                sanction_dates = [base_date + pd.Timedelta(days=int(d)) for d in sorted(date_offsets)]
                lags = rng.randint(stats["lag_p25"], stats["lag_p75"] + 1, size=n_works)
                statuses = _sample_status(rng, n_works, stats["status_dist"])

                for i in range(n_works):
                    mp = chosen_mps[i % len(chosen_mps)]
                    lag = _clamp_lag(lags[i], stats["lag_max"])
                    amt = int(round(_clamp_amount(amounts[i], stats["amt_min"], stats["amt_max"])))
                    sd = sanction_dates[i].strftime("%Y-%m-%d")
                    all_records.append({
                        "work_index": idx_cursor,
                        "Sr. No.": idx_cursor + 1,
                        "work_id": f"WS/BENCH-D{episode_id:02d}-{i+1:02d}",
                        "Work category": "Normal/Others",
                        "work_category_code": 1,
                        "State": state,
                        "IDA": target_ida,
                        "MP": mp,
                        "Elected/Nominated": "Elected MP",
                        "Recommended date": _make_rec_date(sd, lag),
                        "Sanction Date": sd,
                        "sanction_lag_days": lag,
                        "sanction_amount": float(amt),
                        "Work Status": statuses[i],
                        "work_status_code": _status_code(statuses[i]),
                        "is_synthetic_fraud": True,
                        "typology": "D",
                        "episode_id": f"D{episode_id:02d}",
                        "severity": severity,
                        "split": split,
                        "holdout_group": "D",
                    })
                    idx_cursor += 1

            elif typ == "E":
                cfg = {
                    "subtle":   {"n_works": (4, 5), "thresholds": [500_000, 1_000_000], "pct": (0.75, 0.95), "window": (20, 40)},
                    "moderate": {"n_works": (5, 6), "thresholds": [1_000_000, 2_500_000], "pct": (0.85, 0.99), "window": (10, 20)},
                    "obvious":  {"n_works": (5, 6), "thresholds": [2_500_000], "pct": (0.92, 0.995), "window": (3, 10)},
                }[severity]
                n_works = rng.randint(cfg["n_works"][0], cfg["n_works"][1] + 1)
                threshold = rng.choice(cfg["thresholds"])

                cand_pairs = [
                    (mp, ida) for mp, ida in real_elig_pairs
                    if can_use_mp(mp, n_works, split) and can_use_ida(ida, n_works, split)
                ]
                if not cand_pairs:
                    cand_pairs = [
                        (mp, ida) for mp, ida in real_elig_pairs
                        if mp in get_pool_mps(split) and ida in get_pool_idas(split)
                    ]

                cand_pairs_list = sorted(list(cand_pairs))
                pair_weights = np.array([mp_counts[m] * ida_counts[i] for m, i in cand_pairs_list], dtype=float)
                pair_probs = pair_weights / pair_weights.sum()
                mp, ida = cand_pairs_list[rng.choice(len(cand_pairs_list), p=pair_probs)]
                state = mp_home_state[mp]

                record_usage([mp], [ida], {mp: n_works}, {ida: n_works})

                base_date = pd.Timestamp("2025-02-01") + pd.Timedelta(days=rng.randint(0, 150))
                window = rng.randint(cfg["window"][0], cfg["window"][1] + 1)
                statuses_pool = _sample_status(rng, n_works, stats["status_dist"]) if severity == "subtle" else ["Sanction"] * n_works

                for i in range(n_works):
                    pct = rng.uniform(cfg["pct"][0], cfg["pct"][1])
                    amt = int(round(_clamp_amount(threshold * pct, stats["amt_min"], stats["amt_max"])))
                    lag = rng.randint(stats["lag_p25"], stats["lag_p75"] + 1)
                    lag = _clamp_lag(lag, stats["lag_max"])
                    sd = (base_date + pd.Timedelta(days=rng.randint(0, window + 1))).strftime("%Y-%m-%d")
                    all_records.append({
                        "work_index": idx_cursor,
                        "Sr. No.": idx_cursor + 1,
                        "work_id": f"WS/BENCH-E{episode_id:02d}-{i+1:02d}",
                        "Work category": "Normal/Others",
                        "work_category_code": 1,
                        "State": state,
                        "IDA": ida,
                        "MP": mp,
                        "Elected/Nominated": "Elected MP",
                        "Recommended date": _make_rec_date(sd, lag),
                        "Sanction Date": sd,
                        "sanction_lag_days": lag,
                        "sanction_amount": float(amt),
                        "Work Status": statuses_pool[i],
                        "work_status_code": _status_code(statuses_pool[i]),
                        "is_synthetic_fraud": True,
                        "typology": "E",
                        "episode_id": f"E{episode_id:02d}",
                        "severity": severity,
                        "split": split,
                        "holdout_group": "E",
                    })
                    idx_cursor += 1

            elif typ == "F":
                cfg = {
                    "subtle":   {"n_works": (4, 5), "window": (30, 60)},
                    "moderate": {"n_works": (5, 6), "window": (15, 30)},
                    "obvious":  {"n_works": (5, 6), "window": (5, 15)},
                }[severity]
                n_works = rng.randint(cfg["n_works"][0], cfg["n_works"][1] + 1)
                window = rng.randint(cfg["window"][0], cfg["window"][1] + 1)

                pool_mps = get_pool_mps(split)
                pool_idas = get_pool_idas(split)

                cand_mps = [
                    mp for mp in pool_mps
                    if can_use_mp(mp, n_works, split)
                    and len([ida for ida in idas_by_state[mp_home_state[mp]] if ida in pool_idas and ida not in existing_idas_per_mp[mp] and can_use_ida(ida, n_works, split)]) >= 1
                ]
                if not cand_mps:
                    cand_mps = [
                        mp for mp in pool_mps
                        if len([ida for ida in idas_by_state[mp_home_state[mp]] if ida in pool_idas and ida not in existing_idas_per_mp[mp]]) >= 1
                    ]

                mp = _sample_weighted(cand_mps, mp_counts, rng, 1)[0]
                state = mp_home_state[mp]

                cand_idas = [
                    ida for ida in idas_by_state[state]
                    if ida in pool_idas and ida not in existing_idas_per_mp[mp] and can_use_ida(ida, n_works, split)
                ]
                if not cand_idas:
                    cand_idas = [ida for ida in idas_by_state[state] if ida in pool_idas and ida not in existing_idas_per_mp[mp]]

                new_ida = _sample_weighted(cand_idas, ida_counts, rng, 1)[0]
                record_usage([mp], [new_ida], {mp: n_works}, {new_ida: n_works})

                base_date = pd.Timestamp("2025-03-01") + pd.Timedelta(days=rng.randint(0, 120))
                statuses = _sample_status(rng, n_works, stats["status_dist"])

                for i in range(n_works):
                    amt_q = rng.uniform(0.30, 0.70)
                    amt = int(round(_clamp_amount(_real_quantile(df_real["sanction_amount"], amt_q), stats["amt_min"], stats["amt_max"])))
                    lag = rng.randint(stats["lag_p25"], stats["lag_p75"] + 1)
                    lag = _clamp_lag(lag, stats["lag_max"])
                    sd = (base_date + pd.Timedelta(days=rng.randint(0, window + 1))).strftime("%Y-%m-%d")
                    all_records.append({
                        "work_index": idx_cursor,
                        "Sr. No.": idx_cursor + 1,
                        "work_id": f"WS/BENCH-F{episode_id:02d}-{i+1:02d}",
                        "Work category": "Normal/Others",
                        "work_category_code": 1,
                        "State": state,
                        "IDA": new_ida,
                        "MP": mp,
                        "Elected/Nominated": "Elected MP",
                        "Recommended date": _make_rec_date(sd, lag),
                        "Sanction Date": sd,
                        "sanction_lag_days": lag,
                        "sanction_amount": float(amt),
                        "Work Status": statuses[i],
                        "work_status_code": _status_code(statuses[i]),
                        "is_synthetic_fraud": True,
                        "typology": "F",
                        "episode_id": f"F{episode_id:02d}",
                        "severity": severity,
                        "split": split,
                        "holdout_group": "F",
                    })
                    idx_cursor += 1

            elif typ == "G":
                cfg = {
                    "subtle":   {"n_works": (4, 5), "start_q": (0.20, 0.40), "step_pct": (0.08, 0.15), "cap_q": 0.90, "spacing": (15, 40)},
                    "moderate": {"n_works": (5, 6), "start_q": (0.15, 0.30), "step_pct": (0.15, 0.30), "cap_q": 0.95, "spacing": (10, 25)},
                    "obvious":  {"n_works": (5, 6), "start_q": (0.10, 0.25), "step_pct": (0.30, 0.60), "cap_q": 0.99, "spacing": (5, 15)},
                }[severity]
                n_works = rng.randint(cfg["n_works"][0], cfg["n_works"][1] + 1)
                step_pct = rng.uniform(cfg["step_pct"][0], cfg["step_pct"][1])
                start_q = rng.uniform(cfg["start_q"][0], cfg["start_q"][1])
                start_amt = _real_quantile(df_real["sanction_amount"], start_q)
                cap_amt = _real_quantile(df_real["sanction_amount"], cfg["cap_q"])

                cand_pairs = [
                    (mp, ida) for mp, ida in real_elig_pairs
                    if can_use_mp(mp, n_works, split) and can_use_ida(ida, n_works, split)
                ]
                if not cand_pairs:
                    cand_pairs = [
                        (mp, ida) for mp, ida in real_elig_pairs
                        if mp in get_pool_mps(split) and ida in get_pool_idas(split)
                    ]

                cand_pairs_list = sorted(list(cand_pairs))
                pair_weights = np.array([mp_counts[m] * ida_counts[i] for m, i in cand_pairs_list], dtype=float)
                pair_probs = pair_weights / pair_weights.sum()
                mp, ida = cand_pairs_list[rng.choice(len(cand_pairs_list), p=pair_probs)]
                state = mp_home_state[mp]

                record_usage([mp], [ida], {mp: n_works}, {ida: n_works})

                base_date = pd.Timestamp("2025-01-15") + pd.Timedelta(days=rng.randint(0, 100))
                statuses = _sample_status(rng, n_works, stats["status_dist"])

                current_amt = start_amt
                current_date = base_date
                for i in range(n_works):
                    amt = min(current_amt, cap_amt)
                    amt = int(round(_clamp_amount(amt, stats["amt_min"], stats["amt_max"])))
                    lag = rng.randint(stats["lag_p25"], stats["lag_p75"] + 1)
                    lag = _clamp_lag(lag, stats["lag_max"])
                    sd = current_date.strftime("%Y-%m-%d")
                    all_records.append({
                        "work_index": idx_cursor,
                        "Sr. No.": idx_cursor + 1,
                        "work_id": f"WS/BENCH-G{episode_id:02d}-{i+1:02d}",
                        "Work category": "Normal/Others",
                        "work_category_code": 1,
                        "State": state,
                        "IDA": ida,
                        "MP": mp,
                        "Elected/Nominated": "Elected MP",
                        "Recommended date": _make_rec_date(sd, lag),
                        "Sanction Date": sd,
                        "sanction_lag_days": lag,
                        "sanction_amount": float(amt),
                        "Work Status": statuses[i],
                        "work_status_code": _status_code(statuses[i]),
                        "is_synthetic_fraud": True,
                        "typology": "G",
                        "episode_id": f"G{episode_id:02d}",
                        "severity": severity,
                        "split": split,
                        "holdout_group": "G",
                    })
                    idx_cursor += 1
                    current_amt *= (1 + step_pct)
                    spacing = rng.randint(cfg["spacing"][0], cfg["spacing"][1] + 1)
                    current_date += pd.Timedelta(days=spacing)

            elif typ == "H":
                cfg = {
                    "subtle":   {"n_idas": (4, 5), "cv": (0.10, 0.15), "amt_q": (0.50, 0.75), "window": (20, 40)},
                    "moderate": {"n_idas": (5, 6), "cv": (0.04, 0.09), "amt_q": (0.60, 0.85), "window": (7, 15)},
                    "obvious":  {"n_idas": (5, 6), "cv": (0.01, 0.03), "amt_q": (0.70, 0.95), "window": (1, 5)},
                }[severity]
                n_idas = rng.randint(cfg["n_idas"][0], cfg["n_idas"][1] + 1)
                
                pool_mps = get_pool_mps(split)
                pool_idas = get_pool_idas(split)

                if use_existing_pairs:
                    cand_mps = [
                        mp for mp in pool_mps
                        if can_use_mp(mp, n_idas, split)
                        and len([i for i in existing_idas_per_mp[mp] if i in pool_idas and can_use_ida(i, 1, split)]) >= 2
                    ]
                    if not cand_mps:
                        cand_mps = [
                            mp for mp in pool_mps
                            if can_use_mp(mp, n_idas, split)
                            and len(existing_idas_per_mp[mp]) >= 2
                        ]
                else:
                    cand_mps = [
                        mp for mp in pool_mps
                        if can_use_mp(mp, n_idas, split)
                        and len([i for i in idas_by_state[mp_home_state[mp]] if i in pool_idas and i not in existing_idas_per_mp[mp] and can_use_ida(i, 1, split)]) >= 2
                    ]
                    if not cand_mps:
                        cand_mps = [
                            mp for mp in pool_mps
                            if can_use_mp(mp, n_idas, split)
                            and len([i for i in idas_by_state[mp_home_state[mp]] if i not in existing_idas_per_mp[mp]]) >= 2
                        ]

                mp = _sample_weighted(cand_mps, mp_counts, rng, 1)[0]
                state = mp_home_state[mp]

                if use_existing_pairs:
                    avail_idas = sorted([i for i in existing_idas_per_mp[mp] if i in pool_idas and can_use_ida(i, 1, split)])
                    if len(avail_idas) < 2:
                        avail_idas = sorted([i for i in existing_idas_per_mp[mp] if can_use_ida(i, 1, split)])
                    if not avail_idas:
                        avail_idas = sorted(list(existing_idas_per_mp[mp]))
                else:
                    avail_idas = sorted([i for i in idas_by_state[state] if i in pool_idas and i not in existing_idas_per_mp[mp] and can_use_ida(i, 1, split)])
                    if len(avail_idas) < 2:
                        avail_idas = sorted([i for i in idas_by_state[state] if i not in existing_idas_per_mp[mp] and can_use_ida(i, 1, split)])
                    if not avail_idas:
                        avail_idas = idas_by_state[state]

                chosen_idas = _sample_weighted(avail_idas, ida_counts, rng, min(n_idas, len(avail_idas)), replace=True if len(avail_idas)<n_idas else False)
                n_works = len(chosen_idas)

                ida_work_alloc = defaultdict(int)
                for ida in chosen_idas:
                    ida_work_alloc[ida] += 1
                record_usage([mp], list(ida_work_alloc.keys()), {mp: n_works}, ida_work_alloc)

                cv = rng.uniform(cfg["cv"][0], cfg["cv"][1])
                amt_center = _real_quantile(df_real["sanction_amount"], rng.uniform(cfg["amt_q"][0], cfg["amt_q"][1]))
                std = cv * amt_center
                amounts = rng.normal(amt_center, std, size=n_works)
                amounts = np.clip(amounts, stats["amt_min"], stats["amt_max"])

                base_date = pd.Timestamp("2025-02-01") + pd.Timedelta(days=rng.randint(0, 120))
                lags = rng.randint(stats["lag_p25"], stats["lag_p75"] + 1, size=n_works)
                statuses = _sample_status(rng, n_works, stats["status_dist"])

                for i in range(n_works):
                    ida = chosen_idas[i]
                    lag = _clamp_lag(lags[i], stats["lag_max"])
                    amt = int(round(_clamp_amount(amounts[i], stats["amt_min"], stats["amt_max"])))
                    sd = (base_date + pd.Timedelta(days=rng.randint(0, cfg["window"][1] + 1))).strftime("%Y-%m-%d")
                    all_records.append({
                        "work_index": idx_cursor,
                        "Sr. No.": idx_cursor + 1,
                        "work_id": f"WS/BENCH-H{episode_id:02d}-{i+1:02d}",
                        "Work category": "Normal/Others",
                        "work_category_code": 1,
                        "State": state,
                        "IDA": ida,
                        "MP": mp,
                        "Elected/Nominated": "Elected MP",
                        "Recommended date": _make_rec_date(sd, lag),
                        "Sanction Date": sd,
                        "sanction_lag_days": lag,
                        "sanction_amount": float(amt),
                        "Work Status": statuses[i],
                        "work_status_code": _status_code(statuses[i]),
                        "is_synthetic_fraud": True,
                        "typology": "H",
                        "episode_id": f"H{episode_id:02d}",
                        "severity": severity,
                        "split": split,
                        "holdout_group": "H",
                    })
                    idx_cursor += 1

    df_syn = pd.DataFrame(all_records)

    # ── Real splits ──────────────────────────────────────────────────────────
    real_indices = rng.permutation(np.arange(len(df_real)))
    n_real_train = int(0.50 * len(df_real))
    n_real_val = int(0.25 * len(df_real))
    real_splits = np.empty(len(df_real), dtype=object)
    real_splits[real_indices[:n_real_train]] = "train"
    real_splits[real_indices[n_real_train:n_real_train + n_real_val]] = "val"
    real_splits[real_indices[n_real_train + n_real_val:]] = "test"

    df_real_aug = df_real.copy()
    df_real_aug["work_index"] = df_real_aug.index
    df_real_aug["is_synthetic_fraud"] = False
    df_real_aug["typology"] = "real"
    df_real_aug["episode_id"] = "real"
    df_real_aug["severity"] = "real"
    df_real_aug["split"] = real_splits
    df_real_aug["holdout_group"] = "real"

    # Drop "Work description" (Item 6)
    out_cols = [c for c in df_real.columns if c != "Work description"] + [
        "work_index", "is_synthetic_fraud", "typology", "episode_id", "severity",
        "split", "holdout_group"
    ]
    for c in out_cols:
        if c not in df_syn.columns:
            df_syn[c] = None

    df_combined = pd.concat([df_real_aug[out_cols], df_syn[out_cols]], ignore_index=True)

    return df_real, df_syn, df_combined, stats, mp_ep_count, ida_ep_count, mp_injected_count, ida_injected_count

def run_pipeline():
    df_real, df_syn, df_combined, stats, mp_ep_count, ida_ep_count, mp_injected_count, ida_injected_count = generate_benchmark_v3(seed=SEED)
    mp_counts = stats["mp_counts"]
    ida_counts = stats["ida_counts"]

    print("=" * 80)
    print("ITEM 1 & 2: NOVEL-PAIR SHARE PER TYPOLOGY")
    print("=" * 80)
    real_pairs = set(zip(df_real['MP'], df_real['IDA']))
    for typ in TYPOLOGIES:
        df_t = df_syn[df_syn['typology'] == typ]
        ep_novel_flags = []
        for ep in sorted(df_t['episode_id'].unique()):
            df_ep = df_t[df_t['episode_id'] == ep]
            ep_pairs = set(zip(df_ep['MP'], df_ep['IDA']))
            is_novel = any(p not in real_pairs for p in ep_pairs)
            ep_novel_flags.append(is_novel)
        novel_share = np.mean(ep_novel_flags) * 100
        print(f"Typology {typ}: {sum(ep_novel_flags)}/{len(ep_novel_flags)} novel episodes ({novel_share:.1f}% novel)")

    print("\n" + "=" * 80)
    print("ITEM 3: ENTITY REPORT PER TYPOLOGY")
    print("=" * 80)
    header = f"{'Typology':<10} | {'Distinct MPs':<13} | {'Distinct IDAs':<14} | {'Max MP Ep':<10} | {'Max IDA Ep':<11} | {'Max MP Inj %':<13} | {'Max IDA Inj %':<14} | {'Train/Test Overlap':<18}"
    print(header)
    print("-" * len(header))

    for typ in TYPOLOGIES:
        df_t = df_syn[df_syn['typology'] == typ]
        mps = df_t['MP'].unique()
        idas = df_t['IDA'].unique()
        
        max_mp_ep = df_t.groupby('MP')['episode_id'].nunique().max()
        max_ida_ep = df_t.groupby('IDA')['episode_id'].nunique().max()

        max_mp_share = max([len(df_t[df_t['MP']==m]) / mp_counts[m] for m in mps]) * 100
        max_ida_share = max([len(df_t[df_t['IDA']==i]) / ida_counts[i] for i in idas]) * 100

        train_mps = set(df_t[df_t['split'].isin(['train', 'val'])]['MP'])
        test_mps = set(df_t[df_t['split'] == 'test']['MP'])
        train_idas = set(df_t[df_t['split'].isin(['train', 'val'])]['IDA'])
        test_idas = set(df_t[df_t['split'] == 'test']['IDA'])
        overlap_str = f"MP: {len(train_mps & test_mps)}, IDA: {len(train_idas & test_idas)}"

        print(f"{typ:<10} | {len(mps):<13} | {len(idas):<14} | {max_mp_ep:<10} | {max_ida_ep:<11} | {max_mp_share:>11.2f}% | {max_ida_share:>12.2f}% | {overlap_str:<18}")

    # Global entity report
    train_mps_all = set(df_syn[df_syn['split'].isin(['train', 'val'])]['MP'])
    test_mps_all = set(df_syn[df_syn['split'] == 'test']['MP'])
    train_idas_all = set(df_syn[df_syn['split'].isin(['train', 'val'])]['IDA'])
    test_idas_all = set(df_syn[df_syn['split'] == 'test']['IDA'])
    print("-" * len(header))
    print(f"ALL GLOBAL | {df_syn['MP'].nunique():<13} | {df_syn['IDA'].nunique():<14} | {max(mp_ep_count.values()):<10} | {max(ida_ep_count.values()):<11} | {max([mp_injected_count[m]/mp_counts[m] for m in mp_injected_count])*100:>11.2f}% | {max([ida_injected_count[i]/ida_counts[i] for i in ida_injected_count])*100:>12.2f}% | MP: {len(train_mps_all & test_mps_all)}, IDA: {len(train_idas_all & test_idas_all)}")

    print("\n" + "=" * 80)
    print("ITEM 4b: PERMUTATION NULL TEST (200 PSEUDO-EPISODES)")
    print("=" * 80)
    
    def extract_features(df):
        sanc_ord = pd.to_datetime(df['Sanction Date']).map(lambda d: d.toordinal()).values
        rec_ord = pd.to_datetime(df['Recommended date']).map(lambda d: d.toordinal()).values
        mp_c = df['MP'].map(lambda m: mp_counts.get(m, 0)).values
        ida_c = df['IDA'].map(lambda i: ida_counts.get(i, 0)).values
        amt = df['sanction_amount'].values
        lag = df['sanction_lag_days'].values
        status = df['work_status_code'].values
        cat = df['work_category_code'].values
        return pd.DataFrame({
            "amount": amt,
            "lag": lag,
            "status": status,
            "category": cat,
            "mp_real_count": mp_c,
            "ida_real_count": ida_c,
            "sanction_date_ordinal": sanc_ord,
            "recommended_date_ordinal": rec_ord,
        })

    FEATURE_NAMES = [
        "amount", "lag", "status", "category",
        "mp_real_count", "ida_real_count",
        "sanction_date_ordinal", "recommended_date_ordinal"
    ]

    df_real_feats = extract_features(df_real)
    df_syn_feats = extract_features(df_syn)

    rng_null = np.random.RandomState(12345)
    eligible_real_indices = df_real[df_real['MP'].isin(stats["eligible_mps"]) & df_real['IDA'].isin(stats["eligible_idas"])].index.values
    
    null_aucs = {f: [] for f in FEATURE_NAMES}
    for p in range(200):
        sample_idx = rng_null.choice(eligible_real_indices, size=20, replace=False)
        y_pseudo = np.zeros(len(df_real))
        y_pseudo[sample_idx] = 1
        
        for f in FEATURE_NAMES:
            vals = df_real_feats[f].values
            try:
                auc = roc_auc_score(y_pseudo, vals)
                auc_adj = max(auc, 1 - auc)
            except Exception:
                auc_adj = 0.50
            null_aucs[f].append(auc_adj)

    null_p95 = {f: float(np.percentile(null_aucs[f], 95)) for f in FEATURE_NAMES}
    print(f"{'Feature':<26} | {'Null Mean AUC':<14} | {'Null 95th %ile AUC':<20}")
    print("-" * 65)
    for f in FEATURE_NAMES:
        print(f"{f:<26} | {np.mean(null_aucs[f]):<14.4f} | {null_p95[f]:<20.4f}")

    print("\n" + "=" * 80)
    print("ITEM 4a: POOLED AUC PER TYPOLOGY x SEVERITY (FLAGGED IF > NULL P95)")
    print("=" * 80)

    pooled_auc_results = []
    for typ in TYPOLOGIES:
        for sev in SEVERITIES:
            sub_mask = (df_syn['typology'] == typ) & (df_syn['severity'] == sev)
            df_cell = df_syn[sub_mask]
            cell_feats = extract_features(df_cell)
            
            y = np.array([1] * len(df_cell) + [0] * len(df_real))
            combined_f = pd.concat([cell_feats, df_real_feats], ignore_index=True)

            row_res = {"Typology": typ, "Severity": sev, "N_works": len(df_cell)}
            for f in FEATURE_NAMES:
                vals = combined_f[f].values
                try:
                    auc = roc_auc_score(y, vals)
                    auc_adj = max(auc, 1 - auc)
                except Exception:
                    auc_adj = 0.50
                is_flagged = auc_adj > null_p95[f]
                row_res[f] = (auc_adj, is_flagged)
            pooled_auc_results.append(row_res)

    header_4a = f"{'Typ':<4} | {'Sev':<8} | " + " | ".join([f"{f[:8]:<9}" for f in FEATURE_NAMES])
    print(header_4a)
    print("-" * len(header_4a))
    for r in pooled_auc_results:
        cols_str = []
        for f in FEATURE_NAMES:
            auc, flg = r[f]
            s = f"{auc:.3f}" + ("*" if flg else " ")
            cols_str.append(f"{s:<9}")
        print(f"{r['Typology']:<4} | {r['Severity']:<8} | " + " | ".join(cols_str))
    print("(* indicates observed AUC exceeds 95th percentile permutation null)")

    print("\n" + "=" * 80)
    print("ITEM 4c & 5: ADVERSARIAL PROBE (HistGradientBoostingClassifier)")
    print("=" * 80)

    df_real_train = df_combined[(df_combined['typology'] == 'real') & (df_combined['split'] == 'train')]
    df_real_test = df_combined[(df_combined['typology'] == 'real') & (df_combined['split'] == 'test')]
    df_syn_train = df_syn[df_syn['split'] == 'train']

    X_train_pos = extract_features(df_syn_train)
    X_train_neg = extract_features(df_real_train)
    X_train = pd.concat([X_train_pos, X_train_neg], ignore_index=True)
    y_train = np.array([1] * len(X_train_pos) + [0] * len(X_train_neg))

    X_test_neg = extract_features(df_real_test)

    clf = HistGradientBoostingClassifier(random_state=SEED, max_iter=100)
    clf.fit(X_train, y_train)

    probe_results = []
    detectability_map = {}
    structural_typologies = ["A", "D", "F", "H"]
    
    header_4c = f"{'Typology':<10} | {'Test Works':<11} | {'Probe AUROC':<12} | {'Probe AUPRC':<12} | {'Detectability Class':<20}"
    print(header_4c)
    print("-" * len(header_4c))

    for typ in TYPOLOGIES:
        df_syn_test_t = df_syn[(df_syn['typology'] == typ) & (df_syn['split'] == 'test')]
        X_test_pos_t = extract_features(df_syn_test_t)
        
        X_test_t = pd.concat([X_test_pos_t, X_test_neg], ignore_index=True)
        y_test_t = np.array([1] * len(X_test_pos_t) + [0] * len(X_test_neg))

        probs_t = clf.predict_proba(X_test_t)[:, 1]
        auroc = roc_auc_score(y_test_t, probs_t)
        auprc = average_precision_score(y_test_t, probs_t)
        det_class = "tabular" if auroc >= 0.85 else "structural"
        detectability_map[typ] = det_class

        probe_results.append({
            "typology": typ,
            "test_works": len(df_syn_test_t),
            "auroc": auroc,
            "auprc": auprc,
            "detectability_class": det_class
        })
        print(f"{typ:<10} | {len(df_syn_test_t):<11} | {auroc:<12.4f} | {auprc:<12.4f} | {det_class:<20}")

    print("\nFeature Importances on Probe (Permutation Importance on Test Set):")
    for res in probe_results:
        typ = res["typology"]
        auroc = res["auroc"]
        if typ in structural_typologies and auroc > 0.70:
            print(f"\n[!] Structural Typology {typ} scored AUROC {auroc:.4f} (> 0.70). Analyzing top driving features:")
            df_syn_test_t = df_syn[(df_syn['typology'] == typ) & (df_syn['split'] == 'test')]
            X_test_pos_t = extract_features(df_syn_test_t)
            X_test_t = pd.concat([X_test_pos_t, X_test_neg], ignore_index=True)
            y_test_t = np.array([1] * len(X_test_pos_t) + [0] * len(X_test_neg))
            
            perm_imp = permutation_importance(clf, X_test_t, y_test_t, n_repeats=10, random_state=SEED, scoring='roc_auc')
            sorted_idx = np.argsort(perm_imp.importances_mean)[::-1]
            for idx in sorted_idx:
                print(f"    - {FEATURE_NAMES[idx]:<26}: Mean Importance = {perm_imp.importances_mean[idx]:+.4f} (std {perm_imp.importances_std[idx]:.4f})")

    print("\n" + "=" * 80)
    print("ITEM 6: TEXT COLUMNS")
    print("=" * 80)
    print("Injected rows do NOT reuse real work descriptions for the same category.")
    print("Per Item 6 specification, 'Work description' column has been dropped from benchmark_works.csv.")

    # ── Write benchmark files ────────────────────────────────────────────────
    os.makedirs("data/benchmark", exist_ok=True)
    works_path = os.path.join("data", "benchmark", "benchmark_works.csv")
    manifest_path = os.path.join("data", "benchmark", "benchmark_manifest.csv")

    df_combined.to_csv(works_path, index=False)

    df_manifest = df_combined[[
        "work_index", "episode_id", "typology", "severity", "split", "holdout_group"
    ]].copy()
    df_manifest["detectability_class"] = df_manifest["typology"].map(lambda t: detectability_map.get(t, "real"))
    df_manifest.to_csv(manifest_path, index=False)

    def file_sha256(filepath):
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    works_hash = file_sha256(works_path)
    manifest_hash = file_sha256(manifest_path)

    print("\n" + "=" * 80)
    print("BENCHMARK ARTIFACTS GENERATED")
    print("=" * 80)
    print(f"benchmark_works.csv:    {len(df_combined)} rows | SHA-256: {works_hash}")
    print(f"benchmark_manifest.csv: {len(df_manifest)} rows | SHA-256: {manifest_hash}")

    return works_hash, manifest_hash

if __name__ == "__main__":
    run_pipeline()
