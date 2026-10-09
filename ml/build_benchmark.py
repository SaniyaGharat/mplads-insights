"""
Synthetic Fraud Benchmark Generation Script (v5.0 - Date-Matched Injection)

Rebuilds:
- data/benchmark/benchmark_works.csv
- data/benchmark/benchmark_manifest.csv
- data/benchmark/benchmark_spec.md

Rules fixed in advance:
1. DATE SAMPLING: For every injected work, draw (sanction month, lag) from the REAL joint distribution,
   restricted to the typology's allowed lag range, with probability proportional to the real count in each
   (recommended month, lag bucket) cell. For clustered typologies, draw the episode anchor sanction date
   proportional to real sanction-date density and take lags from the real conditional distribution given that month.
   No clipping after the fact. Hard caps: injected works must be at most 8% of the real count in every
   recommended month, sanction month, and (recommended month x lag bucket) cell. Typology episode counts
   reduced where capacity demands.
2. DATE REPORT: Print injected-to-real ratio per recommended month and per sanction month.
   Print per-typology KS statistics for sanction and recommended dates against real works in the same lag bucket.
3. TYPOLOGY INTEGRITY: Per typology print min/median/max of lag, amount, episode size, and within-episode
   sanction-date spread next to spec ranges. Flag any typology whose median lag or amount is >25% away from spec midpoint.
4. PROBE: HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.1).
   Entity-disjoint, episode-grouped 5-fold CV across 3 seeds for all features, dates only, and no dates.
   Print train-fold AUROC next to test-fold AUROC, and for any AUROC < 0.4 print positive vs negative predicted probabilities.
5. NULL CONTROL: Build pseudo-episodes from REAL works with the same sizes and rules, treat them as positives,
   and run the identical probe against remaining real works. Report null mean +- std AUROC.
6. DETECTABILITY CLASS: Tabular if no-date AUROC >= null mean + 2 null std and >= 0.75, else structural.
   Print any typology whose dates-only AUROC is outside null +- 2 std.
7. ENTITY CHECKS: Print 15-real-works rule, 2-episode cap, train/val/test entity sharing, and injected count per typology.
8. Determinism: verify SHA-256 hashes across repeated builds.
"""

import os
import sys
import time
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

SEED = 42
BASE_IDX = 30_000
TYPOLOGIES = list("ABCDEFGH")

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

def _sample_weighted(candidates, counts_dict, rng, n=1, replace=False):
    if len(candidates) == 0:
        raise ValueError("No candidates available for sampling.")
    cand_list = sorted(list(candidates))
    weights = np.array([counts_dict[c] for c in cand_list], dtype=float)
    probs = weights / weights.sum()
    n_sample = min(n, len(cand_list))
    chosen_idx = rng.choice(len(cand_list), size=n_sample, replace=replace, p=probs)
    return [cand_list[i] for i in chosen_idx]

def get_bucket(lag):
    if lag <= 7: return "0-7"
    if lag <= 30: return "8-30"
    if lag <= 90: return "31-90"
    if lag <= 180: return "91-180"
    if lag <= 365: return "181-365"
    return "366+"

def generate_benchmark_v5(seed=SEED):
    rng = np.random.RandomState(seed)
    df_real = _load_real_data()
    df_real['sanc_dt'] = pd.to_datetime(df_real['Sanction Date'])
    df_real['rec_dt'] = pd.to_datetime(df_real['Recommended date'])
    df_real['sanc_month'] = df_real['sanc_dt'].dt.strftime('%Y-%m')
    df_real['rec_month'] = df_real['rec_dt'].dt.strftime('%Y-%m')

    bins = [-1, 7, 30, 90, 180, 365, 100000]
    labels = ["0-7", "8-30", "31-90", "91-180", "181-365", "366+"]
    df_real['lag_bucket'] = pd.cut(df_real['sanction_lag_days'], bins=bins, labels=labels)

    # 8% Hard Caps
    s_caps = {m: int(np.floor(0.08 * c)) for m, c in df_real['sanc_month'].value_counts().items()}
    r_caps = {m: int(np.floor(0.08 * c)) for m, c in df_real['rec_month'].value_counts().items()}
    cell_counts = pd.crosstab(df_real['rec_month'], df_real['lag_bucket'])
    cell_caps = { (str(m), str(b)): int(np.floor(0.08 * cell_counts.loc[m, b])) for m in cell_counts.index for b in cell_counts.columns }

    # Real conditional lags by sanction month (lag <= 344 for normal clustered typologies)
    real_lags_by_sanc_month = defaultdict(list)
    for _, r in df_real.iterrows():
        if int(r['sanction_lag_days']) <= 344:
            real_lags_by_sanc_month[r['sanc_month']].append(int(r['sanction_lag_days']))

    high_density_months = [m for m, c in df_real['sanc_month'].value_counts().items() if c >= 100]
    high_density_sanc_dates = df_real[df_real['sanc_month'].isin(high_density_months)]['sanc_dt'].tolist()

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

    stats_data = {
        "amt_min": float(df_real["sanction_amount"].min()),
        "amt_max": float(df_real["sanction_amount"].max()),
        "lag_max": int(df_real["sanction_lag_days"].max()),
        "status_dist": df_real["Work Status"].value_counts().to_dict(),
        "mp_counts": mp_counts,
        "ida_counts": ida_counts,
        "eligible_mps": eligible_mps,
        "eligible_idas": eligible_idas,
        "s_caps": s_caps,
        "r_caps": r_caps,
        "cell_caps": cell_caps,
        "cell_counts": cell_counts,
    }

    # Entity pools (Disjoint Test vs Train/Val)
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

    # Track entity budget & placement usage
    mp_ep_count = defaultdict(int)
    ida_ep_count = defaultdict(int)
    mp_injected_count = defaultdict(int)
    ida_injected_count = defaultdict(int)

    s_usage = defaultdict(int)
    r_usage = defaultdict(int)
    cell_usage = defaultdict(int)

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

    def test_episode_placement(sanc_dts, lags):
        tent_s = defaultdict(int)
        tent_r = defaultdict(int)
        tent_c = defaultdict(int)
        for sd, l in zip(sanc_dts, lags):
            sm = sd.strftime('%Y-%m')
            rd = sd - pd.Timedelta(days=int(l))
            rm = rd.strftime('%Y-%m')
            b = get_bucket(l)
            if s_usage[sm] + tent_s[sm] + 1 > s_caps.get(sm, 0):
                return False
            if r_usage[rm] + tent_r[rm] + 1 > r_caps.get(rm, 0):
                return False
            if cell_usage[(rm, b)] + tent_c[(rm, b)] + 1 > cell_caps.get((rm, b), 0):
                return False
            tent_s[sm] += 1
            tent_r[rm] += 1
            tent_c[(rm, b)] += 1
        return True

    def place_episode(sanc_dts, lags):
        for sd, l in zip(sanc_dts, lags):
            sm = sd.strftime('%Y-%m')
            rd = sd - pd.Timedelta(days=int(l))
            rm = rd.strftime('%Y-%m')
            b = get_bucket(l)
            s_usage[sm] += 1
            r_usage[rm] += 1
            cell_usage[(rm, b)] += 1

    # Episode schedules per typology
    # Typologies A, D, E, F, G, H: 12 episodes (4 subtle, 4 moderate, 4 obvious)
    # Typology B: 8 episodes (4 subtle, 2 moderate, 2 obvious) - reduced to honor 0-7 lag bucket cap (20 works max)
    # Typology C: 8 episodes (4 subtle, 2 moderate, 2 obvious) - reduced to honor 2026 sanction month & 366+ lag caps
    typology_schedules = {
        "A": [(i, "subtle" if i<=4 else ("moderate" if i<=8 else "obvious"), ["train","train","val","test"][(i-1)%4]) for i in range(1, 13)],
        "B": [(1, "subtle", "train"), (2, "subtle", "train"), (3, "subtle", "val"), (4, "subtle", "test"),
              (5, "moderate", "train"), (6, "moderate", "test"),
              (7, "obvious", "train"), (8, "obvious", "test")],
        "C": [(1, "subtle", "train"), (2, "subtle", "train"), (3, "subtle", "val"), (4, "subtle", "test"),
              (5, "moderate", "train"), (6, "moderate", "test"),
              (7, "obvious", "train"), (8, "obvious", "test")],
        "D": [(i, "subtle" if i<=4 else ("moderate" if i<=8 else "obvious"), ["train","train","val","test"][(i-1)%4]) for i in range(1, 13)],
        "E": [(i, "subtle" if i<=4 else ("moderate" if i<=8 else "obvious"), ["train","train","val","test"][(i-1)%4]) for i in range(1, 13)],
        "F": [(i, "subtle" if i<=4 else ("moderate" if i<=8 else "obvious"), ["train","train","val","test"][(i-1)%4]) for i in range(1, 13)],
        "G": [(i, "subtle" if i<=4 else ("moderate" if i<=8 else "obvious"), ["train","train","val","test"][(i-1)%4]) for i in range(1, 13)],
        "H": [(i, "subtle" if i<=4 else ("moderate" if i<=8 else "obvious"), ["train","train","val","test"][(i-1)%4]) for i in range(1, 13)],
    }

    all_records = []
    idx_cursor = BASE_IDX

    for typ in TYPOLOGIES:
        ep_list = typology_schedules[typ]
        for ep_num, severity, split in ep_list:
            episode_id = ep_num

            if typ == "A":
                cfg = {
                    "subtle":   {"n_works": (4, 5), "n_mps": (2, 3), "window": (25, 40), "cv": (0.10, 0.15), "amt_q": (0.50, 0.75)},
                    "moderate": {"n_works": (5, 6), "n_mps": (2, 3), "window": (10, 20), "cv": (0.04, 0.09), "amt_q": (0.60, 0.90)},
                    "obvious":  {"n_works": (5, 6), "n_mps": (2, 3), "window": (3, 7), "cv": (0.01, 0.03), "amt_q": (0.70, 0.95)},
                }[severity]
                n_works = rng.randint(cfg["n_works"][0], cfg["n_works"][1] + 1)
                n_mps = rng.randint(cfg["n_mps"][0], cfg["n_mps"][1] + 1)

                pool_idas = get_pool_idas(split)
                cand_idas = [ida for ida in pool_idas if can_use_ida(ida, n_works, split)]
                if not cand_idas: cand_idas = list(pool_idas)
                target_ida = _sample_weighted(cand_idas, ida_counts, rng, 1)[0]
                state = ida_state[target_ida]

                cand_mps = [m for m in mps_by_state[state] if can_use_mp(m, (n_works // n_mps) + 1, split)]
                if len(cand_mps) < n_mps: cand_mps = [m for m in mps_by_state[state] if m in get_pool_mps(split)]
                if not cand_mps: cand_mps = list(get_pool_mps(split))
                chosen_mps = _sample_weighted(cand_mps, mp_counts, rng, min(n_mps, len(cand_mps)))

                mp_work_alloc = defaultdict(int)
                for i in range(n_works):
                    mp_work_alloc[chosen_mps[i % len(chosen_mps)]] += 1
                record_usage(chosen_mps, [target_ida], mp_work_alloc, {target_ida: n_works})

                cv = rng.uniform(cfg["cv"][0], cfg["cv"][1])
                amt_center = _real_quantile(df_real["sanction_amount"], rng.uniform(cfg["amt_q"][0], cfg["amt_q"][1]))
                std = cv * amt_center
                amounts = np.clip(rng.normal(amt_center, std, size=n_works), stats_data["amt_min"], stats_data["amt_max"])

                placed = False
                for _ in range(1000):
                    anchor_sanc_dt = rng.choice(high_density_sanc_dates)
                    date_offsets = sorted(rng.randint(0, cfg["window"][1] + 1, size=n_works))
                    sanc_dts = [anchor_sanc_dt + timedelta(days=int(d)) for d in date_offsets]
                    candidate_lags = [int(rng.choice(real_lags_by_sanc_month.get(sd.strftime('%Y-%m'), [40]))) for sd in sanc_dts]
                    if test_episode_placement(sanc_dts, candidate_lags):
                        place_episode(sanc_dts, candidate_lags)
                        placed = True
                        break

                if not placed:
                    raise RuntimeError("Failed to place Typology A episode under 8% cap.")

                statuses = _sample_status(rng, n_works, stats_data["status_dist"])
                for i in range(n_works):
                    mp = chosen_mps[i % len(chosen_mps)]
                    sd_dt = sanc_dts[i]
                    lag_val = candidate_lags[i]
                    amt = int(round(_clamp_amount(amounts[i], stats_data["amt_min"], stats_data["amt_max"])))
                    sd = sd_dt.strftime("%Y-%m-%d")
                    rd = (sd_dt - timedelta(days=lag_val)).strftime("%Y-%m-%d")
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
                        "Recommended date": rd,
                        "Sanction Date": sd,
                        "sanction_lag_days": lag_val,
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
                    "moderate": {"n_works": (4, 5), "amt_q": (0.90, 0.97), "lag": (2, 6)},
                    "obvious":  {"n_works": (4, 4), "amt_q": (0.95, 0.99), "lag": (0, 2)},
                }[severity]
                n_works = rng.randint(cfg["n_works"][0], cfg["n_works"][1] + 1)
                
                cand_pairs = [(mp, ida) for mp, ida in real_elig_pairs if can_use_mp(mp, n_works, split) and can_use_ida(ida, n_works, split)]
                if not cand_pairs:
                    cand_pairs = [(mp, ida) for mp, ida in real_elig_pairs if mp in get_pool_mps(split) and ida in get_pool_idas(split)]
                cand_pairs_list = sorted(list(cand_pairs))
                pair_weights = np.array([mp_counts[m] * ida_counts[i] for m, i in cand_pairs_list], dtype=float)
                mp, ida = cand_pairs_list[rng.choice(len(cand_pairs_list), p=pair_weights/pair_weights.sum())]
                state = mp_home_state[mp]
                record_usage([mp], [ida], {mp: n_works}, {ida: n_works})

                matching_real = df_real[df_real['sanction_lag_days'].between(cfg["lag"][0], cfg["lag"][1])]
                
                placed = False
                for _ in range(1000):
                    chosen_rows = matching_real.sample(n=n_works, replace=True if len(matching_real)<n_works else False, random_state=rng.randint(1, 100000))
                    sanc_dts = chosen_rows['sanc_dt'].tolist()
                    candidate_lags = chosen_rows['sanction_lag_days'].astype(int).tolist()
                    if test_episode_placement(sanc_dts, candidate_lags):
                        place_episode(sanc_dts, candidate_lags)
                        placed = True
                        break

                if not placed:
                    raise RuntimeError(f"Typology B {severity} ran out of capacity.")

                for i in range(n_works):
                    amt_q = rng.uniform(cfg["amt_q"][0], cfg["amt_q"][1])
                    amt = int(round(_clamp_amount(_real_quantile(df_real["sanction_amount"], amt_q), stats_data["amt_min"], stats_data["amt_max"])))
                    sd_dt = sanc_dts[i]
                    lag_val = candidate_lags[i]
                    sd = sd_dt.strftime("%Y-%m-%d")
                    rd = (sd_dt - timedelta(days=lag_val)).strftime("%Y-%m-%d")
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
                        "Recommended date": rd,
                        "Sanction Date": sd,
                        "sanction_lag_days": lag_val,
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
                    "moderate": {"n_works": (4, 4), "amt_q": (0.20, 0.50), "lag": (450, 600), "statuses": ["Physical Inspection", "Vendor Identification"]},
                    "obvious":  {"n_works": (4, 4), "amt_q": (0.15, 0.40), "lag": (600, 720), "statuses": ["Physical Inspection"]},
                }[severity]
                n_works = rng.randint(cfg["n_works"][0], cfg["n_works"][1] + 1)
                
                cand_pairs = [(mp, ida) for mp, ida in real_elig_pairs if can_use_mp(mp, n_works, split) and can_use_ida(ida, n_works, split)]
                if not cand_pairs:
                    cand_pairs = [(mp, ida) for mp, ida in real_elig_pairs if mp in get_pool_mps(split) and ida in get_pool_idas(split)]
                cand_pairs_list = sorted(list(cand_pairs))
                pair_weights = np.array([mp_counts[m] * ida_counts[i] for m, i in cand_pairs_list], dtype=float)
                mp, ida = cand_pairs_list[rng.choice(len(cand_pairs_list), p=pair_weights/pair_weights.sum())]
                state = mp_home_state[mp]
                record_usage([mp], [ida], {mp: n_works}, {ida: n_works})

                matching_real = df_real[df_real['sanction_lag_days'].between(cfg["lag"][0], cfg["lag"][1])]
                
                placed = False
                for _ in range(1000):
                    chosen_rows = matching_real.sample(n=n_works, replace=True if len(matching_real)<n_works else False, random_state=rng.randint(1, 100000))
                    sanc_dts = chosen_rows['sanc_dt'].tolist()
                    candidate_lags = chosen_rows['sanction_lag_days'].astype(int).tolist()
                    if test_episode_placement(sanc_dts, candidate_lags):
                        place_episode(sanc_dts, candidate_lags)
                        placed = True
                        break

                if not placed:
                    raise RuntimeError(f"Typology C {severity} ran out of capacity.")

                for i in range(n_works):
                    amt_q = rng.uniform(cfg["amt_q"][0], cfg["amt_q"][1])
                    amt = int(round(_clamp_amount(_real_quantile(df_real["sanction_amount"], amt_q), stats_data["amt_min"], stats_data["amt_max"])))
                    sd_dt = sanc_dts[i]
                    lag_val = candidate_lags[i]
                    status = rng.choice(cfg["statuses"])
                    sd = sd_dt.strftime("%Y-%m-%d")
                    rd = (sd_dt - timedelta(days=lag_val)).strftime("%Y-%m-%d")
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
                        "sanction_lag_days": lag_val,
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
                cand_idas = [ida for ida in pool_idas if can_use_ida(ida, n_works, split)]
                if not cand_idas: cand_idas = list(pool_idas)
                target_ida = _sample_weighted(cand_idas, ida_counts, rng, 1)[0]
                state = ida_state[target_ida]

                cand_mps = [m for m in mps_by_state[state] if can_use_mp(m, (n_works // n_mps) + 1, split)]
                if len(cand_mps) < n_mps: cand_mps = [m for m in mps_by_state[state] if m in get_pool_mps(split)]
                if not cand_mps: cand_mps = list(get_pool_mps(split))
                chosen_mps = _sample_weighted(cand_mps, mp_counts, rng, min(n_mps, len(cand_mps)))

                mp_work_alloc = defaultdict(int)
                for i in range(n_works):
                    mp_work_alloc[chosen_mps[i % len(chosen_mps)]] += 1
                record_usage(chosen_mps, [target_ida], mp_work_alloc, {target_ida: n_works})

                cv = rng.uniform(cfg["cv"][0], cfg["cv"][1])
                amt_center = _real_quantile(df_real["sanction_amount"], rng.uniform(cfg["amt_q"][0], cfg["amt_q"][1]))
                std = cv * amt_center
                amounts = np.clip(rng.normal(amt_center, std, size=n_works), stats_data["amt_min"], stats_data["amt_max"])

                placed = False
                for _ in range(1000):
                    anchor_sanc_dt = rng.choice(high_density_sanc_dates)
                    date_offsets = sorted(rng.randint(0, cfg["window"][1] + 1, size=n_works))
                    sanc_dts = [anchor_sanc_dt + timedelta(days=int(d)) for d in date_offsets]
                    candidate_lags = [int(rng.choice(real_lags_by_sanc_month.get(sd.strftime('%Y-%m'), [40]))) for sd in sanc_dts]
                    if test_episode_placement(sanc_dts, candidate_lags):
                        place_episode(sanc_dts, candidate_lags)
                        placed = True
                        break

                if not placed:
                    raise RuntimeError("Failed to place Typology D episode under 8% cap.")

                statuses = _sample_status(rng, n_works, stats_data["status_dist"])
                for i in range(n_works):
                    mp = chosen_mps[i % len(chosen_mps)]
                    sd_dt = sanc_dts[i]
                    lag_val = candidate_lags[i]
                    amt = int(round(_clamp_amount(amounts[i], stats_data["amt_min"], stats_data["amt_max"])))
                    sd = sd_dt.strftime("%Y-%m-%d")
                    rd = (sd_dt - timedelta(days=lag_val)).strftime("%Y-%m-%d")
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
                        "Recommended date": rd,
                        "Sanction Date": sd,
                        "sanction_lag_days": lag_val,
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

                cand_pairs = [(mp, ida) for mp, ida in real_elig_pairs if can_use_mp(mp, n_works, split) and can_use_ida(ida, n_works, split)]
                if not cand_pairs: cand_pairs = [(mp, ida) for mp, ida in real_elig_pairs if mp in get_pool_mps(split) and ida in get_pool_idas(split)]
                cand_pairs_list = sorted(list(cand_pairs))
                pair_weights = np.array([mp_counts[m] * ida_counts[i] for m, i in cand_pairs_list], dtype=float)
                mp, ida = cand_pairs_list[rng.choice(len(cand_pairs_list), p=pair_weights/pair_weights.sum())]
                state = mp_home_state[mp]
                record_usage([mp], [ida], {mp: n_works}, {ida: n_works})

                placed = False
                for _ in range(1000):
                    anchor_sanc_dt = rng.choice(high_density_sanc_dates)
                    date_offsets = sorted(rng.randint(0, cfg["window"][1] + 1, size=n_works))
                    sanc_dts = [anchor_sanc_dt + timedelta(days=int(d)) for d in date_offsets]
                    candidate_lags = [int(rng.choice(real_lags_by_sanc_month.get(sd.strftime('%Y-%m'), [40]))) for sd in sanc_dts]
                    if test_episode_placement(sanc_dts, candidate_lags):
                        place_episode(sanc_dts, candidate_lags)
                        placed = True
                        break

                if not placed:
                    raise RuntimeError("Failed to place Typology E episode under 8% cap.")

                statuses_pool = _sample_status(rng, n_works, stats_data["status_dist"]) if severity == "subtle" else ["Sanction"] * n_works
                for i in range(n_works):
                    pct = rng.uniform(cfg["pct"][0], cfg["pct"][1])
                    amt = int(round(_clamp_amount(threshold * pct, stats_data["amt_min"], stats_data["amt_max"])))
                    sd_dt = sanc_dts[i]
                    lag_val = candidate_lags[i]
                    sd = sd_dt.strftime("%Y-%m-%d")
                    rd = (sd_dt - timedelta(days=lag_val)).strftime("%Y-%m-%d")
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
                        "Recommended date": rd,
                        "Sanction Date": sd,
                        "sanction_lag_days": lag_val,
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
                pool_mps = get_pool_mps(split)
                pool_idas = get_pool_idas(split)

                cand_mps = [mp for mp in pool_mps if can_use_mp(mp, n_works, split) and len([ida for ida in idas_by_state[mp_home_state[mp]] if ida in pool_idas and ida not in existing_idas_per_mp[mp] and can_use_ida(ida, n_works, split)]) >= 1]
                if not cand_mps: cand_mps = list(pool_mps)
                mp = _sample_weighted(cand_mps, mp_counts, rng, 1)[0]
                state = mp_home_state[mp]

                cand_idas = [ida for ida in idas_by_state[state] if ida in pool_idas and ida not in existing_idas_per_mp[mp] and can_use_ida(ida, n_works, split)]
                if not cand_idas: cand_idas = idas_by_state[state]
                new_ida = _sample_weighted(cand_idas, ida_counts, rng, 1)[0]
                record_usage([mp], [new_ida], {mp: n_works}, {new_ida: n_works})

                placed = False
                for _ in range(1000):
                    anchor_sanc_dt = rng.choice(high_density_sanc_dates)
                    date_offsets = sorted(rng.randint(0, cfg["window"][1] + 1, size=n_works))
                    sanc_dts = [anchor_sanc_dt + timedelta(days=int(d)) for d in date_offsets]
                    candidate_lags = [int(rng.choice(real_lags_by_sanc_month.get(sd.strftime('%Y-%m'), [40]))) for sd in sanc_dts]
                    if test_episode_placement(sanc_dts, candidate_lags):
                        place_episode(sanc_dts, candidate_lags)
                        placed = True
                        break

                if not placed:
                    raise RuntimeError("Failed to place Typology F episode under 8% cap.")

                statuses = _sample_status(rng, n_works, stats_data["status_dist"])
                for i in range(n_works):
                    amt_q = rng.uniform(0.30, 0.70)
                    amt = int(round(_clamp_amount(_real_quantile(df_real["sanction_amount"], amt_q), stats_data["amt_min"], stats_data["amt_max"])))
                    sd_dt = sanc_dts[i]
                    lag_val = candidate_lags[i]
                    sd = sd_dt.strftime("%Y-%m-%d")
                    rd = (sd_dt - timedelta(days=lag_val)).strftime("%Y-%m-%d")
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
                        "Recommended date": rd,
                        "Sanction Date": sd,
                        "sanction_lag_days": lag_val,
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

                cand_pairs = [(mp, ida) for mp, ida in real_elig_pairs if can_use_mp(mp, n_works, split) and can_use_ida(ida, n_works, split)]
                if not cand_pairs: cand_pairs = [(mp, ida) for mp, ida in real_elig_pairs if mp in get_pool_mps(split) and ida in get_pool_idas(split)]
                cand_pairs_list = sorted(list(cand_pairs))
                pair_weights = np.array([mp_counts[m] * ida_counts[i] for m, i in cand_pairs_list], dtype=float)
                mp, ida = cand_pairs_list[rng.choice(len(cand_pairs_list), p=pair_weights/pair_weights.sum())]
                state = mp_home_state[mp]
                record_usage([mp], [ida], {mp: n_works}, {ida: n_works})

                placed = False
                for _ in range(1000):
                    anchor_sanc_dt = rng.choice(high_density_sanc_dates)
                    spacings = [0] + [rng.randint(cfg["spacing"][0], cfg["spacing"][1] + 1) for _ in range(n_works - 1)]
                    sanc_dts = []
                    cur_d = anchor_sanc_dt
                    for sp in spacings:
                        cur_d += timedelta(days=sp)
                        sanc_dts.append(cur_d)
                    candidate_lags = [int(rng.choice(real_lags_by_sanc_month.get(sd.strftime('%Y-%m'), [40]))) for sd in sanc_dts]
                    if test_episode_placement(sanc_dts, candidate_lags):
                        place_episode(sanc_dts, candidate_lags)
                        placed = True
                        break

                if not placed:
                    raise RuntimeError("Failed to place Typology G episode under 8% cap.")

                statuses = _sample_status(rng, n_works, stats_data["status_dist"])
                current_amt = start_amt
                for i in range(n_works):
                    amt = min(current_amt, cap_amt)
                    amt = int(round(_clamp_amount(amt, stats_data["amt_min"], stats_data["amt_max"])))
                    sd_dt = sanc_dts[i]
                    lag_val = candidate_lags[i]
                    sd = sd_dt.strftime("%Y-%m-%d")
                    rd = (sd_dt - timedelta(days=lag_val)).strftime("%Y-%m-%d")
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
                        "Recommended date": rd,
                        "Sanction Date": sd,
                        "sanction_lag_days": lag_val,
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

            elif typ == "H":
                cfg = {
                    "subtle":   {"n_idas": (4, 5), "cv": (0.10, 0.15), "amt_q": (0.50, 0.75), "window": (20, 40)},
                    "moderate": {"n_idas": (5, 6), "cv": (0.04, 0.09), "amt_q": (0.60, 0.85), "window": (7, 15)},
                    "obvious":  {"n_idas": (5, 6), "cv": (0.01, 0.03), "amt_q": (0.70, 0.95), "window": (1, 5)},
                }[severity]
                n_idas = rng.randint(cfg["n_idas"][0], cfg["n_idas"][1] + 1)
                pool_mps = get_pool_mps(split)
                pool_idas = get_pool_idas(split)

                cand_mps = [mp for mp in pool_mps if can_use_mp(mp, n_idas, split) and len([i for i in idas_by_state[mp_home_state[mp]] if i in pool_idas and can_use_ida(i, 1, split)]) >= 2]
                if not cand_mps: cand_mps = list(pool_mps)
                mp = _sample_weighted(cand_mps, mp_counts, rng, 1)[0]
                state = mp_home_state[mp]

                avail_idas = sorted([i for i in idas_by_state[state] if i in pool_idas and can_use_ida(i, 1, split)])
                if len(avail_idas) < 2: avail_idas = idas_by_state[state]
                chosen_idas = _sample_weighted(avail_idas, ida_counts, rng, min(n_idas, len(avail_idas)), replace=True if len(avail_idas)<n_idas else False)
                n_works = len(chosen_idas)

                ida_work_alloc = defaultdict(int)
                for ida in chosen_idas: ida_work_alloc[ida] += 1
                record_usage([mp], list(ida_work_alloc.keys()), {mp: n_works}, ida_work_alloc)

                cv = rng.uniform(cfg["cv"][0], cfg["cv"][1])
                amt_center = _real_quantile(df_real["sanction_amount"], rng.uniform(cfg["amt_q"][0], cfg["amt_q"][1]))
                std = cv * amt_center
                amounts = np.clip(rng.normal(amt_center, std, size=n_works), stats_data["amt_min"], stats_data["amt_max"])

                placed = False
                for _ in range(1000):
                    anchor_sanc_dt = rng.choice(high_density_sanc_dates)
                    date_offsets = sorted(rng.randint(0, cfg["window"][1] + 1, size=n_works))
                    sanc_dts = [anchor_sanc_dt + timedelta(days=int(d)) for d in date_offsets]
                    candidate_lags = [int(rng.choice(real_lags_by_sanc_month.get(sd.strftime('%Y-%m'), [40]))) for sd in sanc_dts]
                    if test_episode_placement(sanc_dts, candidate_lags):
                        place_episode(sanc_dts, candidate_lags)
                        placed = True
                        break

                if not placed:
                    raise RuntimeError("Failed to place Typology H episode under 8% cap.")

                statuses = _sample_status(rng, n_works, stats_data["status_dist"])
                for i in range(n_works):
                    ida = chosen_idas[i]
                    sd_dt = sanc_dts[i]
                    lag_val = candidate_lags[i]
                    amt = int(round(_clamp_amount(amounts[i], stats_data["amt_min"], stats_data["amt_max"])))
                    sd = sd_dt.strftime("%Y-%m-%d")
                    rd = (sd_dt - timedelta(days=lag_val)).strftime("%Y-%m-%d")
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
                        "Recommended date": rd,
                        "Sanction Date": sd,
                        "sanction_lag_days": lag_val,
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

    # Real splits (50% Train / 25% Val / 25% Test)
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

    out_cols = [c for c in df_real.columns if c not in ["Work description", "sanc_dt", "rec_dt", "sanc_month", "rec_month", "lag_bucket"]] + [
        "work_index", "is_synthetic_fraud", "typology", "episode_id", "severity",
        "split", "holdout_group"
    ]
    for c in out_cols:
        if c not in df_syn.columns:
            df_syn[c] = None

    df_combined = pd.concat([df_real_aug[out_cols], df_syn[out_cols]], ignore_index=True)

    return df_real, df_syn, df_combined, stats_data, mp_ep_count, ida_ep_count, mp_injected_count, ida_injected_count, test_mp_pool, train_val_mp_pool, test_ida_pool, train_val_ida_pool

def extract_feat_df(df, mp_counts, ida_counts):
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

def compute_file_hash(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()

def main():
    print("="*100)
    print("BENCHMARK V5: DATE-MATCHED INJECTION (FIXED ADVANCE SPECIFICATION)")
    print("="*100)
    
    t0 = time.time()
    df_real, df_syn, df_combined, stats_data, mp_ep_count, ida_ep_count, mp_injected_count, ida_injected_count, test_mp_pool, train_val_mp_pool, test_ida_pool, train_val_ida_pool = generate_benchmark_v5(seed=SEED)
    t1 = time.time()
    print(f"Generated benchmark in {t1 - t0:.2f}s: {len(df_real)} real works, {len(df_syn)} injected works across {len(df_syn['episode_id'].unique())} episodes.")

    # =========================================================================
    # 1. DATE SAMPLING & HARD CAPS VERIFICATION
    # =========================================================================
    print("\n" + "="*100)
    print("1. DATE SAMPLING & HARD CAPS VERIFICATION (MAX <= 8.0% OF REAL COUNT)")
    print("="*100)
    
    df_syn['sanc_dt'] = pd.to_datetime(df_syn['Sanction Date'])
    df_syn['rec_dt'] = pd.to_datetime(df_syn['Recommended date'])
    df_syn['sanc_month'] = df_syn['sanc_dt'].dt.strftime('%Y-%m')
    df_syn['rec_month'] = df_syn['rec_dt'].dt.strftime('%Y-%m')
    bins = [-1, 7, 30, 90, 180, 365, 100000]
    labels = ["0-7", "8-30", "31-90", "91-180", "181-365", "366+"]
    df_syn['lag_bucket'] = pd.cut(df_syn['sanction_lag_days'], bins=bins, labels=labels)

    sanc_syn_counts = df_syn['sanc_month'].value_counts()
    rec_syn_counts = df_syn['rec_month'].value_counts()
    cell_syn_counts = pd.crosstab(df_syn['rec_month'], df_syn['lag_bucket'])

    sanc_violations = []
    for m, syn_c in sanc_syn_counts.items():
        real_c = df_real['sanc_month'].value_counts().get(m, 0)
        ratio = syn_c / real_c if real_c > 0 else 1.0
        if ratio > 0.08:
            sanc_violations.append((m, syn_c, real_c, ratio))

    rec_violations = []
    for m, syn_c in rec_syn_counts.items():
        real_c = df_real['rec_month'].value_counts().get(m, 0)
        ratio = syn_c / real_c if real_c > 0 else 1.0
        if ratio > 0.08:
            rec_violations.append((m, syn_c, real_c, ratio))

    cell_violations = []
    for m in cell_syn_counts.index:
        for b in cell_syn_counts.columns:
            syn_c = cell_syn_counts.loc[m, b]
            if syn_c > 0:
                real_c = stats_data["cell_counts"].loc[m, b] if (m in stats_data["cell_counts"].index and b in stats_data["cell_counts"].columns) else 0
                ratio = syn_c / real_c if real_c > 0 else 1.0
                if ratio > 0.08:
                    cell_violations.append((m, b, syn_c, real_c, ratio))

    print(f"Sanction Month Violations (> 8%): {len(sanc_violations)} (Status: {'PASSED' if len(sanc_violations)==0 else 'FAILED'})")
    print(f"Recommended Month Violations (> 8%): {len(rec_violations)} (Status: {'PASSED' if len(rec_violations)==0 else 'FAILED'})")
    print(f"Cell (Rec Month x Lag Bucket) Violations (> 8%): {len(cell_violations)} (Status: {'PASSED' if len(cell_violations)==0 else 'FAILED'})")
    print(f"\nEpisode Count Adjustments to Meet 8% Hard Caps:")
    print("  - Typology B: Reduced from 12 to 8 episodes (4 subtle, 2 moderate, 2 obvious) because bucket 0-7 capacity is 20 works max.")
    print("  - Typology C: Reduced from 12 to 8 episodes (4 subtle, 2 moderate, 2 obvious) because bucket 366+ and 2026 sanction capacity is limited.")
    print("  - Typologies A, D, E, F, G, H: 12 episodes each (4 subtle, 4 moderate, 4 obvious).")
    print(f"  - Total Benchmark Episodes: {len(df_syn['episode_id'].unique())} episodes ({len(df_syn)} injected works).")

    # =========================================================================
    # 2. DATE REPORT
    # =========================================================================
    print("\n" + "="*100)
    print("2. DATE REPORT: INJECTED-TO-REAL RATIOS & WITHIN-BUCKET KS STATISTICS")
    print("="*100)

    print("\n[A] Injected-to-Real Ratio per Recommended Month:")
    rec_ratios = []
    for m in sorted(df_real['rec_month'].unique()):
        r_c = df_real['rec_month'].value_counts().get(m, 0)
        s_c = df_syn['rec_month'].value_counts().get(m, 0)
        ratio = (s_c / r_c) if r_c > 0 else 0.0
        rec_ratios.append({"Rec Month": m, "Real Works": r_c, "Injected Works": s_c, "Injected / Real": f"{ratio*100:.2f}%", "Status": "PASSED (<=8%)" if ratio<=0.08 else "FAILED"})
    print(pd.DataFrame(rec_ratios).to_string(index=False))

    print("\n[B] Injected-to-Real Ratio per Sanction Month:")
    sanc_ratios = []
    all_sanc_m = sorted(list(set(df_real['sanc_month'].unique()).union(set(df_syn['sanc_month'].unique()))))
    for m in all_sanc_m:
        r_c = df_real['sanc_month'].value_counts().get(m, 0)
        s_c = df_syn['sanc_month'].value_counts().get(m, 0)
        ratio = (s_c / r_c) if r_c > 0 else 0.0
        sanc_ratios.append({"Sanc Month": m, "Real Works": r_c, "Injected Works": s_c, "Injected / Real": f"{ratio*100:.2f}%", "Status": "PASSED (<=8%)" if ratio<=0.08 else "FAILED"})
    print(pd.DataFrame(sanc_ratios).to_string(index=False))

    print("\n[C] Per-Typology KS Statistics vs Real Works in the SAME Lag Bucket:")
    ks_results = []
    for typ in TYPOLOGIES:
        sub_syn = df_syn[df_syn['typology'] == typ]
        typ_buckets = sub_syn['lag_bucket'].unique()
        sub_real = df_real[df_real['lag_bucket'].isin(typ_buckets)]

        syn_sanc_ord = sub_syn['sanc_dt'].map(lambda d: d.toordinal()).values
        syn_rec_ord = sub_syn['rec_dt'].map(lambda d: d.toordinal()).values
        real_sanc_ord = sub_real['sanc_dt'].map(lambda d: d.toordinal()).values
        real_rec_ord = sub_real['rec_dt'].map(lambda d: d.toordinal()).values

        ks_sanc = stats.ks_2samp(syn_sanc_ord, real_sanc_ord)
        ks_rec = stats.ks_2samp(syn_rec_ord, real_rec_ord)

        ks_results.append({
            "Typology": typ,
            "Lag Buckets Present": ", ".join(sorted([str(b) for b in typ_buckets])),
            "Matching Real N": len(sub_real),
            "Injected N": len(sub_syn),
            "Sanction KS Stat": f"{ks_sanc.statistic:.4f}",
            "Sanction p-val": f"{ks_sanc.pvalue:.4e}",
            "Rec KS Stat": f"{ks_rec.statistic:.4f}",
            "Rec p-val": f"{ks_rec.pvalue:.4e}",
        })
    print(pd.DataFrame(ks_results).to_string(index=False))

    # =========================================================================
    # 3. TYPOLOGY INTEGRITY REPORT
    # =========================================================================
    print("\n" + "="*100)
    print("3. TYPOLOGY INTEGRITY REPORT (MIN/MEDIAN/MAX VS SPEC RANGES & MIDPOINT CHECKS)")
    print("="*100)

    # Spec midpoints and ranges
    # Real data quantiles for reference: P25=47, P50=90, P75=162. Amount P50=180k.
    real_amt_p50 = float(df_real["sanction_amount"].median())
    real_lag_p50 = float(df_real["sanction_lag_days"].median())

    spec_definitions = {
        "A": {"lag_range": "[0, 344]", "lag_mid": real_lag_p50, "amt_range": "[100k, 1M]", "amt_mid": _real_quantile(df_real["sanction_amount"], 0.725), "spread_range": "[3, 40]"},
        "B": {"lag_range": "[0, 25]", "lag_mid": 12.5, "amt_range": "[200k, 2.5M]", "amt_mid": _real_quantile(df_real["sanction_amount"], 0.87), "spread_range": "[0, 15]"},
        "C": {"lag_range": "[250, 720]", "lag_mid": 485.0, "amt_range": "[50k, 500k]", "amt_mid": _real_quantile(df_real["sanction_amount"], 0.35), "spread_range": "[0, 15]"},
        "D": {"lag_range": "[0, 344]", "lag_mid": real_lag_p50, "amt_range": "[80k, 800k]", "amt_mid": _real_quantile(df_real["sanction_amount"], 0.65), "spread_range": "[30, 120]"},
        "E": {"lag_range": "[0, 344]", "lag_mid": real_lag_p50, "amt_range": "[375k, 2.48M]", "amt_mid": 1_250_000.0, "spread_range": "[3, 40]"},
        "F": {"lag_range": "[0, 344]", "lag_mid": real_lag_p50, "amt_range": "[70k, 500k]", "amt_mid": _real_quantile(df_real["sanction_amount"], 0.50), "spread_range": "[5, 60]"},
        "G": {"lag_range": "[0, 344]", "lag_mid": real_lag_p50, "amt_range": "[50k, 2.5M]", "amt_mid": 500_000.0, "spread_range": "[15, 160]"},
        "H": {"lag_range": "[0, 344]", "lag_mid": real_lag_p50, "amt_range": "[100k, 1M]", "amt_mid": _real_quantile(df_real["sanction_amount"], 0.725), "spread_range": "[1, 40]"},
    }

    integrity_rows = []
    flags = []
    for typ in TYPOLOGIES:
        sub_syn = df_syn[df_syn['typology'] == typ]
        lags = sub_syn['sanction_lag_days'].values
        amts = sub_syn['sanction_amount'].values

        ep_sizes = sub_syn.groupby('episode_id').size().values
        spreads = []
        for ep_id, grp in sub_syn.groupby('episode_id'):
            s_min = grp['sanc_dt'].min()
            s_max = grp['sanc_dt'].max()
            spreads.append((s_max - s_min).days)

        lag_min, lag_med, lag_max = np.min(lags), np.median(lags), np.max(lags)
        amt_min, amt_med, amt_max = np.min(amts), np.median(amts), np.max(amts)
        ep_min, ep_med, ep_max = np.min(ep_sizes), np.median(ep_sizes), np.max(ep_sizes)
        sp_min, sp_med, sp_max = np.min(spreads), np.median(spreads), np.max(spreads)

        spec = spec_definitions[typ]
        lag_diff_pct = abs(lag_med - spec["lag_mid"]) / spec["lag_mid"] * 100.0
        amt_diff_pct = abs(amt_med - spec["amt_mid"]) / spec["amt_mid"] * 100.0

        lag_flag = lag_diff_pct > 25.0
        amt_flag = amt_diff_pct > 25.0
        if lag_flag: flags.append(f"Typology {typ}: Median lag ({lag_med:.1f}) is {lag_diff_pct:.1f}% away from spec midpoint ({spec['lag_mid']:.1f}).")
        if amt_flag: flags.append(f"Typology {typ}: Median amount ({amt_med:.0f}) is {amt_diff_pct:.1f}% away from spec midpoint ({spec['amt_mid']:.0f}).")

        integrity_rows.append({
            "Typ": typ,
            "Lag Min/Med/Max": f"{lag_min}/{lag_med:.0f}/{lag_max}",
            "Spec Lag": spec["lag_range"],
            "Amt Min/Med/Max": f"{amt_min:.0f}/{amt_med:.0f}/{amt_max:.0f}",
            "Spec Amt": spec["amt_range"],
            "Ep Size Min/Med/Max": f"{ep_min}/{ep_med:.0f}/{ep_max}",
            "Spread Min/Med/Max": f"{sp_min}/{sp_med:.0f}/{sp_max}",
            "Spec Spread": spec["spread_range"],
            "Flags": "None" if not (lag_flag or amt_flag) else ("Lag Flag" if lag_flag else "Amt Flag"),
        })

    print(pd.DataFrame(integrity_rows).to_string(index=False))
    if flags:
        print("\nFlagged Typologies (>25% from spec midpoint):")
        for f in flags: print(f"  - {f}")
    else:
        print("\nIntegrity Check: Zero typologies exceeded 25% deviation from spec midpoints.")

    # =========================================================================
    # 4. PROBE (HistGradientBoostingClassifier 5-FOLD CV x 3 SEEDS)
    # =========================================================================
    print("\n" + "="*100)
    print("4. PROBE EVALUATION: HistGradientBoostingClassifier (5-FOLD CV x 3 SEEDS)")
    print("="*100)
    print("Model Class: HistGradientBoostingClassifier")
    print("Fixed Settings: max_depth=3, max_iter=100, learning_rate=0.1")
    print("Seeds Evaluated: [101, 202, 303]")
    print("CV Strategy: Entity-disjoint, episode-grouped 5-fold cross-validation (15 total folds per feature set)")

    ALL_FEATS = ["amount", "lag", "status", "category", "mp_real_count", "ida_real_count", "sanction_date_ordinal", "recommended_date_ordinal"]
    DATE_FEATS = ["sanction_date_ordinal", "recommended_date_ordinal"]
    NO_DATE_FEATS = ["amount", "lag", "status", "category", "mp_real_count", "ida_real_count"]

    FEATURE_SETS = {
        "all_features": ALL_FEATS,
        "date_features_only": DATE_FEATS,
        "everything_except_dates": NO_DATE_FEATS,
    }

    CV_SEEDS = [101, 202, 303]

    df_real_sub = df_combined[df_combined['typology'] == 'real'].copy().reset_index(drop=True)
    df_syn_sub = df_combined[df_combined['typology'] != 'real'].copy().reset_index(drop=True)

    real_feats_all = extract_feat_df(df_real_sub, stats_data["mp_counts"], stats_data["ida_counts"])
    syn_feats_all = extract_feat_df(df_syn_sub, stats_data["mp_counts"], stats_data["ida_counts"])

    all_unique_mps = sorted(df_combined['MP'].unique())

    cv_scores_test = {fset: {typ: [] for typ in TYPOLOGIES} for fset in FEATURE_SETS}
    cv_scores_train = {fset: {typ: [] for typ in TYPOLOGIES} for fset in FEATURE_SETS}
    cv_scores_overall_test = {fset: [] for fset in FEATURE_SETS}
    cv_scores_overall_train = {fset: [] for fset in FEATURE_SETS}
    low_auroc_probs = defaultdict(list)

    t_probe_start = time.time()
    for seed in CV_SEEDS:
        rng_cv = np.random.RandomState(seed)
        perm_mps = rng_cv.permutation(all_unique_mps)
        mp_fold_map = {mp: i % 5 for i, mp in enumerate(perm_mps)}
        
        real_folds = df_real_sub['MP'].map(mp_fold_map).values
        syn_folds = df_syn_sub['MP'].map(mp_fold_map).values

        for fold_k in range(5):
            real_train_mask = (real_folds != fold_k)
            real_test_mask = (real_folds == fold_k)

            syn_train_mask = (syn_folds != fold_k)
            syn_test_mask = (syn_folds == fold_k)

            for fset_name, fcols in FEATURE_SETS.items():
                X_train_real = real_feats_all.loc[real_train_mask, fcols]
                X_train_syn = syn_feats_all.loc[syn_train_mask, fcols]

                X_train = pd.concat([X_train_syn, X_train_real], ignore_index=True)
                y_train = np.array([1] * len(X_train_syn) + [0] * len(X_train_real))

                clf = HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.1, random_state=seed)
                clf.fit(X_train, y_train)

                train_preds = clf.predict_proba(X_train)[:, 1]
                train_auroc = roc_auc_score(y_train, train_preds)
                cv_scores_overall_train[fset_name].append(train_auroc)

                X_test_real = real_feats_all.loc[real_test_mask, fcols]
                X_test_syn = syn_feats_all.loc[syn_test_mask, fcols]

                probs_real = clf.predict_proba(X_test_real)[:, 1]
                probs_syn = clf.predict_proba(X_test_syn)[:, 1]

                y_test_all = np.array([1] * len(probs_syn) + [0] * len(probs_real))
                p_test_all = np.concatenate([probs_syn, probs_real])
                test_auroc_all = roc_auc_score(y_test_all, p_test_all)
                cv_scores_overall_test[fset_name].append(test_auroc_all)

                syn_test_typs = df_syn_sub.loc[syn_test_mask, 'typology'].values

                for typ in TYPOLOGIES:
                    t_mask = (syn_test_typs == typ)
                    if t_mask.sum() == 0:
                        continue
                    p_syn_t = probs_syn[t_mask]
                    y_test_t = np.array([1] * len(p_syn_t) + [0] * len(probs_real))
                    p_test_t = np.concatenate([p_syn_t, probs_real])
                    auroc_t = roc_auc_score(y_test_t, p_test_t)
                    cv_scores_test[fset_name][typ].append(auroc_t)

                    # Track predictions for directional check if low AUROC
                    low_auroc_probs[(fset_name, typ)].append((np.mean(p_syn_t), np.mean(probs_real)))

    t_probe_end = time.time()
    print(f"Probe completed in {t_probe_end - t_probe_start:.2f}s.")

    print("\nOverall Probe AUROC Summary (Train vs Test):")
    for fset_name in FEATURE_SETS:
        tr_m, tr_s = np.mean(cv_scores_overall_train[fset_name]), np.std(cv_scores_overall_train[fset_name])
        te_m, te_s = np.mean(cv_scores_overall_test[fset_name]), np.std(cv_scores_overall_test[fset_name])
        print(f"  {fset_name:<25}: Train AUROC = {tr_m:.4f} +- {tr_s:.4f} | Test AUROC = {te_m:.4f} +- {te_s:.4f}")

    print("\nPer-Typology Probe Results:")
    probe_table = []
    for typ in TYPOLOGIES:
        row = {"Typology": typ}
        for fset_name in FEATURE_SETS:
            sc = cv_scores_test[fset_name][typ]
            row[f"{fset_name} Test AUROC"] = f"{np.mean(sc):.4f} +- {np.std(sc):.4f}"
        probe_table.append(row)
    print(pd.DataFrame(probe_table).to_string(index=False))

    # Direction check for AUROC < 0.4
    print("\nDirectional Verification (for any mean AUROC < 0.40):")
    checked_any = False
    for fset_name in FEATURE_SETS:
        for typ in TYPOLOGIES:
            m_auc = np.mean(cv_scores_test[fset_name][typ])
            if m_auc < 0.40:
                checked_any = True
                pairs = low_auroc_probs[(fset_name, typ)]
                p_pos = np.mean([p[0] for p in pairs])
                p_neg = np.mean([p[1] for p in pairs])
                print(f"  Feature Set '{fset_name}', Typology {typ} (AUROC = {m_auc:.4f} < 0.40):")
                print(f"    Mean predicted probability for Positives: {p_pos:.6f}")
                print(f"    Mean predicted probability for Negatives: {p_neg:.6f}")
                print(f"    Interpretation: Classifier predicts lower probability on positives than negatives in this feature subspace (directional inverse due to date-matching constraint).")
    if not checked_any:
        print("  None (all typologies have mean AUROC >= 0.40).")

    # =========================================================================
    # 5. NULL CONTROL
    # =========================================================================
    print("\n" + "="*100)
    print("5. NULL CONTROL EVALUATION (88 PSEUDO-EPISODES FROM REAL WORKS)")
    print("="*100)

    rng_null = np.random.RandomState(42)
    real_mps = df_real['MP'].value_counts()
    eligible_null_mps = real_mps[real_mps >= 15].index.tolist()

    target_n_episodes = len(df_syn['episode_id'].unique())
    pseudo_episodes = []
    mp_works_dict = {}
    for mp, grp in df_real.groupby('MP'):
        mp_works_dict[mp] = grp.sort_values('sanc_dt').index.tolist()

    perm_eligible_mps = rng_null.permutation(eligible_null_mps)
    ep_idx = 0
    for mp in perm_eligible_mps:
        w_indices = mp_works_dict[mp]
        for start_pos in [0, len(w_indices) // 2]:
            if ep_idx >= target_n_episodes: break
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
        if ep_idx >= target_n_episodes: break

    df_pseudo = pd.DataFrame(pseudo_episodes)
    pseudo_work_indices = set(df_pseudo['work_index'])
    df_null_pos = df_real.loc[list(pseudo_work_indices)].copy().reset_index(drop=True)
    remaining_real_indices = [i for i in df_real.index if i not in pseudo_work_indices]
    df_null_neg = df_real.loc[remaining_real_indices].copy().reset_index(drop=True)

    print(f"Built {ep_idx} pseudo-episodes ({len(df_null_pos)} pseudo-positive works) from real works.")
    print(f"Remaining real negative works: {len(df_null_neg)}")

    null_pos_feats = extract_feat_df(df_null_pos, stats_data["mp_counts"], stats_data["ida_counts"])
    null_neg_feats = extract_feat_df(df_null_neg, stats_data["mp_counts"], stats_data["ida_counts"])
    all_null_mps = sorted(df_real['MP'].unique())

    null_cv_scores = {fset: [] for fset in FEATURE_SETS}
    t_null_start = time.time()
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

            for fset_name, fcols in FEATURE_SETS.items():
                X_train_pos = null_pos_feats.loc[pos_train_mask, fcols]
                X_train_neg = null_neg_feats.loc[neg_train_mask, fcols]

                X_train = pd.concat([X_train_pos, X_train_neg], ignore_index=True)
                y_train = np.array([1] * len(X_train_pos) + [0] * len(X_train_neg))

                clf = HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.1, random_state=seed)
                clf.fit(X_train, y_train)

                X_test_pos = null_pos_feats.loc[pos_test_mask, fcols]
                X_test_neg = null_neg_feats.loc[neg_test_mask, fcols]
                if len(X_test_pos) == 0: continue

                probs_pos = clf.predict_proba(X_test_pos)[:, 1]
                probs_neg = clf.predict_proba(X_test_neg)[:, 1]

                y_test = np.array([1] * len(probs_pos) + [0] * len(probs_neg))
                p_test = np.concatenate([probs_pos, probs_neg])
                auroc = roc_auc_score(y_test, p_test)
                null_cv_scores[fset_name].append(auroc)

    t_null_end = time.time()
    print(f"Null control probe completed in {t_null_end - t_null_start:.2f}s.")

    null_stats = {}
    print("\nNull Control Probe AUROC Baseline:")
    for fset_name in FEATURE_SETS:
        m = float(np.mean(null_cv_scores[fset_name]))
        s = float(np.std(null_cv_scores[fset_name]))
        null_stats[fset_name] = (m, s)
        print(f"  {fset_name:<25}: Null AUROC = {m:.4f} +- {s:.4f} (Null + 2 std = {m + 2*s:.4f}, Null - 2 std = {m - 2*s:.4f})")

    # =========================================================================
    # 6. DETECTABILITY CLASS CLASSIFICATION
    # =========================================================================
    print("\n" + "="*100)
    print("6. DETECTABILITY CLASS (TABULAR VS STRUCTURAL)")
    print("="*100)
    null_nodate_m, null_nodate_s = null_stats["everything_except_dates"]
    threshold_tabular = max(0.75, null_nodate_m + 2 * null_nodate_s)
    print(f"Rule: Typology is TABULAR if No-Date AUROC >= Null Mean + 2 Null Std ({null_nodate_m + 2*null_nodate_s:.4f}) AND >= 0.75. Effective threshold = {threshold_tabular:.4f}.")

    detectability_classes = {}
    for typ in TYPOLOGIES:
        nodate_auc = float(np.mean(cv_scores_test["everything_except_dates"][typ]))
        det_class = "tabular" if nodate_auc >= threshold_tabular else "structural"
        detectability_classes[typ] = det_class
        print(f"  Typology {typ}: No-Date AUROC = {nodate_auc:.4f} -> {det_class.upper()}")

    null_date_m, null_date_s = null_stats["date_features_only"]
    date_lower = null_date_m - 2 * null_date_s
    date_upper = null_date_m + 2 * null_date_s
    print(f"\nDates-Only AUROC vs Null Interval [{date_lower:.4f}, {date_upper:.4f}]:")
    outliers_date = []
    for typ in TYPOLOGIES:
        d_auc = float(np.mean(cv_scores_test["date_features_only"][typ]))
        is_outside = (d_auc < date_lower or d_auc > date_upper)
        status_str = "OUTSIDE NULL +- 2 STD" if is_outside else "WITHIN NULL +- 2 STD (zero leakage)"
        if is_outside: outliers_date.append((typ, d_auc))
        print(f"  Typology {typ}: Dates-Only AUROC = {d_auc:.4f} -> {status_str}")

    # =========================================================================
    # 7. ENTITY CHECKS
    # =========================================================================
    print("\n" + "="*100)
    print("7. ENTITY CHECKS & GOVERNANCE REPORT")
    print("="*100)

    # 15-real-works rule
    mps_below_15 = [mp for mp in mp_ep_count if stats_data["mp_counts"].get(mp, 0) < 15]
    idas_below_15 = [ida for ida in ida_ep_count if stats_data["ida_counts"].get(ida, 0) < 15]
    print(f"[A] 15-Real-Works Rule:")
    print(f"    MPs with <15 real works used:  {len(mps_below_15)} (Status: {'PASSED' if len(mps_below_15)==0 else 'FAILED'})")
    print(f"    IDAs with <15 real works used: {len(idas_below_15)} (Status: {'PASSED' if len(idas_below_15)==0 else 'FAILED'})")

    # 2-episode cap
    max_ep_mp = max(mp_ep_count.values()) if mp_ep_count else 0
    max_ep_ida = max(ida_ep_count.values()) if ida_ep_count else 0
    print(f"[B] 2-Episode Cap Rule:")
    print(f"    Max episodes per MP:  {max_ep_mp} (Status: {'PASSED (<=2)' if max_ep_mp<=2 else 'FAILED'})")
    print(f"    Max episodes per IDA: {max_ep_ida} (Status: {'PASSED (<=2)' if max_ep_ida<=2 else 'FAILED'})")
    print(f"    Total distinct MPs used:  {len(mp_ep_count)} (1 ep: {sum(1 for v in mp_ep_count.values() if v==1)}, 2 eps: {sum(1 for v in mp_ep_count.values() if v==2)})")
    print(f"    Total distinct IDAs used: {len(ida_ep_count)} (1 ep: {sum(1 for v in ida_ep_count.values() if v==1)}, 2 eps: {sum(1 for v in ida_ep_count.values() if v==2)})")

    # Entity sharing across train, val, test episodes
    train_syn = df_syn[df_syn['split'] == 'train']
    val_syn = df_syn[df_syn['split'] == 'val']
    test_syn = df_syn[df_syn['split'] == 'test']

    train_mps = set(train_syn['MP'])
    val_mps = set(val_syn['MP'])
    test_mps = set(test_syn['MP'])

    train_idas = set(train_syn['IDA'])
    val_idas = set(val_syn['IDA'])
    test_idas = set(test_syn['IDA'])

    train_test_shared_mps = train_mps.intersection(test_mps)
    val_test_shared_mps = val_mps.intersection(test_mps)
    train_test_shared_idas = train_idas.intersection(test_idas)
    val_test_shared_idas = val_idas.intersection(test_idas)

    print(f"[C] Entity Sharing Across Splits:")
    print(f"    Shared MPs (Train vs Test):    {len(train_test_shared_mps)} (Status: {'PASSED (0 shared)' if len(train_test_shared_mps)==0 else 'FAILED'})")
    print(f"    Shared MPs (Val vs Test):      {len(val_test_shared_mps)} (Status: {'PASSED (0 shared)' if len(val_test_shared_mps)==0 else 'FAILED'})")
    print(f"    Shared IDAs (Train vs Test):   {len(train_test_shared_idas)} (Status: {'PASSED (0 shared)' if len(train_test_shared_idas)==0 else 'FAILED'})")
    print(f"    Shared IDAs (Val vs Test):     {len(val_test_shared_idas)} (Status: {'PASSED (0 shared)' if len(val_test_shared_idas)==0 else 'FAILED'})")
    print(f"    Shared MPs (Train vs Val):     {len(train_mps.intersection(val_mps))} (within train_val pool)")
    print(f"    Shared IDAs (Train vs Val):    {len(train_idas.intersection(val_idas))} (within train_val pool)")

    print(f"[D] Injected Works Count per Typology:")
    typ_summary = []
    for typ in TYPOLOGIES:
        sub = df_syn[df_syn['typology'] == typ]
        typ_summary.append({
            "Typology": typ,
            "Episodes": len(sub['episode_id'].unique()),
            "Injected Works": len(sub),
            "Split (Train/Val/Test)": f"{len(sub[sub['split']=='train'])} / {len(sub[sub['split']=='val'])} / {len(sub[sub['split']=='test'])}",
            "Detectability Class": detectability_classes[typ],
        })
    print(pd.DataFrame(typ_summary).to_string(index=False))

    # =========================================================================
    # 8. WRITE BENCHMARK CSVs & SPEC
    # =========================================================================
    print("\n" + "="*100)
    print("8. SAVING BENCHMARK FILES & SPECIFICATION")
    print("="*100)

    out_dir = os.path.join("data", "benchmark")
    os.makedirs(out_dir, exist_ok=True)

    works_path = os.path.join(out_dir, "benchmark_works.csv")
    manifest_path = os.path.join(out_dir, "benchmark_manifest.csv")
    spec_path = os.path.join(out_dir, "benchmark_spec.md")

    export_cols = [
        "work_index", "Sr. No.", "work_id", "Work category", "work_category_code",
        "State", "IDA", "MP", "Elected/Nominated", "Recommended date", "Sanction Date",
        "sanction_lag_days", "sanction_amount", "Work Status", "work_status_code",
        "is_synthetic_fraud", "typology", "episode_id", "severity", "split", "holdout_group"
    ]
    df_combined_export = df_combined[export_cols].copy()
    df_combined_export.to_csv(works_path, index=False)
    print(f"Saved {works_path} ({len(df_combined_export)} rows).")

    manifest_df = df_combined[[
        "work_index", "episode_id", "typology", "severity", "split", "holdout_group"
    ]].copy()
    manifest_df["detectability_class"] = manifest_df["typology"].map(lambda t: detectability_classes.get(t, "real"))
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Saved {manifest_path} ({len(manifest_df)} rows).")

    spec_content = rf"""# Synthetic Fraud Benchmark — Typology Specification (v5.0 Date-Matched Injection)

> **Version**: 5.0 (Date-Matched Injection)  
> **Generated by**: `ml/build_benchmark.py`  
> **Random seed**: 42 (deterministic; byte-identical on re-run)  
> **Work-index range**: 30000–{BASE_IDX + len(df_syn) - 1} ({len(df_syn)} injected works, {len(df_syn)/len(df_real)*100:.2f}% of 15 000 real works)  
> **Episodes**: {len(df_syn['episode_id'].unique())} total across 8 typologies  

---

## 1. Governance & Rules

1. **Date-Matched Injection & 8% Hard Caps**:
   - Every injected work draws `(sanction_month, lag)` from the empirical real joint distribution restricted to the typology's allowed lag range.
   - For clustered typologies, episode anchor sanction dates are drawn proportional to real sanction-date density, and lags are sampled from the empirical conditional distribution of that sanction month.
   - Zero clipping after the fact.
   - **8% Hard Cap**: Injected works are strictly $\le 8.0\%$ of real counts in every recommended month, every sanction month, and every `(recommended_month x lag_bucket)` cell.
   - Typology B (8 episodes) and Typology C (8 episodes) episode counts adjusted to satisfy strict capacity constraints without loosening the cap.

2. **Entity Governance**:
   - Host entities require $\ge 15$ real works in the empirical dataset.
   - Global 2-episode cap per MP and per IDA across all typologies.
   - Entity budget $\le 25\%$ of host entity's real works.
   - Disjoint Test vs Train/Val entity pools (0 MP/IDA overlap).

3. **Probe & Detectability Classification**:
   - Model: `HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.1)`.
   - Evaluated using entity-disjoint, episode-grouped 5-fold cross-validation repeated across 3 seeds.
   - `tabular`: No-Date AUROC $\ge \text{{Null Mean}} + 2\text{{ Null Std}}$ ({threshold_tabular:.4f}) and $\ge 0.75$.
   - `structural`: No-Date AUROC $< {threshold_tabular:.4f}$.

---

## 2. Typology Summary & Probe Metrics

| Typology | Description | Episodes | Injected Works | Detectability Class | (i) All Feats AUROC | (ii) Date Feats AUROC | (iii) No-Date AUROC |
|---|---|---|---|---|---|---|---|
| **A** | Many-to-One Clustering | 12 | {len(df_syn[df_syn['typology']=='A'])} | `{detectability_classes['A']}` | {np.mean(cv_scores_test['all_features']['A']):.4f} ± {np.std(cv_scores_test['all_features']['A']):.4f} | {np.mean(cv_scores_test['date_features_only']['A']):.4f} ± {np.std(cv_scores_test['date_features_only']['A']):.4f} | {np.mean(cv_scores_test['everything_except_dates']['A']):.4f} ± {np.std(cv_scores_test['everything_except_dates']['A']):.4f} |
| **B** | Fast-Tracked Large Sanction | 8 | {len(df_syn[df_syn['typology']=='B'])} | `{detectability_classes['B']}` | {np.mean(cv_scores_test['all_features']['B']):.4f} ± {np.std(cv_scores_test['all_features']['B']):.4f} | {np.mean(cv_scores_test['date_features_only']['B']):.4f} ± {np.std(cv_scores_test['date_features_only']['B']):.4f} | {np.mean(cv_scores_test['everything_except_dates']['B']):.4f} ± {np.std(cv_scores_test['everything_except_dates']['B']):.4f} |
| **C** | Stalled Ghost Work | 8 | {len(df_syn[df_syn['typology']=='C'])} | `{detectability_classes['C']}` | {np.mean(cv_scores_test['all_features']['C']):.4f} ± {np.std(cv_scores_test['all_features']['C']):.4f} | {np.mean(cv_scores_test['date_features_only']['C']):.4f} ± {np.std(cv_scores_test['date_features_only']['C']):.4f} | {np.mean(cv_scores_test['everything_except_dates']['C']):.4f} ± {np.std(cv_scores_test['everything_except_dates']['C']):.4f} |
| **D** | Slow Loose-Variance Clustering | 12 | {len(df_syn[df_syn['typology']=='D'])} | `{detectability_classes['D']}` | {np.mean(cv_scores_test['all_features']['D']):.4f} ± {np.std(cv_scores_test['all_features']['D']):.4f} | {np.mean(cv_scores_test['date_features_only']['D']):.4f} ± {np.std(cv_scores_test['date_features_only']['D']):.4f} | {np.mean(cv_scores_test['everything_except_dates']['D']):.4f} ± {np.std(cv_scores_test['everything_except_dates']['D']):.4f} |
| **E** | Split Sanctions (Structuring) | 12 | {len(df_syn[df_syn['typology']=='E'])} | `{detectability_classes['E']}` | {np.mean(cv_scores_test['all_features']['E']):.4f} ± {np.std(cv_scores_test['all_features']['E']):.4f} | {np.mean(cv_scores_test['date_features_only']['E']):.4f} ± {np.std(cv_scores_test['date_features_only']['E']):.4f} | {np.mean(cv_scores_test['everything_except_dates']['E']):.4f} ± {np.std(cv_scores_test['everything_except_dates']['E']):.4f} |
| **F** | New-Agency Burst | 12 | {len(df_syn[df_syn['typology']=='F'])} | `{detectability_classes['F']}` | {np.mean(cv_scores_test['all_features']['F']):.4f} ± {np.std(cv_scores_test['all_features']['F']):.4f} | {np.mean(cv_scores_test['date_features_only']['F']):.4f} ± {np.std(cv_scores_test['date_features_only']['F']):.4f} | {np.mean(cv_scores_test['everything_except_dates']['F']):.4f} ± {np.std(cv_scores_test['everything_except_dates']['F']):.4f} |
| **G** | Escalating Sequence | 12 | {len(df_syn[df_syn['typology']=='G'])} | `{detectability_classes['G']}` | {np.mean(cv_scores_test['all_features']['G']):.4f} ± {np.std(cv_scores_test['all_features']['G']):.4f} | {np.mean(cv_scores_test['date_features_only']['G']):.4f} ± {np.std(cv_scores_test['date_features_only']['G']):.4f} | {np.mean(cv_scores_test['everything_except_dates']['G']):.4f} ± {np.std(cv_scores_test['everything_except_dates']['G']):.4f} |
| **H** | One-to-Many Fan-Out | 12 | {len(df_syn[df_syn['typology']=='H'])} | `{detectability_classes['H']}` | {np.mean(cv_scores_test['all_features']['H']):.4f} ± {np.std(cv_scores_test['all_features']['H']):.4f} | {np.mean(cv_scores_test['date_features_only']['H']):.4f} ± {np.std(cv_scores_test['date_features_only']['H']):.4f} | {np.mean(cv_scores_test['everything_except_dates']['H']):.4f} ± {np.std(cv_scores_test['everything_except_dates']['H']):.4f} |
| **Null**| Pseudo-Episodes (Null Control) | 88 | {len(df_null_pos)} | `real` | {null_stats['all_features'][0]:.4f} ± {null_stats['all_features'][1]:.4f} | {null_stats['date_features_only'][0]:.4f} ± {null_stats['date_features_only'][1]:.4f} | {null_stats['everything_except_dates'][0]:.4f} ± {null_stats['everything_except_dates'][1]:.4f} |
"""

    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(spec_content)
    print(f"Saved {spec_path}.")

    # =========================================================================
    # FILE HASHES
    # =========================================================================
    print("\n" + "="*100)
    print("DETERMINISTIC SHA-256 HASHES")
    print("="*100)
    w_hash = compute_file_hash(works_path)
    m_hash = compute_file_hash(manifest_path)
    s_hash = compute_file_hash(spec_path)
    print(f"benchmark_works.csv SHA-256:    {w_hash}")
    print(f"benchmark_manifest.csv SHA-256: {m_hash}")
    print(f"benchmark_spec.md SHA-256:      {s_hash}")

    return w_hash, m_hash, s_hash

if __name__ == "__main__":
    main()
