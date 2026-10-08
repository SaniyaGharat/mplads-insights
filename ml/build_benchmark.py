"""
ml/build_benchmark.py
=====================
Deterministic synthetic fraud benchmark generator (v2).

Produces 8 typologies (A–H) × 12 episodes (4 per severity) with randomized
parameters drawn from ranges defined in data/benchmark/benchmark_spec.md.

Outputs (all under data/benchmark/):
  - benchmark_works.csv     : real 15 000 + 500 injected works
  - benchmark_manifest.csv  : per-work metadata (episode, typology, severity, split, holdout_group, detectability_class)

Safety:
  - Reads data/works_sanctioned_clean.csv READ-ONLY.
  - Does NOT modify any existing file.
  - work_index starts at 30 000.
  - Fixed random seed 42; byte-identical on re-run.

Usage:
    python ml/build_benchmark.py
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score

# ── Global seed ──────────────────────────────────────────────────────────────
SEED = 42

# ── Constants ────────────────────────────────────────────────────────────────
BASE_IDX = 30_000
TYPOLOGIES = list("ABCDEFGH")
SEVERITIES = ["subtle", "moderate", "obvious"]
EPISODES_PER_SEVERITY = 4
EPISODES_PER_TYPOLOGY = 12  # 4 * 3 = 12 episodes per typology (96 episodes total)

STATUS_TO_CODE = {
    "Physical Inspection": 0,
    "Sanction": 1,
    "Time Estimation": 2,
    "Vendor Identification": 3,
    "Work Completed": 4,
    "Work partially Completed": 5,
}


def _load_real_data():
    """Load the real dataset (read-only)."""
    path = os.path.join("data", "works_sanctioned_clean.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def _real_quantile(series, q):
    return float(np.quantile(series.dropna(), q))


def _sample_status(rng, n, real_status_dist):
    """Sample work statuses proportionally from real distribution."""
    statuses = sorted(real_status_dist.keys())
    probs = np.array([real_status_dist[s] for s in statuses], dtype=float)
    probs /= probs.sum()
    chosen = rng.choice(len(statuses), size=n, p=probs)
    return [statuses[c] for c in chosen]


def _status_code(status_name):
    return STATUS_TO_CODE.get(status_name, 0)


def _clamp_amount(val, real_min, real_max):
    return float(np.clip(val, real_min, real_max))


def _clamp_lag(val, real_max):
    return int(np.clip(round(val), 0, real_max))


def _make_rec_date(sanction_date, lag_days):
    """Compute recommended date from sanction date and lag."""
    sanc = pd.Timestamp(sanction_date)
    rec = sanc - pd.Timedelta(days=int(lag_days))
    return rec.strftime("%Y-%m-%d")


# ── Typology generators ─────────────────────────────────────────────────────

def _gen_typology_A(rng, df_real, severity, episode_id, idx_start, stats):
    """Many-to-one clustering with State consistency."""
    cfg = {
        "subtle":   {"n_works": (4, 5), "n_mps": (3, 4), "window": (25, 40), "cv": (0.10, 0.15), "amt_q": (0.50, 0.75)},
        "moderate": {"n_works": (5, 6), "n_mps": (3, 3), "window": (10, 20), "cv": (0.04, 0.09), "amt_q": (0.60, 0.90)},
        "obvious":  {"n_works": (5, 6), "n_mps": (2, 3), "window": (3, 7), "cv": (0.01, 0.03), "amt_q": (0.70, 0.95)},
    }[severity]

    n_works = rng.randint(cfg["n_works"][0], cfg["n_works"][1] + 1)
    n_mps = rng.randint(cfg["n_mps"][0], cfg["n_mps"][1] + 1)
    window = rng.randint(cfg["window"][0], cfg["window"][1] + 1)
    cv = rng.uniform(cfg["cv"][0], cfg["cv"][1])
    amt_center = _real_quantile(df_real["sanction_amount"], rng.uniform(cfg["amt_q"][0], cfg["amt_q"][1]))

    eligible_idas = sorted([ida for ida, st in stats["ida_state"].items() if len(stats["mps_by_state"].get(st, [])) >= 3])
    target_ida = rng.choice(eligible_idas)
    state = stats["ida_state"][target_ida]

    state_mps = stats["mps_by_state"][state]
    perm_mps = rng.permutation(len(state_mps))
    chosen_mps = [state_mps[i] for i in perm_mps[:min(n_mps, len(state_mps))]]

    std = cv * amt_center
    amounts = rng.normal(amt_center, std, size=n_works)
    amounts = np.clip(amounts, stats["amt_min"], stats["amt_max"])

    base_date = pd.Timestamp("2025-02-01") + pd.Timedelta(days=rng.randint(0, 100))
    date_offsets = rng.randint(0, window + 1, size=n_works)
    sanction_dates = [base_date + pd.Timedelta(days=int(d)) for d in sorted(date_offsets)]
    lags = rng.randint(stats["lag_p25"], stats["lag_p75"] + 1, size=n_works)
    statuses = _sample_status(rng, n_works, stats["status_dist"])

    records = []
    for i in range(n_works):
        mp = chosen_mps[i % len(chosen_mps)]
        lag = _clamp_lag(lags[i], stats["lag_max"])
        amt = int(round(_clamp_amount(amounts[i], stats["amt_min"], stats["amt_max"])))
        sd = sanction_dates[i].strftime("%Y-%m-%d")
        records.append({
            "work_index": idx_start + i,
            "Sr. No.": idx_start + i + 1,
            "work_id": f"WS/BENCH-A{episode_id:02d}-{i+1:02d}",
            "Work category": "Normal/Others",
            "work_category_code": 1,
            "State": state,
            "IDA": target_ida,
            "MP": mp,
            "Elected/Nominated": "Elected MP",
            "Work description": f"Community infrastructure development work package {i+1}",
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
            "detectability_class": "structural",
        })
    return records


def _gen_typology_B(rng, df_real, severity, episode_id, idx_start, stats):
    """Fast-tracked large sanction."""
    cfg = {
        "subtle":   {"n_works": (4, 5), "amt_q": (0.75, 0.90), "lag": (10, 25)},
        "moderate": {"n_works": (5, 6), "amt_q": (0.90, 0.97), "lag": (2, 6)},
        "obvious":  {"n_works": (5, 6), "amt_q": (0.95, 0.99), "lag": (0, 2)},
    }[severity]

    n_works = rng.randint(cfg["n_works"][0], cfg["n_works"][1] + 1)
    pairs_pool = stats["valid_pairs"]
    perm = rng.permutation(len(pairs_pool))
    chosen_pairs = [pairs_pool[i] for i in perm[:n_works]]

    base_date = pd.Timestamp("2025-02-01") + pd.Timedelta(days=rng.randint(0, 120))
    records = []
    for i in range(n_works):
        mp, ida, state = chosen_pairs[i]
        amt_q = rng.uniform(cfg["amt_q"][0], cfg["amt_q"][1])
        amt = int(round(_clamp_amount(_real_quantile(df_real["sanction_amount"], amt_q), stats["amt_min"], stats["amt_max"])))
        lag = rng.randint(cfg["lag"][0], cfg["lag"][1] + 1)
        lag = _clamp_lag(lag, stats["lag_max"])
        sd = (base_date + pd.Timedelta(days=rng.randint(0, 30))).strftime("%Y-%m-%d")
        records.append({
            "work_index": idx_start + i,
            "Sr. No.": idx_start + i + 1,
            "work_id": f"WS/BENCH-B{episode_id:02d}-{i+1:02d}",
            "Work category": "Normal/Others",
            "work_category_code": 1,
            "State": state,
            "IDA": ida,
            "MP": mp,
            "Elected/Nominated": "Elected MP",
            "Work description": f"Priority infrastructure scheme {i+1}",
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
            "detectability_class": "tabular",
        })
    return records


def _gen_typology_C(rng, df_real, severity, episode_id, idx_start, stats):
    """Stalled ghost work (with date boundaries strictly inside real ranges)."""
    cfg = {
        "subtle":   {"n_works": (4, 5), "amt_q": (0.25, 0.70), "lag": (250, 400), "statuses": ["Physical Inspection", "Vendor Identification"]},
        "moderate": {"n_works": (5, 6), "amt_q": (0.20, 0.50), "lag": (450, 600), "statuses": ["Physical Inspection", "Vendor Identification"]},
        "obvious":  {"n_works": (5, 6), "amt_q": (0.15, 0.40), "lag": (600, 720), "statuses": ["Physical Inspection"]},
    }[severity]

    n_works = rng.randint(cfg["n_works"][0], cfg["n_works"][1] + 1)
    pairs_pool = stats["valid_pairs"]
    perm = rng.permutation(len(pairs_pool))
    chosen_pairs = [pairs_pool[i] for i in perm[:n_works]]

    records = []
    for i in range(n_works):
        mp, ida, state = chosen_pairs[i]
        amt_q = rng.uniform(cfg["amt_q"][0], cfg["amt_q"][1])
        amt = int(round(_clamp_amount(_real_quantile(df_real["sanction_amount"], amt_q), stats["amt_min"], stats["amt_max"])))
        lag = rng.randint(cfg["lag"][0], min(cfg["lag"][1], stats["lag_max"]) + 1)
        lag = _clamp_lag(lag, stats["lag_max"])
        status = rng.choice(cfg["statuses"])

        # Pick rec_base in [2024-07-15, 2024-09-01] so rec_base >= 2024-07-08 and rec_base + lag <= 2026-09-04
        rec_base = pd.Timestamp("2024-07-15") + pd.Timedelta(days=rng.randint(0, 45))
        sd_dt = rec_base + pd.Timedelta(days=lag)
        if sd_dt > stats["real_max_sanc"]:
            sd_dt = stats["real_max_sanc"] - pd.Timedelta(days=rng.randint(1, 10))
            rec_base = sd_dt - pd.Timedelta(days=lag)

        sd = sd_dt.strftime("%Y-%m-%d")
        rd = rec_base.strftime("%Y-%m-%d")

        records.append({
            "work_index": idx_start + i,
            "Sr. No.": idx_start + i + 1,
            "work_id": f"WS/BENCH-C{episode_id:02d}-{i+1:02d}",
            "Work category": "Normal/Others",
            "work_category_code": 1,
            "State": state,
            "IDA": ida,
            "MP": mp,
            "Elected/Nominated": "Elected MP",
            "Work description": f"Municipal facility stage-1 work {i+1}",
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
            "detectability_class": "tabular",
        })
    return records


def _gen_typology_D(rng, df_real, severity, episode_id, idx_start, stats):
    """Slow loose-variance clustering."""
    cfg = {
        "subtle":   {"n_works": (4, 5), "n_mps": (3, 4), "window": (80, 120), "cv": (0.18, 0.25), "amt_q": (0.40, 0.70)},
        "moderate": {"n_works": (5, 6), "n_mps": (3, 3), "window": (50, 80), "cv": (0.10, 0.17), "amt_q": (0.50, 0.80)},
        "obvious":  {"n_works": (5, 6), "n_mps": (2, 3), "window": (30, 50), "cv": (0.06, 0.09), "amt_q": (0.50, 0.80)},
    }[severity]

    n_works = rng.randint(cfg["n_works"][0], cfg["n_works"][1] + 1)
    n_mps = rng.randint(cfg["n_mps"][0], cfg["n_mps"][1] + 1)
    window = rng.randint(cfg["window"][0], cfg["window"][1] + 1)
    cv = rng.uniform(cfg["cv"][0], cfg["cv"][1])
    amt_center = _real_quantile(df_real["sanction_amount"], rng.uniform(cfg["amt_q"][0], cfg["amt_q"][1]))

    eligible_idas = sorted([ida for ida, st in stats["ida_state"].items() if len(stats["mps_by_state"].get(st, [])) >= 3])
    target_ida = rng.choice(eligible_idas)
    state = stats["ida_state"][target_ida]

    state_mps = stats["mps_by_state"][state]
    perm_mps = rng.permutation(len(state_mps))
    chosen_mps = [state_mps[i] for i in perm_mps[:min(n_mps, len(state_mps))]]

    std = cv * amt_center
    amounts = rng.normal(amt_center, std, size=n_works)
    amounts = np.clip(amounts, stats["amt_min"], stats["amt_max"])

    base_date = pd.Timestamp("2025-01-15") + pd.Timedelta(days=rng.randint(0, 100))
    date_offsets = rng.randint(0, window + 1, size=n_works)
    sanction_dates = [base_date + pd.Timedelta(days=int(d)) for d in sorted(date_offsets)]
    lags = rng.randint(stats["lag_p25"], stats["lag_p75"] + 1, size=n_works)
    statuses = _sample_status(rng, n_works, stats["status_dist"])

    records = []
    for i in range(n_works):
        mp = chosen_mps[i % len(chosen_mps)]
        lag = _clamp_lag(lags[i], stats["lag_max"])
        amt = int(round(_clamp_amount(amounts[i], stats["amt_min"], stats["amt_max"])))
        sd = sanction_dates[i].strftime("%Y-%m-%d")
        records.append({
            "work_index": idx_start + i,
            "Sr. No.": idx_start + i + 1,
            "work_id": f"WS/BENCH-D{episode_id:02d}-{i+1:02d}",
            "Work category": "Normal/Others",
            "work_category_code": 1,
            "State": state,
            "IDA": target_ida,
            "MP": mp,
            "Elected/Nominated": "Elected MP",
            "Work description": f"Area development scheme block {i+1}",
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
            "detectability_class": "structural",
        })
    return records


def _gen_typology_E(rng, df_real, severity, episode_id, idx_start, stats):
    """Split sanctions (structuring)."""
    cfg = {
        "subtle":   {"n_works": (4, 5), "thresholds": [500_000, 1_000_000], "pct": (0.75, 0.95), "window": (20, 40)},
        "moderate": {"n_works": (5, 6), "thresholds": [1_000_000, 2_500_000], "pct": (0.85, 0.99), "window": (10, 20)},
        "obvious":  {"n_works": (5, 6), "thresholds": [2_500_000], "pct": (0.92, 0.995), "window": (3, 10)},
    }[severity]

    n_works = rng.randint(cfg["n_works"][0], cfg["n_works"][1] + 1)
    threshold = rng.choice(cfg["thresholds"])

    pairs_pool = stats["valid_pairs"]
    perm = rng.permutation(len(pairs_pool))
    mp, ida, state = pairs_pool[perm[0]]

    base_date = pd.Timestamp("2025-02-01") + pd.Timedelta(days=rng.randint(0, 150))
    window = rng.randint(cfg["window"][0], cfg["window"][1] + 1)
    statuses_pool = _sample_status(rng, n_works, stats["status_dist"]) if severity == "subtle" else ["Sanction"] * n_works

    records = []
    for i in range(n_works):
        pct = rng.uniform(cfg["pct"][0], cfg["pct"][1])
        amt = int(round(_clamp_amount(threshold * pct, stats["amt_min"], stats["amt_max"])))
        lag = rng.randint(stats["lag_p25"], stats["lag_p75"] + 1)
        lag = _clamp_lag(lag, stats["lag_max"])
        sd = (base_date + pd.Timedelta(days=rng.randint(0, window + 1))).strftime("%Y-%m-%d")
        records.append({
            "work_index": idx_start + i,
            "Sr. No.": idx_start + i + 1,
            "work_id": f"WS/BENCH-E{episode_id:02d}-{i+1:02d}",
            "Work category": "Normal/Others",
            "work_category_code": 1,
            "State": state,
            "IDA": ida,
            "MP": mp,
            "Elected/Nominated": "Elected MP",
            "Work description": f"Public utility improvement sub-package {i+1}",
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
            "detectability_class": "tabular",
        })
    return records


def _gen_typology_F(rng, df_real, severity, episode_id, idx_start, stats):
    """New-agency burst (new IDA within MP's own state)."""
    cfg = {
        "subtle":   {"n_works": (4, 5), "window": (30, 60)},
        "moderate": {"n_works": (5, 6), "window": (15, 30)},
        "obvious":  {"n_works": (5, 6), "window": (5, 15)},
    }[severity]

    n_works = rng.randint(cfg["n_works"][0], cfg["n_works"][1] + 1)
    window = rng.randint(cfg["window"][0], cfg["window"][1] + 1)

    candidates = sorted([
        mp for mp, st in stats["mp_home_state"].items()
        if len(set(df_real[df_real["MP"] == mp]["IDA"])) <= 2
        and len(set(stats["idas_by_state"].get(st, [])) - set(df_real[df_real["MP"] == mp]["IDA"])) >= 1
    ])
    perm_cand = rng.permutation(len(candidates))
    mp = candidates[perm_cand[0]]
    state = stats["mp_home_state"][mp]

    used_idas = set(df_real[df_real["MP"] == mp]["IDA"])
    available_idas = sorted(list(set(stats["idas_by_state"][state]) - used_idas))
    perm_idas = rng.permutation(len(available_idas))
    new_ida = available_idas[perm_idas[0]]

    base_date = pd.Timestamp("2025-03-01") + pd.Timedelta(days=rng.randint(0, 120))
    statuses = _sample_status(rng, n_works, stats["status_dist"])

    records = []
    for i in range(n_works):
        amt_q = rng.uniform(0.30, 0.70)
        amt = int(round(_clamp_amount(_real_quantile(df_real["sanction_amount"], amt_q), stats["amt_min"], stats["amt_max"])))
        lag = rng.randint(stats["lag_p25"], stats["lag_p75"] + 1)
        lag = _clamp_lag(lag, stats["lag_max"])
        sd = (base_date + pd.Timedelta(days=rng.randint(0, window + 1))).strftime("%Y-%m-%d")
        records.append({
            "work_index": idx_start + i,
            "Sr. No.": idx_start + i + 1,
            "work_id": f"WS/BENCH-F{episode_id:02d}-{i+1:02d}",
            "Work category": "Normal/Others",
            "work_category_code": 1,
            "State": state,
            "IDA": new_ida,
            "MP": mp,
            "Elected/Nominated": "Elected MP",
            "Work description": f"Community development initiative {i+1}",
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
            "detectability_class": "structural",
        })
    return records


def _gen_typology_G(rng, df_real, severity, episode_id, idx_start, stats):
    """Escalating sequence."""
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

    pairs_pool = stats["valid_pairs"]
    perm = rng.permutation(len(pairs_pool))
    mp, ida, state = pairs_pool[perm[0]]

    base_date = pd.Timestamp("2025-01-15") + pd.Timedelta(days=rng.randint(0, 100))
    statuses = _sample_status(rng, n_works, stats["status_dist"])

    records = []
    current_amt = start_amt
    current_date = base_date
    for i in range(n_works):
        amt = min(current_amt, cap_amt)
        amt = int(round(_clamp_amount(amt, stats["amt_min"], stats["amt_max"])))
        lag = rng.randint(stats["lag_p25"], stats["lag_p75"] + 1)
        lag = _clamp_lag(lag, stats["lag_max"])
        sd = current_date.strftime("%Y-%m-%d")
        records.append({
            "work_index": idx_start + i,
            "Sr. No.": idx_start + i + 1,
            "work_id": f"WS/BENCH-G{episode_id:02d}-{i+1:02d}",
            "Work category": "Normal/Others",
            "work_category_code": 1,
            "State": state,
            "IDA": ida,
            "MP": mp,
            "Elected/Nominated": "Elected MP",
            "Work description": f"Phased development project stage {i+1}",
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
            "detectability_class": "structural",
        })
        current_amt *= (1 + step_pct)
        spacing = rng.randint(cfg["spacing"][0], cfg["spacing"][1] + 1)
        current_date += pd.Timedelta(days=spacing)
    return records


def _gen_typology_H(rng, df_real, severity, episode_id, idx_start, stats):
    """One-to-many fan-out (single MP to multiple IDAs in same state)."""
    cfg = {
        "subtle":   {"n_idas": (4, 5), "cv": (0.10, 0.15), "amt_q": (0.50, 0.75), "window": (20, 40)},
        "moderate": {"n_idas": (5, 6), "cv": (0.04, 0.09), "amt_q": (0.60, 0.85), "window": (7, 15)},
        "obvious":  {"n_idas": (5, 6), "cv": (0.01, 0.03), "amt_q": (0.70, 0.95), "window": (1, 5)},
    }[severity]

    n_idas = rng.randint(cfg["n_idas"][0], cfg["n_idas"][1] + 1)
    cv = rng.uniform(cfg["cv"][0], cfg["cv"][1])
    amt_center = _real_quantile(df_real["sanction_amount"], rng.uniform(cfg["amt_q"][0], cfg["amt_q"][1]))
    window = rng.randint(cfg["window"][0], cfg["window"][1] + 1)

    candidates = sorted([
        mp for mp, st in stats["mp_home_state"].items()
        if len(set(stats["idas_by_state"].get(st, [])) - set(df_real[df_real["MP"] == mp]["IDA"])) >= n_idas
    ])
    if not candidates:
        candidates = sorted([
            mp for mp, st in stats["mp_home_state"].items()
            if len(stats["idas_by_state"].get(st, [])) >= n_idas
        ])
    perm_cand = rng.permutation(len(candidates))
    mp = candidates[perm_cand[0]]
    state = stats["mp_home_state"][mp]

    used_idas = set(df_real[df_real["MP"] == mp]["IDA"])
    available_idas = sorted(list(set(stats["idas_by_state"][state]) - used_idas))
    if len(available_idas) < n_idas:
        available_idas = stats["idas_by_state"][state]
    perm_idas = rng.permutation(len(available_idas))
    chosen_idas = [available_idas[i] for i in perm_idas[:n_idas]]

    n_works = n_idas
    std = cv * amt_center
    amounts = rng.normal(amt_center, std, size=n_works)
    amounts = np.clip(amounts, stats["amt_min"], stats["amt_max"])

    base_date = pd.Timestamp("2025-02-01") + pd.Timedelta(days=rng.randint(0, 120))
    lags = rng.randint(stats["lag_p25"], stats["lag_p75"] + 1, size=n_works)
    statuses = _sample_status(rng, n_works, stats["status_dist"])

    records = []
    for i in range(n_works):
        ida = chosen_idas[i]
        lag = _clamp_lag(lags[i], stats["lag_max"])
        amt = int(round(_clamp_amount(amounts[i], stats["amt_min"], stats["amt_max"])))
        sd = (base_date + pd.Timedelta(days=rng.randint(0, window + 1))).strftime("%Y-%m-%d")
        records.append({
            "work_index": idx_start + i,
            "Sr. No.": idx_start + i + 1,
            "work_id": f"WS/BENCH-H{episode_id:02d}-{i+1:02d}",
            "Work category": "Normal/Others",
            "work_category_code": 1,
            "State": state,
            "IDA": ida,
            "MP": mp,
            "Elected/Nominated": "Elected MP",
            "Work description": f"Standard public works project {i+1}",
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
            "detectability_class": "structural",
        })
    return records

GENERATORS = {
    "A": _gen_typology_A,
    "B": _gen_typology_B,
    "C": _gen_typology_C,
    "D": _gen_typology_D,
    "E": _gen_typology_E,
    "F": _gen_typology_F,
    "G": _gen_typology_G,
    "H": _gen_typology_H,
}

# ── Split assignment ─────────────────────────────────────────────────────────

def _assign_episode_splits():
    """
    For each typology x severity (4 episodes each):
      2 train / 1 val / 1 test
    """
    split_map = {}
    for typ in TYPOLOGIES:
        for sev_idx in range(3):
            base_ep = sev_idx * 4 + 1
            splits = ["train", "train", "val", "test"]
            for i, sp in enumerate(splits):
                eid = f"{typ}{base_ep + i:02d}"
                split_map[eid] = sp
    return split_map


def _assign_real_splits(rng, n_real):
    """50% train (7500), 25% val (3750), 25% test (3750)."""
    indices = rng.permutation(np.arange(n_real))
    n_train = int(0.50 * n_real)
    n_val = int(0.25 * n_real)
    splits = np.empty(n_real, dtype=object)
    splits[indices[:n_train]] = "train"
    splits[indices[n_train:n_train + n_val]] = "val"
    splits[indices[n_train + n_val:]] = "test"
    return splits


# ── Main build ───────────────────────────────────────────────────────────────

def build_benchmark():
    print("=" * 90)
    print("MPLADS Synthetic Fraud Benchmark Generator (v2)")
    print("=" * 90)
    print()

    rng = np.random.RandomState(SEED)
    df_real = _load_real_data()

    mp_home_state = df_real.groupby('MP')['State'].agg(lambda s: s.mode().iloc[0]).to_dict()
    ida_state = df_real.groupby('IDA')['State'].agg(lambda s: s.mode().iloc[0]).to_dict()
    mp_work_counts = df_real['MP'].value_counts().to_dict()
    ida_work_counts = df_real['IDA'].value_counts().to_dict()

    mps_by_state = {}
    for mp, st in mp_home_state.items():
        mps_by_state.setdefault(st, []).append(mp)
    for st in mps_by_state:
        mps_by_state[st].sort()

    idas_by_state = {}
    for ida, st in ida_state.items():
        idas_by_state.setdefault(st, []).append(ida)
    for st in idas_by_state:
        idas_by_state[st].sort()

    valid_pairs = sorted([
        (row['MP'], row['IDA'], row['State'])
        for _, row in df_real[['MP', 'IDA', 'State']].drop_duplicates().iterrows()
        if mp_home_state.get(row['MP']) == ida_state.get(row['IDA'])
    ])

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
        "valid_pairs": valid_pairs,
        "mp_work_counts": mp_work_counts,
        "ida_work_counts": ida_work_counts,
        "real_min_sanc": real_min_sanc,
        "real_max_sanc": real_max_sanc,
        "real_min_rec": real_min_rec,
        "real_max_rec": real_max_rec,
    }

    all_records = []
    idx_cursor = BASE_IDX
    severity_schedule = ["subtle"] * 4 + ["moderate"] * 4 + ["obvious"] * 4

    for typ in TYPOLOGIES:
        gen_fn = GENERATORS[typ]
        for ep_num in range(EPISODES_PER_TYPOLOGY):
            severity = severity_schedule[ep_num]
            episode_id = ep_num + 1
            records = gen_fn(rng, df_real, severity, episode_id, idx_cursor, stats)
            all_records.extend(records)
            idx_cursor += len(records)

    df_syn = pd.DataFrame(all_records)
    print(f"Generated {len(df_syn)} synthetic fraud works across {df_syn['episode_id'].nunique()} episodes.")
    print(f"Injected proportion: {len(df_syn)/len(df_real)*100:.2f}% of 15,000 real works ({len(df_syn)/(len(df_real)+len(df_syn))*100:.2f}% of combined).")
    print(f"Work index range: [{df_syn['work_index'].min()}, {df_syn['work_index'].max()}]\n")

    # Splits
    episode_splits = _assign_episode_splits()
    df_syn["split"] = df_syn["episode_id"].map(episode_splits)
    df_syn["holdout_group"] = df_syn["typology"]

    real_splits = _assign_real_splits(rng, len(df_real))
    df_real_aug = df_real.copy()
    df_real_aug["work_index"] = df_real_aug.index
    df_real_aug["is_synthetic_fraud"] = False
    df_real_aug["typology"] = "real"
    df_real_aug["episode_id"] = "real"
    df_real_aug["severity"] = "real"
    df_real_aug["split"] = real_splits
    df_real_aug["holdout_group"] = "real"
    df_real_aug["detectability_class"] = "real"

    out_cols = list(df_real.columns) + [
        "work_index", "is_synthetic_fraud", "typology", "episode_id", "severity",
        "split", "holdout_group", "detectability_class"
    ]
    for c in out_cols:
        if c not in df_syn.columns:
            df_syn[c] = None

    df_combined = pd.concat([df_real_aug[out_cols], df_syn[out_cols]], ignore_index=True)
    manifest_cols = ["work_index", "episode_id", "typology", "severity", "split", "holdout_group", "detectability_class"]
    df_manifest = df_combined[manifest_cols].copy()

    out_dir = os.path.join("data", "benchmark")
    os.makedirs(out_dir, exist_ok=True)
    works_path = os.path.join(out_dir, "benchmark_works.csv")
    manifest_path = os.path.join(out_dir, "benchmark_manifest.csv")

    df_combined.to_csv(works_path, index=False)
    df_manifest.to_csv(manifest_path, index=False)

    print(f"Saved: {works_path} ({len(df_combined)} rows)")
    print(f"Saved: {manifest_path} ({len(df_manifest)} rows)\n")

    return df_real, df_syn, df_combined, df_manifest, stats


def compute_file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    df_real, df_syn, df_combined, df_manifest, stats = build_benchmark()
    works_path = os.path.join("data", "benchmark", "benchmark_works.csv")
    manifest_path = os.path.join("data", "benchmark", "benchmark_manifest.csv")
    print("─── File hashes (SHA-256) ───")
    print(f"  benchmark_works.csv:    {compute_file_hash(works_path)}")
    print(f"  benchmark_manifest.csv: {compute_file_hash(manifest_path)}")
    print()


if __name__ == "__main__":
    main()
