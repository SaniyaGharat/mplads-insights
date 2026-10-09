"""
Benchmark v5 Shared Feature Rule and Column Definitions.

Governance Rule:
Downstream models and detection probes may ONLY use tabular features
representing work-level properties:
- Amount: `sanction_amount` (numeric float/int)
- Lag: `sanction_lag_days` (numeric integer)
- Status: `work_status_code` (categorical integer: 0 to 5)
- Category: `work_category_code` (categorical integer: 0 to 2)

EXCLUDED BY DEFAULT:
- Date columns (`Sanction Date`, `Recommended date`, year, month, day, dayofweek, ordinals)
- Entity real-count columns (`mp_real_count`, `ida_real_count`, work counts)
- Direct entity IDs (`MP`, `IDA`, `State`) unless explicitly evaluating graph / structural models.
"""

import pandas as pd
import numpy as np

# Canonical feature names allowed for downstream tabular models
ALLOWED_TABULAR_FEATURES = [
    "amount",
    "lag",
    "status",
    "category",
]

# Source column mappings from benchmark_works.csv
COLUMN_MAPPING = {
    "sanction_amount": "amount",
    "sanction_lag_days": "lag",
    "work_status_code": "status",
    "work_category_code": "category",
}

# Explicitly excluded columns from default tabular detection pipelines
EXCLUDED_DATE_COLUMNS = [
    "Sanction Date",
    "Recommended date",
    "sanc_dt",
    "rec_dt",
    "sanc_month",
    "rec_month",
    "sanction_date_ordinal",
    "recommended_date_ordinal",
]

EXCLUDED_ENTITY_COUNT_COLUMNS = [
    "mp_real_count",
    "ida_real_count",
    "mp_work_count",
    "ida_work_count",
]

# Feature sets used for benchmark audit and probe diagnostic comparisons
FEATURE_SETS = {
    "all_features": [
        "amount",
        "lag",
        "status",
        "category",
        "mp_real_count",
        "ida_real_count",
        "sanction_date_ordinal",
        "recommended_date_ordinal",
    ],
    "date_features_only": [
        "sanction_date_ordinal",
        "recommended_date_ordinal",
    ],
    "everything_except_dates": [
        "amount",
        "lag",
        "status",
        "category",
        "mp_real_count",
        "ida_real_count",
    ],
    "shared_tabular_only": [
        "amount",
        "lag",
        "status",
        "category",
    ],
}


def extract_features(df, mp_counts=None, ida_counts=None):
    """
    Extract a standard feature DataFrame from benchmark or real works DataFrame.
    """
    sanc_dt = pd.to_datetime(df["Sanction Date"])
    rec_dt = pd.to_datetime(df["Recommended date"])
    sanc_ord = sanc_dt.map(lambda d: d.toordinal()).values
    rec_ord = rec_dt.map(lambda d: d.toordinal()).values

    if mp_counts is not None:
        mp_c = df["MP"].map(lambda m: mp_counts.get(m, 0)).values
    else:
        mp_c = np.zeros(len(df), dtype=int)

    if ida_counts is not None:
        ida_c = df["IDA"].map(lambda i: ida_counts.get(i, 0)).values
    else:
        ida_c = np.zeros(len(df), dtype=int)

    amt = df["sanction_amount"].values
    lag = df["sanction_lag_days"].values
    status = df["work_status_code"].values if "work_status_code" in df.columns else np.zeros(len(df), dtype=int)
    cat = df["work_category_code"].values if "work_category_code" in df.columns else np.ones(len(df), dtype=int)

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
