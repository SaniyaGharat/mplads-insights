"""
ml/inject_pattern_d.py
======================
Generates Pattern D: A harder, more realistic clustering fraud benchmark
to evaluate the out-of-distribution generalization of community anomaly detection.

Characteristics:
- 10 synthetic works (work_index 20030 - 20039).
- 3 real MPs channeling works to the same real IDA:
    MP 1: Bhagirath Chaudhary (2024-2029)
    MP 2: Shri Hanuman Beniwal (2024-2029)
    MP 3: BRIJENDRA SINGH OLA (2024-2029)
    IDA:  AJMER(DISTRICT COLLECTOR AJMER_IDA), Rajasthan
- Slower-moving collusion over a 70-day window (2025-03-01 to 2025-05-10),
  exceeding the standard 30-day rolling window.
- Moderate amount variation with CV ~ 0.0988 (between 0.08 and 0.15),
  much looser than Pattern A's near-identical amounts (CV ~ 0.002).
- Normal sanction lags (47 to 63 days) and realistic amounts (Rs. 15.5L to 21L)
  so no individual work edge looks anomalous on its own.

Outputs:
- data/works_pattern_d.csv (10 rows)
- data/works_with_pattern_d.csv (15,010 rows: 15,000 real + 10 Pattern D)
"""

import os
import pandas as pd
import numpy as np

def generate_pattern_d():
    real_csv_path = "data/works_sanctioned_clean.csv"
    if not os.path.exists(real_csv_path):
        raise FileNotFoundError(f"Missing clean dataset at {real_csv_path}")

    df_real = pd.read_csv(real_csv_path)
    print(f"Loaded real dataset: {len(df_real)} rows.")

    base_idx = 20030
    ida_target = "AJMER(DISTRICT COLLECTOR AJMER_IDA)"
    state_target = "Rajasthan"

    mps = [
        "Bhagirath Chaudhary (2024-2029)",
        "Shri Hanuman Beniwal (2024-2029)",
        "BRIJENDRA SINGH OLA (2024-2029)"
    ]

    # 10 amounts with CV ~ 0.0988 (within [0.08, 0.15])
    amounts_d = [
        1550000.0, 1980000.0, 1720000.0, 2100000.0, 1680000.0,
        1920000.0, 1600000.0, 2050000.0, 1750000.0, 1850000.0
    ]

    # Sanction dates spread over 70 days (March 1, 2025 to May 10, 2025)
    dates_sanc_d = [
        "2025-03-01", "2025-03-08", "2025-03-18", "2025-03-27", "2025-04-05",
        "2025-04-14", "2025-04-22", "2025-04-30", "2025-05-05", "2025-05-10"
    ]

    # Normal realistic sanction lags (47 to 63 days)
    lags_d = [52, 48, 61, 55, 47, 59, 63, 50, 56, 49]

    # Multi-MP assignment across the 10 works
    mp_assignment_d = [
        mps[0], mps[0], mps[1], mps[1],
        mps[0], mps[2], mps[2], mps[0],
        mps[1], mps[2]
    ]

    descriptions = [
        "Rural connectivity road upgrading package ward 12",
        "Community development centre hall construction block 3",
        "Installation of high mast solar light units sector 5",
        "Drinking water pipeline and pump shed modernization",
        "Gram panchayat public utility shed construction",
        "Primary school boundary wall and sanitation block upgrade",
        "Drainage culvert and pavement renovation work phase 2",
        "Multipurpose community shed construction sector 9",
        "Approach road metalling and paver installation ward 7",
        "Installation of integrated solar street illumination system"
    ]

    synthetic_records = []
    for i in range(10):
        sanc_dt = pd.to_datetime(dates_sanc_d[i])
        rec_dt = sanc_dt - pd.Timedelta(days=lags_d[i])
        rec_str = rec_dt.strftime("%Y-%m-%d")

        synthetic_records.append({
            "work_index": base_idx + i,
            "Sr. No.": base_idx + i + 1,
            "work_id": f"WS/SYNTH-D{i+1:02d}",
            "Work category": "Normal/Others",
            "work_category_code": 1,
            "State": state_target,
            "IDA": ida_target,
            "MP": mp_assignment_d[i],
            "Elected/Nominated": "Elected MP",
            "Work description": descriptions[i],
            "Recommended date": rec_str,
            "Sanction Date": dates_sanc_d[i],
            "sanction_lag_days": lags_d[i],
            "sanction_amount": amounts_d[i],
            "Work Status": "Sanction",
            "work_status_code": 1,
            "is_synthetic_fraud": True,
            "fraud_pattern": "Pattern D: Slower-moving collusion with loose amounts"
        })

    df_synth_d = pd.DataFrame(synthetic_records)

    # Verification stats
    mean_a = df_synth_d["sanction_amount"].mean()
    std_a = df_synth_d["sanction_amount"].std(ddof=0)
    cv_a = std_a / mean_a
    d_start = pd.to_datetime(df_synth_d["Sanction Date"].min())
    d_end = pd.to_datetime(df_synth_d["Sanction Date"].max())
    span_days = (d_end - d_start).days

    print(f"\nPattern D Verification:")
    print(f"  Total works:       {len(df_synth_d)}")
    print(f"  Distinct MPs:      {df_synth_d['MP'].nunique()}")
    print(f"  Target IDA:        {df_synth_d['IDA'].iloc[0]}")
    print(f"  Time Span:         {span_days} days ({df_synth_d['Sanction Date'].min()} to {df_synth_d['Sanction Date'].max()})")
    print(f"  Mean Amount:       Rs. {mean_a:,.2f}")
    print(f"  Std Amount:        Rs. {std_a:,.2f}")
    print(f"  CV of Amounts:     {cv_a:.4f} (target: 0.08 - 0.15)")
    print(f"  Lags Min/Max/Mean: {df_synth_d['sanction_lag_days'].min()} / {df_synth_d['sanction_lag_days'].max()} / {df_synth_d['sanction_lag_days'].mean():.1f} days")

    # Combine with real dataset
    df_real_copy = df_real.copy()
    if "is_synthetic_fraud" not in df_real_copy.columns:
        df_real_copy["is_synthetic_fraud"] = False
    if "fraud_pattern" not in df_real_copy.columns:
        df_real_copy["fraud_pattern"] = "Real"
    if "work_index" not in df_real_copy.columns:
        df_real_copy["work_index"] = df_real_copy.index

    # Align columns
    cols = [
        "Sr. No.", "work_id", "Work category", "work_category_code",
        "State", "IDA", "MP", "Elected/Nominated", "Work description",
        "Recommended date", "Sanction Date", "sanction_lag_days",
        "sanction_amount", "Work Status", "work_status_code",
        "work_index", "is_synthetic_fraud", "fraud_pattern"
    ]
    df_real_copy = df_real_copy[cols]
    df_synth_d = df_synth_d[cols]

    df_combined = pd.concat([df_real_copy, df_synth_d], ignore_index=True)

    # Save
    out_pattern_d = "data/works_pattern_d.csv"
    out_combined = "data/works_with_pattern_d.csv"

    df_synth_d.to_csv(out_pattern_d, index=False)
    df_combined.to_csv(out_combined, index=False)
    print(f"\nSaved 10 Pattern D works to {out_pattern_d}")
    print(f"Saved combined dataset ({len(df_combined)} rows) to {out_combined}")

    return df_synth_d, df_combined

if __name__ == "__main__":
    generate_pattern_d()
