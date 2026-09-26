"""
ml/inject_synthetic_fraud.py
============================
Generates 30 synthetic fraud work records across 3 realistic fraud typologies
for pipeline validation without touching real artifacts.

Typologies:
- Pattern A (10 works): 'Shell contractor cluster' -- 3 real MPs channel 10 works
  to the same IDA within a tight 3-day window, with repetitive, near-identical amounts
  (structured just below ₹25 Lakhs threshold).
- Pattern B (10 works): 'Fast-tracked large sanction' -- High-value works (top 5%
  amounts, ₹2.2M - ₹4.1M) sanctioned with abnormally fast turnaround (1-3 days vs.
  90-day median).
- Pattern C (10 works): 'Ghost work' -- Standard-amount works stuck in 'Physical Inspection'
  or 'Vendor Identification' for elevated lag (650 - 690 days, at the top 99th percentile
  edge of observed real data, where real stuck max is 693 days).

Outputs:
- data/works_synthetic_labeled.csv (30 rows, is_synthetic_fraud=True)
- data/works_with_synthetic.csv    (15,030 rows: 15,000 real + 30 synthetic)
"""

import os
import pandas as pd
import numpy as np

def generate_synthetic_fraud():
    # 1. Load real clean data (read-only)
    real_csv_path = "data/works_sanctioned_clean.csv"
    if not os.path.exists(real_csv_path):
        raise FileNotFoundError(f"Expected clean dataset at {real_csv_path}")

    df_real = pd.read_csv(real_csv_path)
    print(f"Loaded real dataset: {len(df_real)} rows.")

    synthetic_records = []
    base_idx = 20000

    # ---------------------------------------------------------
    # Pattern A: Shell contractor cluster (10 works: indices 20000 - 20009)
    # ---------------------------------------------------------
    # 3 MPs channeling funds to the same IDA in Solapur, Maharashtra
    # Amounts tightly clustered around ₹2,470,000 - ₹2,485,000 (just under ₹25L threshold)
    # Window: Recommended Jan 15-17, 2025, Sanctioned Feb 1-3, 2025 (lag: 17 days)
    mps_cluster = [
        "PRANITI SUSHILKUMAR SHINDE (2024-2029)",
        "Omprakash Bhupalsinh Alias Pawan Rajenimbalkar (2024-2029)",
        "MOHITE PATIL DHAIRYASHEEL RAJSINH (2024-2029)"
    ]
    ida_cluster = "SOLAPUR(DISTRICT COLLECTOR SOLAPUR_IDA)"
    state_cluster = "Maharashtra"

    amounts_cluster = [
        2475000.0, 2480000.0, 2470000.0, 2485000.0,
        2475000.0, 2480000.0, 2472000.0, 2485000.0,
        2478000.0, 2482000.0
    ]
    dates_rec_cluster = [
        "2025-01-15", "2025-01-15", "2025-01-16", "2025-01-16",
        "2025-01-16", "2025-01-17", "2025-01-17", "2025-01-17",
        "2025-01-17", "2025-01-17"
    ]
    dates_sanc_cluster = [
        "2025-02-01", "2025-02-01", "2025-02-02", "2025-02-02",
        "2025-02-02", "2025-02-03", "2025-02-03", "2025-02-03",
        "2025-02-03", "2025-02-03"
    ]
    mp_assignment_cluster = [
        mps_cluster[0], mps_cluster[0], mps_cluster[0], mps_cluster[0],
        mps_cluster[1], mps_cluster[1], mps_cluster[1],
        mps_cluster[2], mps_cluster[2], mps_cluster[2]
    ]

    for i in range(10):
        rec_dt = pd.to_datetime(dates_rec_cluster[i])
        sanc_dt = pd.to_datetime(dates_sanc_cluster[i])
        lag = (sanc_dt - rec_dt).days
        synthetic_records.append({
            "work_index": base_idx + i,
            "Sr. No.": base_idx + i + 1,
            "work_id": f"WS/SYNTH-A{i+1:02d}",
            "Work category": "Normal/Others",
            "work_category_code": 1,
            "State": state_cluster,
            "IDA": ida_cluster,
            "MP": mp_assignment_cluster[i],
            "Elected/Nominated": "Elected MP",
            "Work description": f"Paver block road and community hall package #{i+1} ward 4/5/6",
            "Recommended date": dates_rec_cluster[i],
            "Sanction Date": dates_sanc_cluster[i],
            "sanction_lag_days": lag,
            "sanction_amount": amounts_cluster[i],
            "Work Status": "Sanction",
            "work_status_code": 1,
            "is_synthetic_fraud": True,
            "fraud_pattern": "Pattern A: Shell contractor cluster"
        })

    # ---------------------------------------------------------
    # Pattern B: Fast-tracked large sanction (10 works: indices 20010 - 20019)
    # ---------------------------------------------------------
    # Large sanction amount in top 5% (₹2.2M to ₹4.2M)
    # Artificially compressed sanction lag (1 to 3 days vs median 90 days)
    # Real MP-IDA pairs across multiple states to avoid spatial clustering
    pairs_pattern_b = [
        ("RAVINDRA DATTARAM WAIKAR (2024-2029)", "MUMBAI SUBURBAN(DISTRICT COLLECTOR MUMBAI SUBURBAN_IDA)", "Maharashtra", "Elected MP"),
        ("SANJNA JATAV (2024-2029)", "ALWAR(DISTRICT COLLECTOR ALWAR_IDA)", "Rajasthan", "Elected MP"),
        ("Prabhubhai Nagarbhai Vasava (2024-2029)", "TAPI(District collector Tapi_IDA)", "Gujarat", "Elected MP"),
        ("ABHAY KUMAR SINHA (2024-2029)", "AURANGABAD(DISTRICT PLANNING OFFICER AURANGABAD_IDA)", "Bihar", "Elected MP"),
        ("RAHUL SINGH LODHI (2024-2029)", "DAMOH(DISTRICT COLLECTOR DAMOH_IDA)", "Madhya Pradesh", "Elected MP"),
        ("Shri B Y Raghavendra  (2024-2029)", "SHIVAMOGGA(DEPUTY COMMISSIONER SHIMOGA_IDA)", "Karnataka", "Elected MP"),
        ("K RADHAKRISHNAN (2024-2029)", "THRISSUR(DISTRICT COLLECTOR THRISSUR_IDA)", "Kerala", "Elected MP"),
        ("BASTIPATI NAGARAJU PANCHALINGALA (2024-2029)", "KURNOOL(DISTRICT COLLECTOR KURNOOL_IDA)", "Andhra Pradesh", "Elected MP"),
        ("MUKESHKUMAR CHANDRAKAANT DALAL (2024-2029)", "SURAT(DISTRICT COLLECTOR SURAT_IDA)", "Gujarat", "Elected MP"),
        ("MADHAVANENI RAGHUNANDAN RAO (2024-2029)", "MEDAK(COLLECTOR MEDAK DISTRICT)", "Telangana", "Elected MP"),
    ]

    amounts_b = [2850000.0, 3400000.0, 2250000.0, 4100000.0, 2600000.0,
                 3750000.0, 2900000.0, 3150000.0, 2400000.0, 3950000.0]
    lags_b = [1, 2, 1, 3, 2, 1, 2, 3, 1, 2]
    dates_rec_b = [
        "2025-03-03", "2025-03-05", "2025-03-10", "2025-03-12", "2025-03-15",
        "2025-03-18", "2025-03-20", "2025-03-24", "2025-03-27", "2025-03-29"
    ]

    for i in range(10):
        mp_b, ida_b, state_b, elect_b = pairs_pattern_b[i]
        rec_dt = pd.to_datetime(dates_rec_b[i])
        sanc_dt = rec_dt + pd.Timedelta(days=lags_b[i])
        synthetic_records.append({
            "work_index": base_idx + 10 + i,
            "Sr. No.": base_idx + 10 + i + 1,
            "work_id": f"WS/SYNTH-B{i+1:02d}",
            "Work category": "Normal/Others",
            "work_category_code": 1,
            "State": state_b,
            "IDA": ida_b,
            "MP": mp_b,
            "Elected/Nominated": elect_b,
            "Work description": f"High priority infrastructure development scheme #{i+1}",
            "Recommended date": rec_dt.strftime("%Y-%m-%d"),
            "Sanction Date": sanc_dt.strftime("%Y-%m-%d"),
            "sanction_lag_days": lags_b[i],
            "sanction_amount": amounts_b[i],
            "Work Status": "Sanction",
            "work_status_code": 1,
            "is_synthetic_fraud": True,
            "fraud_pattern": "Pattern B: Fast-tracked large sanction"
        })

    # ---------------------------------------------------------
    # Pattern C: Ghost work (10 works: indices 20020 - 20029)
    # ---------------------------------------------------------
    # Normal amounts (₹350K - ₹950K, around median), but stuck in early stage
    # ("Physical Inspection" or "Vendor Identification") for 910 - 1040 days
    # (real dataset max stuck lag is 693 days)
    pairs_pattern_c = [
        ("RANJIT DUTTA (2024-2029)", "LAKHIMPUR(DEPUTY COMMISSIONER LAKHIMPUR_IDA)", "Assam", "Elected MP"),
        ("AMARSING TISSO (2024-2029)", "KARBI ANGLONG(DEPUTY COMMISSIONER KARBI ANGLONG_IDA)", "Assam", "Elected MP"),
        ("Faggan Singh Kulaste (2024-2029)", "MANDLA(DISTRICT COLLECTOR MANDLA_IDA)", "Madhya Pradesh", "Elected MP"),
        ("ARUN BHARTI (2024-2029)", "SHEIKHPURA(DISTRICT PLANNING OFFICER SHEIKHPURA_IDA)", "Bihar", "Elected MP"),
        ("DR.K.SUDHAKAR (2024-2029)", "Bengaluru South(DEPUTY COMMISSIONER RAMANAGARA_IDA)", "Karnataka", "Elected MP"),
        ("LUMBARAM (2024-2029)", "SIROHI(DISTRICT COLLECTOR SIROHI_IDA)", "Rajasthan", "Elected MP"),
        ("Pradeep Kumar Singh (2024-2029)", "ARARIA(DISTRICT PLANNING OFFICER ARARIA_IDA)", "Bihar", "Elected MP"),
        ("MAHESH KASHYAP (2024-2029)", "SUKMA(Collector Sukma_IDA)", "Chhattisgarh", "Elected MP"),
        ("RADHE SHYAM RATHIYA (2024-2029)", "JASHPUR(DISTRICT COLLECTOR JASHPUR_IDA)", "Chhattisgarh", "Elected MP"),
        ("ARUN BHARTI (2024-2029)", "MUNGER(DISTRICT PLANNING OFFICER MUNGER_IDA)", "Bihar", "Elected MP"),
    ]

    amounts_c = [450000.0, 600000.0, 380000.0, 750000.0, 520000.0,
                 890000.0, 350000.0, 480000.0, 620000.0, 950000.0]
    # Lags adjusted to 650-695 days: elevated at the 99th percentile / tail,
    # but strictly within the observed training distribution (real stuck max is 693, overall max is 732)
    lags_c = [652, 660, 668, 675, 682, 655, 664, 671, 685, 690]
    dates_rec_c = [
        "2023-08-15", "2023-08-20", "2023-08-25", "2023-09-01", "2023-09-05",
        "2023-08-18", "2023-08-22", "2023-08-28", "2023-09-02", "2023-09-08"
    ]
    # 5 Physical Inspection (code 0) and 5 Vendor Identification (code 3)
    statuses_c = [
        ("Physical Inspection", 0), ("Physical Inspection", 0),
        ("Physical Inspection", 0), ("Physical Inspection", 0),
        ("Physical Inspection", 0), ("Vendor Identification", 3),
        ("Vendor Identification", 3), ("Vendor Identification", 3),
        ("Vendor Identification", 3), ("Vendor Identification", 3)
    ]

    for i in range(10):
        mp_c, ida_c, state_c, elect_c = pairs_pattern_c[i]
        rec_dt = pd.to_datetime(dates_rec_c[i])
        sanc_dt = rec_dt + pd.Timedelta(days=lags_c[i])
        st_name, st_code = statuses_c[i]
        synthetic_records.append({
            "work_index": base_idx + 20 + i,
            "Sr. No.": base_idx + 20 + i + 1,
            "work_id": f"WS/SYNTH-C{i+1:02d}",
            "Work category": "Normal/Others",
            "work_category_code": 1,
            "State": state_c,
            "IDA": ida_c,
            "MP": mp_c,
            "Elected/Nominated": elect_c,
            "Work description": f"Preliminary stage municipal facility work #{i+1}",
            "Recommended date": rec_dt.strftime("%Y-%m-%d"),
            "Sanction Date": sanc_dt.strftime("%Y-%m-%d"),
            "sanction_lag_days": lags_c[i],
            "sanction_amount": amounts_c[i],
            "Work Status": st_name,
            "work_status_code": st_code,
            "is_synthetic_fraud": True,
            "fraud_pattern": "Pattern C: Ghost work"
        })

    df_synthetic = pd.DataFrame(synthetic_records)

    # 2. Save standalone synthetic labeled dataset
    synth_csv_path = "data/works_synthetic_labeled.csv"
    df_synthetic.to_csv(synth_csv_path, index=False)
    print(f"Saved {len(df_synthetic)} synthetic records to {synth_csv_path}")

    # 3. Create combined dataset (15,000 real + 30 synthetic)
    df_real_augmented = df_real.copy()
    df_real_augmented["work_index"] = df_real_augmented.index
    df_real_augmented["is_synthetic_fraud"] = False
    df_real_augmented["fraud_pattern"] = "None"

    # Align column order
    cols = list(df_real.columns) + ["work_index", "is_synthetic_fraud", "fraud_pattern"]
    df_combined = pd.concat([df_real_augmented[cols], df_synthetic[cols]], ignore_index=True)

    combined_csv_path = "data/works_with_synthetic.csv"
    df_combined.to_csv(combined_csv_path, index=False)
    print(f"Saved combined dataset ({len(df_combined)} rows) to {combined_csv_path}")

    # 4. Print detailed summary table
    print("\n" + "="*120)
    print(f"{'Idx':<6} | {'Pattern':<18} | {'Amount (INR)':<13} | {'Lag':<5} | {'Status (Code)':<22} | {'MP':<30} | {'IDA':<22}")
    print("="*120)
    for _, r in df_synthetic.iterrows():
        p_name = r['fraud_pattern'].split(':')[0]
        st_str = f"{r['Work Status']} ({r['work_status_code']})"
        mp_short = r['MP'][:28]
        ida_short = r['IDA'][:20]
        print(f"{r['work_index']:<6} | {p_name:<18} | {r['sanction_amount']:<13,.0f} | {r['sanction_lag_days']:<5} | {st_str:<22} | {mp_short:<30} | {ida_short:<22}")
    print("="*120 + "\n")

    return df_synthetic, df_combined

if __name__ == "__main__":
    generate_synthetic_fraud()
