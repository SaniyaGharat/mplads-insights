import numpy as np
import pandas as pd
import os
import random

def run_midrange_labeling_cli():
    scores_path = 'ml/gat_scores.npy'
    works_path = 'data/works_sanctioned_clean.csv'
    labels_path = 'ml/anomaly_labels.csv'

    if not os.path.exists(scores_path):
        print(f"Error: {scores_path} not found. Please run 'python ml/gat_anomaly_scorer.py' first.")
        return

    if not os.path.exists(works_path):
        print(f"Error: {works_path} not found.")
        return

    # Load data
    scores = np.load(scores_path)
    df_works = pd.read_csv(works_path)

    # Get existing labeled indices to avoid duplicates
    existing_indices = set()
    if os.path.exists(labels_path):
        try:
            df_existing = pd.read_csv(labels_path)
            existing_indices = set(df_existing['work_index'].tolist())
        except Exception as e:
            print(f"Warning: Could not read existing labels: {e}")

    # Target: Works ranked 21st to 80th highest (index 20 to 80)
    sorted_idx = np.argsort(scores)[::-1]
    midrange_candidates = sorted_idx[20:80]

    # Filter out already labeled cases
    eligible_idx = [idx for idx in midrange_candidates if idx not in existing_indices]

    if len(eligible_idx) < 25:
        print(f"Error: Only found {len(eligible_idx)} eligible cases in the 21st-80th range. Need 25.")
        return

    # Randomly sample 25 indices from this mid-tier range
    sample_idx = random.sample(list(eligible_idx), 25)

    labels = []

    print("\n" + "="*60)
    print("MPLADS Mid-Range Case Labeling CLI (Rank 21-80)")
    print("="*60)
    print("Instructions:")
    print("  's'   -> Suspicious")
    print("  'n'   -> Normal")
    print("  'skip'-> Skip this case")
    print("  'q'   -> Quit and save")
    print("="*60 + "\n")

    try:
        for i, idx in enumerate(sample_idx):
            row = df_works.iloc[idx]
            score = scores[idx]

            print(f"Case {i+1}/25 | Work Index: {idx} | GAT Score: {score:.4f}")
            print(f"MP: {row['MP']:<20} | IDA: {row['IDA']:<20}")
            print(f"Amount: {row['sanction_amount']:<12.2f} | Lag: {int(row['sanction_lag_days']):<8} | Status: {row['work_status_code']}")
            print("-" * 40)

            while True:
                choice = input("Label (s/n/skip/q): ").strip().lower()
                if choice in ['s', 'n', 'skip']:
                    break
                elif choice == 'q':
                    print("Quitting...")
                    raise StopIteration
                else:
                    print("Invalid input. Please use 's', 'n', 'skip', or 'q'.")

            labels.append((choice, idx))
            print(f"Recorded: {choice}\n")

    except StopIteration:
        pass

    # Save labels by APPENDING to the existing CSV
    if labels:
        label_df = pd.DataFrame(labels, columns=['label', 'work_index'])

        # Header only if file doesn't exist
        write_header = not os.path.exists(labels_path)

        # Append mode
        label_df.to_csv(labels_path, mode='a', index=False, header=write_header)
        print(f"\nLabels APPENDED to {labels_path}")
        print(f"New labels added: {len(label_df)}")
    else:
        print("\nNo labels recorded.")

if __name__ == "__main__":
    run_midrange_labeling_cli()
