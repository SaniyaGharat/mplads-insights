import numpy as np
import pandas as pd
import os
import random

def run_negative_labeling_cli():
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

    # Sample 15 works from the 40th to 60th percentile of GAT scores
    p40 = np.percentile(scores, 40)
    p60 = np.percentile(scores, 60)

    # Indices within the [40th, 60th] percentile range
    mask = (scores >= p40) & (scores <= p60)
    eligible_idx = np.where(mask)[0]

    if len(eligible_idx) < 15:
        print(f"Error: Only found {len(eligible_idx)} cases in the target percentile range. Need 15.")
        return

    # Randomly sample 15 indices
    neg_idx = random.sample(list(eligible_idx), 15)

    labels = []

    print("\n" + "="*60)
    print("MPLADS Negative (Normal) Case Labeling CLI")
    print("="*60)
    print(f"Sampling from GAT score range: {p40:.4f} to {p60:.4f} (40th-60th percentile)")
    print("Instructions:")
    print("  's'   -> Suspicious")
    print("  'n'   -> Normal")
    print("  'skip'-> Skip this case")
    print("  'q'   -> Quit and save")
    print("="*60 + "\n")

    try:
        for i, idx in enumerate(neg_idx):
            row = df_works.iloc[idx]
            score = scores[idx]

            print(f"Case {i+1}/15 | Work Index: {idx} | GAT Score: {score:.4f}")
            print(f"MP: {row['MP']:<20} | IDA: {row['IDA']:<20}")
            print(f"Amount: {row['sanction_amount']:<12.2f} | Lag: {int(row['sanction_lag_days']):<8} | Status: {row['work_status_code']}")
            print("-" * 40)

            while True:
                choice = input("Label (s/n/skip/q): ").strip().lower()
                if choice in ['s', 'n', 'skip']:
                    break
                elif choice == 'q':
                    print("Quitting...")
                    # Handle remaining cases as 'skip'
                    # Since we are using a loop, we'll just stop collecting labels
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

        # Header should only be written if the file does not exist yet
        write_header = not os.path.exists(labels_path)

        # mode='a' ensures we append to the file
        label_df.to_csv(labels_path, mode='a', index=False, header=write_header)
        print(f"\nLabels APPENDED to {labels_path}")
        print(f"New labels added: {len(label_df)}")
    else:
        print("\nNo labels recorded.")

if __name__ == "__main__":
    run_negative_labeling_cli()
