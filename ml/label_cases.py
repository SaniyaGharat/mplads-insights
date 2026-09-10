import numpy as np
import pandas as pd
import os

def run_labeling_cli():
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

    # Get top 20 anomalies
    top_20_idx = np.argsort(scores)[-20:][::-1]

    labels = []

    print("\n" + "="*60)
    print("MPLADS Anomaly Labeling CLI")
    print("="*60)
    print("Instructions:")
    print("  's'   -> Suspicious")
    print("  'n'   -> Normal")
    print("  'skip'-> Skip this case")
    print("  'q'   -> Quit and save")
    print("="*60 + "\n")

    try:
        for i, idx in enumerate(top_20_idx):
            row = df_works.iloc[idx]
            score = scores[idx]

            print(f"Case {i+1}/20 | Work Index: {idx} | GAT Score: {score:.4f}")
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
                    labels.append(('skip', idx))
                    # We need to break out of the outer loop too
                    # Use a flag or raise a custom exception
                    raise StopIteration
                else:
                    print("Invalid input. Please use 's', 'n', 'skip', or 'q'.")

            labels.append((choice, idx))
            print(f"Recorded: {choice}\n")

    except StopIteration:
        pass

    # Save labels
    if labels:
        # If we quit early, we might have more works to process,
        # but we only save what we have.
        label_df = pd.DataFrame(labels, columns=['label', 'work_index'])

        # If labels file exists, we might want to append or overwrite.
        # For Milestone 6, we want a clean set of expert labels.
        label_df.to_csv(labels_path, index=False)
        print(f"\nLabels saved to {labels_path}")
        print(f"Total labeled: {len(label_df)}")
    else:
        print("\nNo labels recorded.")

if __name__ == "__main__":
    run_labeling_cli()
