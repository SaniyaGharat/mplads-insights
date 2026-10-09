# Synthetic Fraud Benchmark — Post-Freeze Notes & Evaluation Protocols

> **Benchmark Version**: 5.0 (Date-Matched Injection, Frozen)  
> **Evaluation Script**: `ml/evaluate_detectors.py`  
> **Scores Dataset**: `data/benchmark/detector_scores_v2.csv`  

---

## 1. Detectability Classification & Tabular Controls

- **Status of `detectability_class`**: The probe-based `detectability_class` column in earlier specifications is unreliable and unused.
- **Tabular Controls**: Typologies whose best single feature ROC-AUC exceeds both **0.85** and its **95th percentile null threshold** (from 200 pseudo-episode draws) are designated as **Tabular Controls**:
  - **Typology B** (Fast-Tracked Large Sanction): Best feature = `lag` (directional AUROC = 0.9610 vs Null p95 = 0.6982).
  - **Typology C** (Stalled Ghost Work): Best feature = `lag` (directional AUROC = 0.9634 vs Null p95 = 0.6716).
  - **Typology E** (Split Sanctions / Structuring): Best feature = `amount` (directional AUROC = 0.8868 vs Null p95 = 0.7532).
- **Structural Typologies**: Typologies A, D, F, G, and H have single-feature AUROC below 0.85 (or within empirical null p95 variation) and require higher-order graph/structural detection.

---

## 2. Zero-Label Detector Evaluation Protocol (Prompt 2a-Fix)

1. **Dataset Pooling & Split Reservation**:
   - Zero-label detectors and GAT models are evaluated on **ALL 88 episodes** (train, val, and test pooled) against all 15,000 real works.
   - The test split is strictly reserved for downstream learned models and is not used to make any detector selection or tuning decisions.
   - Unsupervised models (Isolation Forest, Robust Z-score) are fit on all 15,430 rows without access to labels.

2. **GAT Anomaly Scoring Variants**:
   - `gat_frozen`: Pre-trained GAT encoder evaluates node embeddings $z$ frozen from the 15,000-work real graph forward pass, and the reconstructor evaluates edge pairs $(u, v)$ for all 15,430 benchmark works. Standardized using parameters from the original 15,000 works. Achieves Spearman correlation $r_s = 1.000000$ with `ml/gat_scores.npy` on real works.
   - `gat_augmented`: Fresh forward pass on the full graph containing 15,430 forward + 15,430 reverse edges. Node feature sums and edge features are standardized using real 15,000 baseline parameters. Achieves Spearman correlation $r_s = 0.951503$ with `ml/gat_scores.npy` on real works.

3. **Directional Significance & Empirical Null**:
   - For each detector $\times$ typology, a null distribution is constructed from 1,000 draws of pseudo-episodes sampled from eligible real works ($\ge 15$ real works), matching the exact episode count and sizes of that typology.
   - One-sided upper-tail empirical p-values are computed for AUROC and AUPRC:
     $$p = \frac{\sum_{d=1}^{1000} \mathbb{I}(M_{null}^{(d)} \ge M_{actual}) + 1}{1000 + 1}$$
   - Holm-Bonferroni step-down adjustment is applied across the 8 typologies for each detector.
   - Typologies are marked 'detected' only if the adjusted p-value $< 0.05$.
   - Any detector $\times$ typology with AUROC below the null's 5th percentile is labeled *"looks more typical than real works"*.

4. **Operating Point Metrics**:
   - Base rate = $430 / 15430 = 0.027868$ (2.79%).
   - Lift = $\text{AUPRC} / \text{Base Rate}$.
   - Recall reported at top 1% ($k=154$), top 2% ($k=309$), and top 5% ($k=772$).

5. **Community Scorer Decomposition & Typology F Behavior**:
   - Typology A achieves 100% activation (`community_score > 0`) with AUROC = 0.9996 and AUPRC = 0.9782 among active works.
   - Typology F (New-Agency Burst) is solo-MP by design; it only triggers `community_score > 0` (4.92% of works) when other real MPs happen to have co-occurring works at the same IDA within the 30-day window.

6. **Sensitivity Analysis**:
   - Evaluation repeated with negatives restricted to 8,903 eligible real works (both MP and IDA $\ge 15$ real works).
   - Zero changes in 'detected' calls across all detectors and typologies, confirming robustness to entity capacity filters.
