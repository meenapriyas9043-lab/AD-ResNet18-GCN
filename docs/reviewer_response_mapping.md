# Reviewer comment → code mapping

Use this file directly while drafting the point-by-point response letter.
Every row names the exact file/function that resolves the comment, so the
response can cite the repository instead of re-arguing in prose.

| # | Reviewer | Comment (short) | Resolved by |
|---|----------|------------------|-------------|
| 1 | R3-C1 | Dataset table missing | `src/generate_dataset_table.py` → `docs/dataset_table.md`. Subjects/images per class computed from disk; age/sex/protocol columns are marked `TODO` for you to fill from ADNI clinical metadata (code cannot read that from pixels). |
| 2 | R3-C2 | Graph node definition ambiguous | `src/graph_utils.py` module docstring + `build_knn_graph`: nodes = MRI images (not feature dimensions), enforced by the function signature (`(N_images, 512) -> (N_images, N_images)`). Test-node incorporation implemented in `attach_query_nodes`. |
| 3 | R3-C3 / R2 | Data leakage risk / "global k-NN graph" | `src/dataset.py::subject_wise_kfold` (GroupKFold on subject_id, asserted disjoint) + `src/graph_utils.py` (graph rebuilt per fold from train-only embeddings; test nodes attached at inference only). |
| 4 | R3-C4 | Ablation not a real ablation | `config.yaml::models_compared` includes `resnet18_standalone` (no GCN) trained identically to the hybrid model in `src/train.py`; compared in `docs/ablation_table.md` via `src/generate_results_table.py::build_ablation_table`. |
| 5 | R3-C5 | Insufficient statistical validation | `src/metrics.py`: per-fold metrics, macro+micro averages, bootstrap CIs, paired Wilcoxon/t-test. Aggregated in `docs/results_table.md` and `docs/significance_table.md`. |
| 6 | R3-C6 | Class imbalance handling unclear | `src/preprocessing.py` docstring gives the exact order (split → augment(train) → SMOTE(train) → class weights(train)); `report_class_distribution` logs before/after counts for every run. |
| 7 | R3-C7 | Epoch/accuracy inconsistencies | `config.yaml::training.epochs = 300` is the single source of truth used by every model; all reported numbers are generated (not hand-typed) by `src/metrics.py`. |
| 8 | R3-C8 | Figure/ROC/confusion-matrix quality | `src/visualize.py`: 300 DPI, labeled axes/legends, per-class AUC in the ROC legend, annotated confusion-matrix cells. |
| 9 | R3-C9 / R2 | Weak/irrelevant references | Not a code fix — see the manuscript reference list. This repository does not add citations. |
| — | Editor | Code Availability / DOI requirement | See `CODE_AVAILABILITY.md` at the repository root. |
