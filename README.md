# Hybrid ResNet18–GCN for Multi-Stage Alzheimer's Disease Classification (MRI)

Reference implementation of the hybrid ResNet18 + Graph Convolutional
Network (GCN) framework for classifying four Alzheimer's disease stages
(Non-Demented, Very Mild Demented, Mild Demented, Moderate Demented)
from MRI, plus every baseline and ablation used to answer the reviewer
comments in the current revision round.

## Why this repository exists

Reviewer 3 and Reviewer 2 raised nine and two comments respectively,
covering: an unclear dataset table, an ambiguous graph-node definition,
possible data leakage, a superficial ablation study, insufficient
statistical validation, unclear class-imbalance handling, numerical
inconsistencies (epoch counts, malformed percentages), low-quality
figures, and weakly related citations. This codebase is the concrete,
runnable answer to every one of the *methodological* comments — see
[`docs/reviewer_response_mapping.md`](docs/reviewer_response_mapping.md)
for a comment-by-comment map to the exact file/function that resolves it.

**Important:** running this code on your real ADNI export is what
produces trustworthy numbers for the revised manuscript. Nothing in this
repository ships pre-baked results — every table and figure under
`docs/` and `results/` is generated fresh from your data, which is what
makes it defensible to the reviewers in the first place.

## Repository layout

```
AD-ResNet18-GCN/
├── README.md                 # this file
├── config.yaml                # single source of truth for every hyperparameter
├── requirements.txt
├── CODE_AVAILABILITY.md       # how to get a Zenodo DOI once code stabilizes (editor requirement)
├── .gitignore
├── src/                       # all code — see per-file docstrings for reviewer-comment mapping
│   ├── utils.py                    # config loading, seeding, device
│   ├── dataset.py                  # subject-wise loading & k-fold splitting (leakage-safe)
│   ├── preprocessing.py            # transforms, class weights, SMOTE, distribution logging
│   ├── backbones.py                # ResNet18 extractor + all baseline CNNs
│   ├── graph_utils.py              # k-NN graph construction & normalization (per-fold, not global)
│   ├── gcn_model.py                # GCN layers + Hybrid ResNet18+GCN model
│   ├── metrics.py                  # full metrics, bootstrap CIs, paired significance tests
│   ├── visualize.py                # confusion matrices, ROC curves, training curves (300 DPI)
│   ├── train.py                    # fold-by-fold training entry point for every model
│   ├── generate_dataset_table.py   # -> docs/dataset_table.md
│   ├── generate_results_table.py   # -> docs/results_table.md, ablation_table.md, significance_table.md
│   └── run_all.py                  # one-command full pipeline
└── docs/
    ├── reviewer_response_mapping.md   # comment -> code map, for the response letter
    ├── dataset_table.md               # generated
    ├── results_table.md               # generated
    ├── ablation_table.md              # generated
    └── significance_table.md          # generated
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data layout expected

```
data/ADNI/
    NonDemented/S0001_scan01.png
    NonDemented/S0001_scan02.png
    VeryMildDemented/S0002_scan01.png
    MildDemented/...
    ModerateDemented/...
```

Filenames must start with a subject identifier followed by an
underscore (`<subject_id>_<anything>.png`) — this is how the leakage-safe
split in `src/dataset.py` knows which images belong to the same patient.
Edit `extract_subject_id()` in `src/dataset.py` if your ADNI export uses
a different naming convention (e.g. DICOM metadata instead of filenames).

Edit `config.yaml::data.root_dir` if your data doesn't live at `data/ADNI`.

## Running the full study

```bash
python -m src.run_all
```

This trains every model in `config.yaml::models_compared`
(GoogleNet, AlexNet, DenseNet121, EfficientNet-B0, standalone ResNet18,
and the proposed Hybrid ResNet18+GCN), each with 5-fold subject-wise
cross-validation repeated 3 times, and writes:

- `docs/dataset_table.md` — subjects/images per class + split strategy
- `results/<model>/summary_metrics.json` — mean ± std per metric
- `docs/results_table.md` — all models side by side
- `docs/ablation_table.md` — ResNet18-only vs Hybrid ResNet18+GCN, with a Wilcoxon p-value
- `docs/significance_table.md` — proposed model vs every baseline, paired t-test + Wilcoxon
- `results/<model>/repeat*_fold*/confusion_matrix.png`, `roc_curve.png`, `training_curve.png`

For a fast smoke test before committing to the full multi-day run:

```bash
python -m src.run_all --quick
```

To train/evaluate a single model:

```bash
python -m src.train --model hybrid_resnet18_gcn
python -m src.train --model resnet18_standalone   # the ablation baseline
```

## Design decisions that map directly to reviewer comments

- **Graph nodes = MRI images**, not feature dimensions (`src/graph_utils.py`).
- **The k-NN graph is rebuilt inside every fold** from that fold's
  training embeddings only; test images are attached as query nodes at
  inference time and never influence the training graph's edges
  (`build_knn_graph`, `attach_query_nodes`).
- **Splitting is subject-wise everywhere** (`GroupKFold` on a parsed
  `subject_id`), with a runtime assertion that no subject crosses the
  train/test boundary — this is not just documented, it fails loudly if
  violated.
- **Imbalance handling has one fixed order**, used nowhere else in the
  codebase: split → train-only augmentation → train-only SMOTE (on
  embeddings) → train-only class weights in the loss.
- **`config.yaml` is the only place hyperparameters live** (300 epochs
  everywhere), so the manuscript's epoch/accuracy inconsistencies cannot
  recur.

## Honesty note on results

The original manuscript reports 96.56% test accuracy for the hybrid
model. This repository does not hardcode that number anywhere — it will
only reappear if your real ADNI data, run through this exact pipeline,
reproduces it. If your numbers differ, report what the pipeline actually
produces; that is what the reviewers are asking for.

## License

No license file is included. Until one is added, default copyright
applies and the code is not licensed for reuse by others — add a
license (e.g. MIT, Apache-2.0) when you are ready to make that decision.
