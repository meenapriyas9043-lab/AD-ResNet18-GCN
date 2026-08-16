"""
run_all.py — One-command reproduction of the entire revision package.

    python -m src.run_all              # full pipeline: all models, all folds/repeats
    python -m src.run_all --quick       # fast sanity check: 1 model, 1 fold, few epochs

This is the single entry point a reviewer or the editor's reproducibility
checker would run after cloning the repository and pointing config.yaml
at a local copy of the ADNI export.
"""
import argparse

from src.generate_dataset_table import main as generate_dataset_table
from src.generate_results_table import main as generate_results_table
from src.train import run as train_model
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                         help="Run a fast 1-fold, few-epoch smoke test instead of the full study.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print("=== Step 1/3: dataset table (Reviewer 3, Comment 1) ===")
    generate_dataset_table()

    print("\n=== Step 2/3: training all models (Reviewer 3, Comments 3, 4, 6) ===")
    models = cfg["models_compared"] if not args.quick else [cfg["models_compared"][-1]]
    for name in models:
        train_model(name, args.config)

    print("\n=== Step 3/3: comparison + ablation + significance tables (Comments 4, 5) ===")
    generate_results_table()

    print("\nDone. See docs/*.md for manuscript-ready tables and results/ for figures.")


if __name__ == "__main__":
    main()
