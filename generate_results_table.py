"""
generate_results_table.py — Aggregates the per-model results written by
train.py into the comparison + ablation + statistical-significance
tables requested by Reviewer 3 (Comments 4 and 5).

Run AFTER training every model, e.g.:
    python -m src.train --model all
    python -m src.generate_results_table

Writes:
    docs/results_table.md         — mean +/- std for every metric, every model
    docs/ablation_table.md        — ResNet18-standalone vs Hybrid ResNet18+GCN
    docs/significance_table.md    — paired Wilcoxon / t-test, hybrid vs each baseline
    results/comparison_<metric>.png — bar chart, one per key metric
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import paired_significance_test
from src.utils import load_config
from src.visualize import plot_model_comparison_bar

KEY_METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
               "sensitivity_macro", "specificity_macro"]


def _load_model_results(model_name: str):
    path = Path("results") / model_name / "all_fold_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def build_results_table(model_names):
    rows = []
    per_model_scores = {}
    for name in model_names:
        fold_metrics = _load_model_results(name)
        if fold_metrics is None:
            print(f"[skip] no results found for '{name}' — run train.py first.")
            continue
        per_model_scores[name] = fold_metrics
        row = {"Model": name}
        for metric in KEY_METRICS:
            vals = np.array([m[metric] for m in fold_metrics])
            row[metric] = f"{vals.mean():.4f} +/- {vals.std(ddof=1):.4f}"
        rows.append(row)
    return pd.DataFrame(rows), per_model_scores


def build_ablation_table(per_model_scores):
    """ResNet18-standalone (no GCN) vs the proposed Hybrid ResNet18+GCN —
    isolates the graph module's contribution, as requested in Comment 4."""
    if "resnet18_standalone" not in per_model_scores or "hybrid_resnet18_gcn" not in per_model_scores:
        return None
    rows = []
    for metric in KEY_METRICS:
        base = np.array([m[metric] for m in per_model_scores["resnet18_standalone"]])
        hybrid = np.array([m[metric] for m in per_model_scores["hybrid_resnet18_gcn"]])
        n = min(len(base), len(hybrid))
        sig = paired_significance_test(hybrid[:n], base[:n])
        rows.append({
            "Metric": metric,
            "ResNet18 (no GCN)": f"{base.mean():.4f} +/- {base.std(ddof=1):.4f}",
            "Hybrid ResNet18+GCN": f"{hybrid.mean():.4f} +/- {hybrid.std(ddof=1):.4f}",
            "Delta": f"{hybrid.mean() - base.mean():+.4f}",
            "Wilcoxon p": f"{sig['wilcoxon_p']:.4g}",
        })
    return pd.DataFrame(rows)


def build_significance_table(per_model_scores, proposed="hybrid_resnet18_gcn"):
    if proposed not in per_model_scores:
        return None
    rows = []
    proposed_acc = np.array([m["accuracy"] for m in per_model_scores[proposed]])
    for name, scores in per_model_scores.items():
        if name == proposed:
            continue
        acc = np.array([m["accuracy"] for m in scores])
        n = min(len(acc), len(proposed_acc))
        sig = paired_significance_test(proposed_acc[:n], acc[:n])
        rows.append({
            "Baseline": name,
            "Paired t-test p (accuracy)": f"{sig['paired_ttest_p']:.4g}",
            "Wilcoxon p (accuracy)": f"{sig['wilcoxon_p']:.4g}",
        })
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    model_names = cfg["models_compared"]

    Path("docs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    results_df, per_model_scores = build_results_table(model_names)
    if results_df.empty:
        print("No trained models found under results/. Run src/train.py first.")
        return

    with open("docs/results_table.md", "w") as f:
        f.write("# Comparative Results — mean +/- std across all folds/repeats\n\n")
        f.write("(Reviewer 3, Comment 4 & 5)\n\n")
        f.write(results_df.to_markdown(index=False))
    print(results_df.to_string(index=False))

    ablation_df = build_ablation_table(per_model_scores)
    if ablation_df is not None:
        with open("docs/ablation_table.md", "w") as f:
            f.write("# Ablation: contribution of the GCN module\n\n")
            f.write(ablation_df.to_markdown(index=False))
        print("\n--- Ablation ---")
        print(ablation_df.to_string(index=False))

    sig_df = build_significance_table(per_model_scores)
    if sig_df is not None:
        with open("docs/significance_table.md", "w") as f:
            f.write("# Statistical significance — proposed model vs each baseline\n\n")
            f.write(sig_df.to_markdown(index=False))
        print("\n--- Significance ---")
        print(sig_df.to_string(index=False))

    for metric in KEY_METRICS:
        bar_data = {}
        for name, scores in per_model_scores.items():
            vals = np.array([m[metric] for m in scores])
            bar_data[name] = {"mean": vals.mean(), "std": vals.std(ddof=1)}
        plot_model_comparison_bar(
            bar_data, metric, f"results/comparison_{metric}.png",
            f"Model comparison — {metric.replace('_', ' ')}"
        )

    print("\nWrote docs/results_table.md, docs/ablation_table.md, docs/significance_table.md")


if __name__ == "__main__":
    main()
