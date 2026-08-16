"""
visualize.py — Publication-quality figures.

Answers Reviewer 3 Comment 8: every figure here is saved at 300 DPI with
labeled axes, a legend, class names on confusion-matrix ticks, and
numeric annotations inside each confusion-matrix cell, so nothing is
left to be "not properly displayed".
"""
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(y_true, y_pred, class_names: List[str], out_path: str, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=True,
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_roc_curves(y_true, y_proba: np.ndarray, class_names: List[str], out_path: str, title: str):
    """One-vs-rest ROC curve per class, all four on a single labeled axis."""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(6, 5))
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_proba[:, c])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_names[c]} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_training_curves(history: dict, out_path: str, title: str):
    """history: {'train_loss': [...], 'val_loss': [...],
                 'train_acc': [...], 'val_acc': [...]}"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["train_loss"], label="Train loss")
    axes[0].plot(history["val_loss"], label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("(a) Loss over epochs")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train accuracy")
    axes[1].plot(history["val_acc"], label="Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("(b) Accuracy over epochs")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_model_comparison_bar(results: dict, metric_key: str, out_path: str, title: str):
    """results: {model_name: {'mean': float, 'std': float}, ...}"""
    names = list(results.keys())
    means = [results[n]["mean"] for n in names]
    stds = [results[n]["std"] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(names, means, yerr=stds, capsize=5, color=sns.color_palette("Blues_d", len(names)))
    ax.set_ylabel(metric_key.replace("_", " ").title())
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
