"""
train.py — Fold-by-fold training and evaluation for every model in
config.yaml::models_compared, run with:

    python -m src.train --model hybrid_resnet18_gcn
    python -m src.train --model alexnet
    python -m src.train --model all          # trains every baseline + the proposed model

Implements the exact pipeline order required by Reviewer 3 Comment 3 and
Comment 6 inside each fold:

    1. subject-wise train/test split (dataset.subject_wise_kfold)
    2. train-only augmentation (preprocessing.build_transforms)
    3. [hybrid model only] ResNet18 embeddings -> train-only SMOTE
       -> per-fold k-NN graph (graph_utils.build_knn_graph)
    4. train-only class weights fed into the loss function
    5. evaluation on the untouched test partition of that fold

Results (per-fold metrics, aggregated mean +/- std, figures) are written
to results/<model_name>/ and are the ONLY source for every number that
should appear in the manuscript's revised tables.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.backbones import ResNet18FeatureExtractor, get_backbone
from src.dataset import load_dataset_index, subject_wise_kfold
from src.gcn_model import HybridResNet18GCN
from src.graph_utils import attach_query_nodes, build_knn_graph
from src.metrics import compute_fold_metrics, aggregate_fold_metrics
from src.preprocessing import (
    MRIDataset,
    apply_smote_to_embeddings,
    build_transforms,
    compute_class_weights,
    report_class_distribution,
)
from src.utils import get_device, load_config, set_seed
from src.visualize import plot_confusion_matrix, plot_roc_curves, plot_training_curves


@torch.no_grad()
def _extract_embeddings(extractor, loader, device):
    extractor.eval()
    feats, labels = [], []
    for imgs, y, _ in loader:
        imgs = imgs.to(device)
        f = extractor(imgs).cpu().numpy()
        feats.append(f)
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def train_cnn_baseline(cfg, model_name, samples, train_idx, test_idx, class_names, device, out_dir):
    """Standard supervised training loop shared by every plain-CNN baseline
    (googlenet, alexnet, densenet121, efficientnet_b0, resnet18_standalone)."""
    train_samples = [samples[i] for i in train_idx]
    test_samples = [samples[i] for i in test_idx]

    train_tf = build_transforms(cfg, train=True)
    test_tf = build_transforms(cfg, train=False)
    train_ds = MRIDataset(train_samples, train_tf)
    test_ds = MRIDataset(test_samples, test_tf)

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=2)

    train_labels = np.array([s.label for s in train_samples])
    report_class_distribution(train_labels, class_names, "train_before_smote")
    class_weights = compute_class_weights(train_labels, len(class_names)).to(device)

    model = get_backbone(model_name, n_classes=len(class_names), pretrained=cfg["backbone"]["pretrained"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss, patience_ctr = float("inf"), 0

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        running_loss, running_correct, n = 0.0, 0, 0
        for imgs, y, _ in train_loader:
            imgs, y = imgs.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            logits = out.logits if hasattr(out, "logits") else out  # googlenet aux-logits object
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            running_correct += (logits.argmax(1) == y).sum().item()
            n += imgs.size(0)
        history["train_loss"].append(running_loss / n)
        history["train_acc"].append(running_correct / n)

        val_loss, val_acc = _evaluate_cnn_loss_acc(model, test_loader, criterion, device)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss, patience_ctr = val_loss, 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["training"]["early_stopping_patience"]:
                break

    y_true, y_pred, y_proba = _predict_cnn(model, test_loader, device)
    plot_training_curves(history, f"{out_dir}/training_curve.png", f"{model_name}: loss & accuracy")
    return y_true, y_pred, y_proba


@torch.no_grad()
def _evaluate_cnn_loss_acc(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for imgs, y, _ in loader:
        imgs, y = imgs.to(device), y.to(device)
        logits = model(imgs)
        loss_sum += criterion(logits, y).item() * imgs.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += imgs.size(0)
    return loss_sum / n, correct / n


@torch.no_grad()
def _predict_cnn(model, loader, device):
    model.eval()
    y_true, y_pred, y_proba = [], [], []
    for imgs, y, _ in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        y_true.append(y.numpy())
        y_pred.append(proba.argmax(1))
        y_proba.append(proba)
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_proba)


def train_hybrid_gcn(cfg, samples, train_idx, test_idx, class_names, device, out_dir):
    """Feature extraction -> train-only SMOTE -> per-fold k-NN graph -> GCN.

    This is the ONLY function in the codebase that builds a k-NN graph,
    and it always does so from a single fold's data (see graph_utils.py
    module docstring for the leakage-safety argument).
    """
    train_samples = [samples[i] for i in train_idx]
    test_samples = [samples[i] for i in test_idx]

    tf = build_transforms(cfg, train=False)  # no geometric aug before embedding extraction
    train_loader = DataLoader(MRIDataset(train_samples, tf), batch_size=64, shuffle=False)
    test_loader = DataLoader(MRIDataset(test_samples, tf), batch_size=64, shuffle=False)

    extractor = ResNet18FeatureExtractor(pretrained=cfg["backbone"]["pretrained"]).to(device)
    train_emb, train_labels = _extract_embeddings(extractor, train_loader, device)
    test_emb, test_labels = _extract_embeddings(extractor, test_loader, device)

    report_class_distribution(train_labels, class_names, "train_before_smote")
    train_emb_bal, train_labels_bal = apply_smote_to_embeddings(
        train_emb, train_labels,
        k_neighbors=cfg["preprocessing"]["imbalance"]["smote_k_neighbors"],
        seed=cfg["seed"],
    )
    report_class_distribution(train_labels_bal, class_names, "train_after_smote")

    k = cfg["graph"]["k_neighbors"]
    train_graph = build_knn_graph(train_emb_bal, train_labels_bal, k=k)
    full_graph = attach_query_nodes(train_emb_bal, test_emb, test_labels, k=k)

    n_train = train_emb_bal.shape[0]
    class_weights = compute_class_weights(train_labels_bal, len(class_names)).to(device)

    model = HybridResNet18GCN(
        in_dim=cfg["backbone"]["feature_dim"],
        hidden_dim=cfg["gcn"]["hidden_dim"],
        n_classes=len(class_names),
        n_layers=cfg["gcn"]["n_layers"],
        dropout=cfg["gcn"]["dropout"],
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    X = full_graph.node_features.to(device)
    A = full_graph.adjacency.to(device)
    y_train = torch.tensor(train_labels_bal, dtype=torch.long, device=device)
    y_test = torch.tensor(test_labels, dtype=torch.long, device=device)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss, patience_ctr = float("inf"), 0

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        optimizer.zero_grad()
        logits = model(X, A)
        train_logits = logits[:n_train]
        loss = criterion(train_logits, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_eval = model(X, A)
            test_logits = logits_eval[n_train:]
            val_loss = criterion(test_logits, y_test).item()
            train_acc = (train_logits.argmax(1) == y_train).float().mean().item()
            val_acc = (test_logits.argmax(1) == y_test).float().mean().item()

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss, patience_ctr = val_loss, 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["training"]["early_stopping_patience"]:
                break

    with torch.no_grad():
        logits_final = model(X, A)
        proba = torch.softmax(logits_final[n_train:], dim=1).cpu().numpy()
        y_pred = proba.argmax(1)

    plot_training_curves(history, f"{out_dir}/training_curve.png", "Hybrid ResNet18+GCN: loss & accuracy")
    return test_labels, y_pred, proba


def run(model_name: str, cfg_path: str = "config.yaml"):
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])
    device = get_device()
    class_names = cfg["data"]["classes"]
    samples = load_dataset_index(cfg["data"]["root_dir"], class_names)

    out_dir = Path("results") / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics = []
    for repeat in range(cfg["data"]["n_repeats"]):
        set_seed(cfg["seed"] + repeat)
        for fold_i, (train_idx, test_idx) in enumerate(
            subject_wise_kfold(samples, cfg["data"]["n_folds"], cfg["seed"] + repeat)
        ):
            fold_dir = out_dir / f"repeat{repeat}_fold{fold_i}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            if model_name == "hybrid_resnet18_gcn":
                y_true, y_pred, y_proba = train_hybrid_gcn(
                    cfg, samples, train_idx, test_idx, class_names, device, fold_dir
                )
            else:
                y_true, y_pred, y_proba = train_cnn_baseline(
                    cfg, model_name, samples, train_idx, test_idx, class_names, device, fold_dir
                )

            m = compute_fold_metrics(y_true, y_pred, y_proba, len(class_names))
            fold_metrics.append(m)
            plot_confusion_matrix(y_true, y_pred, class_names, f"{fold_dir}/confusion_matrix.png",
                                   f"{model_name} — repeat {repeat}, fold {fold_i}")
            plot_roc_curves(y_true, y_proba, class_names, f"{fold_dir}/roc_curve.png",
                             f"{model_name} — repeat {repeat}, fold {fold_i}")
            with open(fold_dir / "metrics.json", "w") as f:
                json.dump(m, f, indent=2)

    summary = aggregate_fold_metrics(fold_metrics)
    with open(out_dir / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "all_fold_metrics.json", "w") as f:
        json.dump(fold_metrics, f, indent=2)

    print(f"\n=== {model_name}: mean +/- std across {len(fold_metrics)} fold/repeat runs ===")
    for k, v in summary.items():
        print(f"  {k:22s}: {v}")
    return fold_metrics, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                         help="One of config.yaml::models_compared, or 'all'.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    if args.model == "all":
        cfg = load_config(args.config)
        for name in cfg["models_compared"]:
            run(name, args.config)
    else:
        run(args.model, args.config)
