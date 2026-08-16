"""
preprocessing.py — Resizing, normalization, augmentation, class-imbalance
handling, and the torch Dataset wrapper.

Reviewer 3, Comment 6 asked for a precise, single account of how class
imbalance is handled and confirmation that it touches ONLY the training
partition. That pipeline lives entirely in this file:

    1. Images are resized/normalized identically for train and test.
    2. Geometric/color augmentation (train only) — see `build_transforms`.
    3. SMOTE (train only) is applied to the 512-d ResNet18 embeddings,
       *after* they are extracted from the training fold, never to the
       raw test embeddings — see `apply_smote_to_embeddings`.
    4. Class weights (train only) are computed from the post-split,
       pre-SMOTE label distribution and passed to the loss function as an
       additional (not a replacement) balancing signal.

`report_class_distribution` prints the exact before/after counts the
reviewer asked for, so every run leaves an auditable log.
"""
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_transforms(cfg: dict, train: bool) -> transforms.Compose:
    size = tuple(cfg["data"]["image_size"])
    mean = cfg["preprocessing"]["normalize_mean"]
    std = cfg["preprocessing"]["normalize_std"]
    aug = cfg["preprocessing"]["augmentation"]

    if train:
        return transforms.Compose([
            transforms.Resize(size),
            transforms.RandomRotation(aug["rotation_degrees"]),
            transforms.RandomHorizontalFlip(p=aug["horizontal_flip_p"]),
            transforms.ColorJitter(
                brightness=aug["color_jitter_brightness"],
                contrast=aug["color_jitter_contrast"],
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class MRIDataset(Dataset):
    """Thin wrapper around a list of `Sample` objects (see dataset.py)."""

    def __init__(self, samples: List, transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        img = self.transform(img)
        return img, s.label, s.subject_id


def compute_class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights computed on the TRAINING labels only."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # guard against an empty class in a small fold
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def apply_smote_to_embeddings(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    k_neighbors: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Oversample the minority classes with SMOTE on TRAINING embeddings only.

    This runs *after* ResNet18 feature extraction and *after* the
    subject-wise split, so:
      - synthetic nodes are interpolations of real training-set features,
        never of test-set features (no leakage);
      - the k-NN graph (graph_utils.build_knn_graph) is built from this
        SMOTE-balanced training embedding set plus the untouched test
        embeddings attached at inference time.
    """
    from imblearn.over_sampling import SMOTE

    class_counts = np.bincount(train_labels)
    min_class_size = class_counts[class_counts > 0].min()
    safe_k = max(1, min(k_neighbors, min_class_size - 1))
    smote = SMOTE(k_neighbors=safe_k, random_state=seed)
    emb_res, labels_res = smote.fit_resample(train_embeddings, train_labels)
    return emb_res, labels_res


def report_class_distribution(labels: np.ndarray, class_names: List[str], stage: str) -> dict:
    """Print + return the class distribution at a named pipeline stage
    (e.g. 'train_before_smote', 'train_after_smote', 'test')."""
    counts = np.bincount(labels, minlength=len(class_names))
    dist = {name: int(c) for name, c in zip(class_names, counts)}
    print(f"[class distribution :: {stage}] {dist}  (total={counts.sum()})")
    return dist
