"""
gcn_model.py — Graph Convolutional layers and the proposed Hybrid
ResNet18+GCN classifier (Equations 4-5 of the manuscript).

    Z = softmax( A_hat . ReLU(A_hat . X . W0) . W1 )

implemented as two stacked `GCNLayer` modules followed by a linear
classification head, matching Table "Hyperparameter Configuration of the
Proposed ResNet18-GCN Model" (2 GCN layers, ReLU activation).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """One graph-convolution: H' = activation(A_hat @ H @ W)."""

    def __init__(self, in_dim: int, out_dim: int, activation: bool = True, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, H: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        H = self.dropout(H)
        H = A_hat @ self.linear(H)
        if self.activation:
            H = F.relu(H)
        return H


class HybridResNet18GCN(nn.Module):
    """GCN head that consumes pre-extracted ResNet18 node embeddings.

    The CNN feature extraction (backbones.ResNet18FeatureExtractor) is
    run once per fold to build node features; this module implements only
    the graph-convolution + classification stage, matching the two-stage
    "extract, then relate" design described in Section 3.3 of the
    manuscript.
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        assert n_layers >= 1
        layers = []
        dim = in_dim
        for i in range(n_layers - 1):
            layers.append(GCNLayer(dim, hidden_dim, activation=True, dropout=dropout))
            dim = hidden_dim
        layers.append(GCNLayer(dim, hidden_dim, activation=True, dropout=dropout))
        self.gcn_layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        H = X
        for layer in self.gcn_layers:
            H = layer(H, A_hat)
        logits = self.classifier(H)
        return logits  # softmax applied by the loss (CrossEntropyLoss) / at inference time
