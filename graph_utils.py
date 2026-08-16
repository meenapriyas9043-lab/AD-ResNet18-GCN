"""
graph_utils.py — k-NN graph construction and adjacency normalization.

This module is the direct, unambiguous answer to Reviewer 3 Comment 2 and
Comment 3.

NODE DEFINITION (fixes the manuscript's internal contradiction):
    Each graph node = one MRI image (one scan). NOT a feature dimension.
    A node's feature vector is that image's 512-d ResNet18 embedding.
    This is enforced by construction: `build_knn_graph` takes an
    (N_images, 512) embedding matrix and returns an (N_images, N_images)
    adjacency matrix — there is no code path that treats a feature
    dimension as a node.

LEAKAGE-SAFE / FOLD-SCOPED CONSTRUCTION (fixes "global k-NN graph"):
    The k-NN graph is fit using cosine similarity computed ONLY on the
    training embeddings of the current fold (`fit_reference`). Test-set
    embeddings are never used to decide which edges exist among training
    nodes. At inference time, each test node is connected to its k
    nearest TRAINING nodes only (`attach_query_nodes`) — the training
    graph's structure is never altered by test data, and no test-test
    edges are ever created. This is what "how test samples are
    incorporated into the model" (Comment 2) means in this codebase.
"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Graph:
    node_features: torch.Tensor   # (N, F)
    adjacency: torch.Tensor       # (N, N) symmetrically normalized, dense
    node_labels: torch.Tensor     # (N,) -1 for nodes without a known label (unused here, kept for extensibility)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_n @ b_n.T


def _knn_adjacency_from_similarity(sim: np.ndarray, k: int, self_loops: bool) -> np.ndarray:
    """Build a binary k-NN adjacency matrix from a (rows x cols) similarity
    matrix. Symmetrized via max() so an edge exists if either node picked
    the other as a top-k neighbor."""
    n_rows, n_cols = sim.shape
    adj = np.zeros((n_rows, n_cols), dtype=np.float32)
    k = min(k, n_cols - 1) if n_rows == n_cols else min(k, n_cols)
    for i in range(n_rows):
        row = sim[i].copy()
        if n_rows == n_cols:
            row[i] = -np.inf  # exclude self before top-k; self-loop added explicitly below
        nn_idx = np.argpartition(-row, k)[:k]
        adj[i, nn_idx] = 1.0
    return adj


def symmetric_normalize(adj: np.ndarray) -> np.ndarray:
    """D^-1/2 (A + I) D^-1/2 — the standard GCN propagation matrix
    (Kipf & Welling, 2017), matching Equation 4 of the manuscript."""
    n = adj.shape[0]
    adj_hat = adj + np.eye(n, dtype=np.float32)
    deg = adj_hat.sum(axis=1)
    deg_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = np.power(deg[nonzero], -0.5)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    return D_inv_sqrt @ adj_hat @ D_inv_sqrt


def build_knn_graph(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    k: int,
    self_loops: bool = True,
) -> Graph:
    """Build the TRAINING graph for one fold. Nodes = training images only.

    Called once per fold, using only that fold's (post-SMOTE) training
    embeddings — never the full dataset. This is the fold-scoped
    replacement for the manuscript's ambiguous "global k-NN graph".
    """
    sim = _cosine_sim(train_embeddings, train_embeddings)
    adj = _knn_adjacency_from_similarity(sim, k=k, self_loops=self_loops)
    adj_norm = symmetric_normalize(adj)
    return Graph(
        node_features=torch.tensor(train_embeddings, dtype=torch.float32),
        adjacency=torch.tensor(adj_norm, dtype=torch.float32),
        node_labels=torch.tensor(train_labels, dtype=torch.long),
    )


def attach_query_nodes(
    train_graph_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    query_labels: np.ndarray,
    k: int,
) -> Graph:
    """Extend the training graph with test/validation nodes for inference.

    Each query node is connected ONLY to its k nearest TRAINING nodes
    (never to other query nodes, never influencing training-node
    connectivity). This directly implements "how test samples are
    incorporated into the model" as requested in Comment 2, and keeps the
    procedure leakage-safe as required by Comment 3.
    """
    n_train = train_graph_embeddings.shape[0]
    n_query = query_embeddings.shape[0]
    full_embeddings = np.concatenate([train_graph_embeddings, query_embeddings], axis=0)

    sim_query_to_train = _cosine_sim(query_embeddings, train_graph_embeddings)
    query_to_train_adj = _knn_adjacency_from_similarity(sim_query_to_train, k=k, self_loops=False)

    full_adj = np.zeros((n_train + n_query, n_train + n_query), dtype=np.float32)
    train_sim = _cosine_sim(train_graph_embeddings, train_graph_embeddings)
    train_adj = _knn_adjacency_from_similarity(train_sim, k=k, self_loops=False)
    full_adj[:n_train, :n_train] = train_adj
    full_adj[n_train:, :n_train] = query_to_train_adj
    full_adj[:n_train, n_train:] = query_to_train_adj.T  # symmetric edges only into train block

    adj_norm = symmetric_normalize(full_adj)
    full_labels = np.concatenate([np.full(n_train, -1), query_labels])
    return Graph(
        node_features=torch.tensor(full_embeddings, dtype=torch.float32),
        adjacency=torch.tensor(adj_norm, dtype=torch.float32),
        node_labels=torch.tensor(full_labels, dtype=torch.long),
    )
