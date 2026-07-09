"""Embedding extraction and retrieval metrics (Recall@K, mAP)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .losses import hard_triplet_loss


@torch.no_grad()
def evaluate_embeddings(model, dataloader, device, margin=0.3) -> Tuple[np.ndarray, np.ndarray, float]:
    """Extract embeddings for a whole dataloader and report the mean triplet loss."""
    model.eval()
    all_embeddings: List[np.ndarray] = []
    all_labels: List[int] = []
    total_loss, n_batches = 0.0, 0

    for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        embeddings = model(images.to(device))
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.extend(labels.numpy())
        if len(labels) >= 4:
            loss = hard_triplet_loss(embeddings, labels.to(device), margin=margin)
            if not torch.isnan(loss):
                total_loss += loss.item()
                n_batches += 1

    return np.vstack(all_embeddings), np.array(all_labels), total_loss / max(1, n_batches)


def compute_retrieval_metrics(
    embeddings: np.ndarray, labels: np.ndarray, k_values=(1, 3, 5, 10)
) -> Tuple[Dict[int, float], float]:
    """Compute Recall@K (for each K) and mean average precision over the embeddings."""
    n_samples = len(labels)
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, -1)  # never retrieve the query itself

    recalls = {k: 0 for k in k_values}
    average_precisions: List[float] = []

    for i in range(n_samples):
        ranking = labels[np.argsort(-sim_matrix[i])]
        query_label = labels[i]

        for k in k_values:
            if query_label in ranking[:k]:
                recalls[k] += 1

        relevant = (ranking == query_label)
        n_relevant = relevant.sum()
        if n_relevant > 0:
            precisions = np.cumsum(relevant) / (np.arange(n_samples) + 1)
            average_precisions.append((precisions * relevant).sum() / n_relevant)

    recalls = {k: v / n_samples * 100 for k, v in recalls.items()}
    mean_ap = float(np.mean(average_precisions) * 100) if average_precisions else 0.0
    return recalls, mean_ap
