"""Triplet loss with online hard mining."""
from __future__ import annotations

import torch


def hard_triplet_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.3
) -> torch.Tensor:
    """Batch-hard triplet loss: hardest positive vs. hardest negative per anchor.

    Distances are cosine (1 - cosine similarity). For every anchor we take the
    farthest same-class sample and the closest different-class sample, then push
    them apart by at least `margin`.
    """
    device = embeddings.device
    B = embeddings.size(0)

    # Pairwise cosine distances.
    sim_matrix = embeddings @ embeddings.T
    dist_matrix = 1.0 - sim_matrix

    # Same-class / different-class masks (excluding the diagonal for positives).
    labels = labels.view(-1, 1)
    same_class = (labels == labels.T).float() - torch.eye(B, device=device)
    diff_class = 1.0 - (labels == labels.T).float()

    # Hardest positive: max distance within the same class.
    dist_pos = dist_matrix.clone()
    dist_pos[same_class == 0] = -float("inf")
    hardest_positive, _ = dist_pos.max(dim=1)

    # Hardest negative: min distance to any other class.
    dist_neg = dist_matrix.clone()
    dist_neg[diff_class == 0] = float("inf")
    hardest_negative, _ = dist_neg.min(dim=1)

    loss = torch.relu(hardest_positive - hardest_negative + margin)
    valid = (hardest_positive > -float("inf")) & (hardest_negative < float("inf"))

    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return loss[valid].mean()
