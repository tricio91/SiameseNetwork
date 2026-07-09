"""The embedding network: a ResNet backbone plus a projection head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .config import Config


class SimpleEmbeddingNet(nn.Module):
    """ResNet backbone + projection head mapping an image to an L2-normalized embedding."""

    def __init__(self, embed_dim: int = 512, pretrained: bool = True, freeze_bn: bool = True):
        """Build the backbone and projection head, then optionally freeze BatchNorm."""
        super().__init__()

        backbone = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # drop avgpool + fc
        self.feat_dim = 512
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, embed_dim),
        )

        self.embed_dim = embed_dim
        self._init_weights()
        if freeze_bn:
            self._freeze_bn()

    def _init_weights(self) -> None:
        """Kaiming-initialize the linear layers of the projection head."""
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _freeze_bn(self) -> None:
        """Put every BatchNorm layer in eval mode and stop its gradients."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the L2-normalized embedding for a batch of images."""
        features = self.backbone(x)
        pooled = self.pool(features).flatten(1)
        embedding = self.proj(pooled)
        return F.normalize(embedding, p=2, dim=1)

    def train(self, mode: bool = True) -> "SimpleEmbeddingNet":
        """Set training mode but keep BatchNorm frozen in eval (running stats stay fixed)."""
        super().train(mode)
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
        return self


def build_model(cfg: Config, device: torch.device, pretrained: bool = True) -> SimpleEmbeddingNet:
    """Instantiate SimpleEmbeddingNet from a Config and move it to the given device."""
    model = SimpleEmbeddingNet(
        embed_dim=cfg.embed_dim, pretrained=pretrained, freeze_bn=cfg.freeze_bn
    ).to(device)
    return model
