from __future__ import annotations

import torch
import torch.nn as nn


class HeatmapCNN(nn.Module):
    """CNN heatmap baseline matched to HeatmapSNN's output contract."""

    def __init__(self, *, width: int = 32):
        super().__init__()
        w = int(width)
        if w <= 0:
            raise ValueError("width must be positive.")
        self.backbone = nn.Sequential(
            nn.Conv2d(3, max(8, w // 2), kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, w // 2), w, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(w, w, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Conv2d(w, 1, kernel_size=1)
        self.conf_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(w, 1),
        )

    def forward(self, x: torch.Tensor, stochastic: bool = False) -> dict[str, torch.Tensor]:
        del stochastic
        feat = self.backbone(x)
        return {
            "heatmap_logits": self.heatmap_head(feat),
            "conf_logits": self.conf_head(feat).squeeze(1),
        }
