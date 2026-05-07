from __future__ import annotations

import torch
import torch.nn as nn


class HeatmapCNN(nn.Module):
    """CNN heatmap baseline matched to HeatmapSNN's output contract.

    ``arch`` controls the backbone capacity:
    - ``legacy``: 3 conv layers, configurable ``width`` (default 32).
    - ``enhanced``: 6 conv layers, 64 channels, skip connections, dilated conv.
      Structurally matches the enhanced HeatmapSNN (LIF → ReLU).
    """

    def __init__(self, *, width: int = 32, arch: str = "enhanced"):
        super().__init__()
        self.arch = str(arch or "enhanced")
        if self.arch not in {"legacy", "enhanced"}:
            raise ValueError("HeatmapCNN arch must be 'legacy' or 'enhanced'.")

        if self.arch == "legacy":
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
            feature_channels = w
            self.heatmap_head = nn.Conv2d(feature_channels, 1, kernel_size=1)
        else:
            # enhanced: matches the enhanced HeatmapSNN layer-for-layer
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
            self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            self.skip2 = nn.Conv2d(32, 64, kernel_size=1)
            feature_channels = 64
            self.heatmap_head = nn.Sequential(
                nn.Conv2d(feature_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1),
            )

        self.conf_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_channels, 1),
        )

    def forward(self, x: torch.Tensor, stochastic: bool = False) -> dict[str, torch.Tensor]:
        del stochastic
        if self.arch == "legacy":
            feat = self.backbone(x)
        else:
            spk1 = nn.functional.relu(self.conv1(x), inplace=True)
            spk2 = nn.functional.relu(self.conv2(spk1), inplace=True)
            spk3 = nn.functional.relu(self.conv3(spk2), inplace=True)
            spk4 = nn.functional.relu(self.conv4(spk3), inplace=True)
            spk5 = nn.functional.relu(self.conv5(spk4), inplace=True)
            spk6 = nn.functional.relu(self.conv6(spk5 + spk3), inplace=True)
            feat = spk6 + self.skip2(spk2)

        return {
            "heatmap_logits": self.heatmap_head(feat),
            "conf_logits": self.conf_head(feat).squeeze(1),
        }
