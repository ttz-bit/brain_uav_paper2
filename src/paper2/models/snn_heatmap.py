from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapSNN(nn.Module):
    def __init__(
        self,
        *,
        beta: float = 0.95,
        num_steps: int = 12,
        train_encoding: str = "rate",
        eval_encoding: str = "direct",
    ):
        super().__init__()
        try:
            import snntorch as snn
            from snntorch import surrogate
        except Exception as e:
            raise RuntimeError("snntorch is required. Install: pip install snntorch") from e

        spike_grad = surrogate.fast_sigmoid(slope=25.0)
        self.num_steps = int(num_steps)
        self.train_encoding = str(train_encoding)
        self.eval_encoding = str(eval_encoding)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.lif1 = snn.Leaky(beta=float(beta), spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.lif2 = snn.Leaky(beta=float(beta), spike_grad=spike_grad)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.lif3 = snn.Leaky(beta=float(beta), spike_grad=spike_grad)
        self.heatmap_head = nn.Conv2d(32, 1, kernel_size=1)
        self.conf_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 1),
        )

    def _rate_encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rand_like(x).le(x).float()

    def _encode(self, x: torch.Tensor, stochastic: bool) -> torch.Tensor:
        mode = self.train_encoding if stochastic else self.eval_encoding
        if mode == "rate":
            return self._rate_encode(x)
        return x

    def forward(self, x: torch.Tensor, stochastic: bool = True) -> dict[str, torch.Tensor]:
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        feat_acc = None
        for _ in range(self.num_steps):
            x_t = self._encode(x, stochastic=stochastic)
            spk1, mem1 = self.lif1(self.conv1(x_t), mem1)
            spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
            spk3, mem3 = self.lif3(self.conv3(spk2), mem3)
            feat_acc = spk3 if feat_acc is None else feat_acc + spk3
        feat = feat_acc / max(1, self.num_steps)
        return {
            "heatmap_logits": self.heatmap_head(feat),
            "conf_logits": self.conf_head(feat).squeeze(1),
        }


def make_gaussian_heatmaps(
    targets: torch.Tensor,
    *,
    heatmap_size: int,
    sigma: float,
) -> torch.Tensor:
    b = int(targets.shape[0])
    size = int(heatmap_size)
    device = targets.device
    dtype = targets.dtype
    ys = torch.arange(size, device=device, dtype=dtype).view(1, size, 1)
    xs = torch.arange(size, device=device, dtype=dtype).view(1, 1, size)
    cx = targets[:, 0].clamp(0.0, 1.0).view(b, 1, 1) * float(size - 1)
    cy = targets[:, 1].clamp(0.0, 1.0).view(b, 1, 1) * float(size - 1)
    valid = targets[:, 2].clamp(0.0, 1.0).view(b, 1, 1)
    dist2 = (xs - cx).pow(2) + (ys - cy).pow(2)
    heatmaps = torch.exp(-dist2 / max(1e-6, 2.0 * float(sigma) * float(sigma)))
    return heatmaps.unsqueeze(1) * valid.view(b, 1, 1, 1)


def soft_argmax_2d(logits: torch.Tensor, *, temperature: float = 10.0) -> torch.Tensor:
    b, _, h, w = logits.shape
    flat = (logits.view(b, -1) * float(temperature)).softmax(dim=1)
    ys, xs = torch.meshgrid(
        torch.linspace(0.0, 1.0, h, device=logits.device, dtype=logits.dtype),
        torch.linspace(0.0, 1.0, w, device=logits.device, dtype=logits.dtype),
        indexing="ij",
    )
    x = (flat * xs.reshape(1, -1)).sum(dim=1)
    y = (flat * ys.reshape(1, -1)).sum(dim=1)
    return torch.stack([x, y], dim=1)


def peak_argmax_2d(logits: torch.Tensor) -> torch.Tensor:
    b, _, h, w = logits.shape
    idx = logits.view(b, -1).argmax(dim=1)
    y = torch.div(idx, w, rounding_mode="floor").to(dtype=logits.dtype)
    x = (idx % w).to(dtype=logits.dtype)
    return torch.stack([x / max(1, w - 1), y / max(1, h - 1)], dim=1)


def heatmap_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    *,
    heatmap_size: int,
    sigma: float,
    coord_weight: float,
    heatmap_weight: float,
    conf_weight: float,
    softargmax_temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_heatmaps = make_gaussian_heatmaps(targets, heatmap_size=heatmap_size, sigma=sigma)
    pred_heatmaps = torch.sigmoid(outputs["heatmap_logits"])
    heatmap_weight_map = 1.0 + target_heatmaps * 20.0
    hm_loss = ((pred_heatmaps - target_heatmaps).pow(2) * heatmap_weight_map).mean()
    pred_xy = soft_argmax_2d(outputs["heatmap_logits"], temperature=softargmax_temperature)
    coord_loss = F.smooth_l1_loss(pred_xy, targets[:, :2])
    conf_loss = F.binary_cross_entropy_with_logits(outputs["conf_logits"], targets[:, 2])
    total = float(heatmap_weight) * hm_loss + float(coord_weight) * coord_loss + float(conf_weight) * conf_loss
    parts = {
        "heatmap_loss": float(hm_loss.detach().cpu().item()),
        "coord_loss": float(coord_loss.detach().cpu().item()),
        "conf_loss": float(conf_loss.detach().cpu().item()),
    }
    return total, parts
