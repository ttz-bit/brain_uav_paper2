from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_HEATMAP_GRID_CACHE: dict[tuple[str, torch.dtype, int], tuple[torch.Tensor, torch.Tensor]] = {}
_SOFTARGMAX_GRID_CACHE: dict[tuple[str, torch.dtype, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _device_key(device: torch.device) -> str:
    if device.index is None:
        return str(device.type)
    return f"{device.type}:{device.index}"


def _has_inference_tensor(cached: tuple[torch.Tensor, ...] | None) -> bool:
    if cached is None:
        return False
    for tensor in cached:
        is_inference = getattr(tensor, "is_inference", None)
        if callable(is_inference) and bool(is_inference()):
            return True
    return False


class HeatmapSNN(nn.Module):
    def __init__(
        self,
        *,
        beta: float = 0.95,
        num_steps: int = 12,
        train_encoding: str = "direct",
        eval_encoding: str = "direct",
        arch: str = "enhanced",
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
        self.arch = str(arch or "enhanced")
        if self.arch not in {"legacy", "enhanced"}:
            raise ValueError("HeatmapSNN arch must be 'legacy' or 'enhanced'.")
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.lif1 = snn.Leaky(beta=float(beta), spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.lif2 = snn.Leaky(beta=float(beta), spike_grad=spike_grad)
        if self.arch == "legacy":
            self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
            self.lif3 = snn.Leaky(beta=float(beta), spike_grad=spike_grad)
            feature_channels = 32
            self.heatmap_head = nn.Conv2d(feature_channels, 1, kernel_size=1)
        else:
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.lif3 = snn.Leaky(beta=float(beta), spike_grad=spike_grad)
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            self.lif4 = snn.Leaky(beta=float(beta), spike_grad=spike_grad)
            self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
            self.lif5 = snn.Leaky(beta=float(beta), spike_grad=spike_grad)
            self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            self.lif6 = snn.Leaky(beta=float(beta), spike_grad=spike_grad)
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

    def _rate_encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rand_like(x).le(x).float()

    def _encode(self, x: torch.Tensor, stochastic: bool) -> torch.Tensor:
        mode = self.train_encoding if stochastic else self.eval_encoding
        if mode == "rate":
            return self._rate_encode(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        stochastic: bool = True,
        *,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky() if self.arch == "enhanced" else None
        mem5 = self.lif5.init_leaky() if self.arch == "enhanced" else None
        mem6 = self.lif6.init_leaky() if self.arch == "enhanced" else None
        feat_acc = None
        spike1_total = None
        spike2_total = None
        spike3_total = None
        spike4_total = None
        spike5_total = None
        spike6_total = None
        for _ in range(self.num_steps):
            x_t = self._encode(x, stochastic=stochastic)
            spk1, mem1 = self.lif1(self.conv1(x_t), mem1)
            spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
            if self.arch == "legacy":
                spk3, mem3 = self.lif3(self.conv3(spk2), mem3)
                feat_t = spk3
            else:
                assert mem4 is not None and mem5 is not None and mem6 is not None
                spk3, mem3 = self.lif3(self.conv3(spk2), mem3)
                spk4, mem4 = self.lif4(self.conv4(spk3), mem4)
                spk5, mem5 = self.lif5(self.conv5(spk4), mem5)
                spk6, mem6 = self.lif6(self.conv6(spk5 + spk3), mem6)
                feat_t = spk6 + self.skip2(spk2)
            feat_acc = feat_t if feat_acc is None else feat_acc + feat_t
            if return_diagnostics:
                spike1_total = spk1 if spike1_total is None else spike1_total + spk1
                spike2_total = spk2 if spike2_total is None else spike2_total + spk2
                spike3_total = spk3 if spike3_total is None else spike3_total + spk3
                if self.arch == "enhanced":
                    spike4_total = spk4 if spike4_total is None else spike4_total + spk4
                    spike5_total = spk5 if spike5_total is None else spike5_total + spk5
                    spike6_total = spk6 if spike6_total is None else spike6_total + spk6
        feat = feat_acc / max(1, self.num_steps)
        heatmap_logits = self.heatmap_head(feat)
        out = {
            "heatmap_logits": heatmap_logits,
            "conf_logits": self.conf_head(feat).squeeze(1),
        }
        if return_diagnostics:
            steps = max(1, self.num_steps)
            assert spike1_total is not None and spike2_total is not None and spike3_total is not None
            out["diagnostics"] = {
                "arch": self.arch,
                "spike_rate_l1": (spike1_total / steps).mean().detach(),
                "spike_rate_l2": (spike2_total / steps).mean().detach(),
                "spike_rate_l3": (spike3_total / steps).mean().detach(),
                "feature_mean": feat.mean().detach(),
                "feature_std": feat.std(unbiased=False).detach(),
                "heatmap_logit_mean": heatmap_logits.mean().detach(),
                "heatmap_logit_std": heatmap_logits.std(unbiased=False).detach(),
                "heatmap_logit_min": heatmap_logits.min().detach(),
                "heatmap_logit_max": heatmap_logits.max().detach(),
            }
            if self.arch == "enhanced":
                assert spike4_total is not None and spike5_total is not None and spike6_total is not None
                out["diagnostics"].update(
                    {
                        "spike_rate_l4": (spike4_total / steps).mean().detach(),
                        "spike_rate_l5": (spike5_total / steps).mean().detach(),
                        "spike_rate_l6": (spike6_total / steps).mean().detach(),
                    }
                )
        return out


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
    key = (_device_key(device), dtype, size)
    cached = _HEATMAP_GRID_CACHE.get(key)
    if _has_inference_tensor(cached):
        _HEATMAP_GRID_CACHE.pop(key, None)
        cached = None
    if cached is None:
        ys = torch.arange(size, device=device, dtype=dtype).view(1, size, 1)
        xs = torch.arange(size, device=device, dtype=dtype).view(1, 1, size)
        cached = (ys, xs)
        if not torch.is_inference_mode_enabled():
            _HEATMAP_GRID_CACHE[key] = cached
    ys, xs = cached
    cx = targets[:, 0].clamp(0.0, 1.0).view(b, 1, 1) * float(size - 1)
    cy = targets[:, 1].clamp(0.0, 1.0).view(b, 1, 1) * float(size - 1)
    valid = targets[:, 2].clamp(0.0, 1.0).view(b, 1, 1)
    dist2 = (xs - cx).pow(2) + (ys - cy).pow(2)
    heatmaps = torch.exp(-dist2 / max(1e-6, 2.0 * float(sigma) * float(sigma)))
    return heatmaps.unsqueeze(1) * valid.view(b, 1, 1, 1)


def _resize_mask_to_logits(mask: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    valid = mask.to(device=logits.device, dtype=logits.dtype)
    if valid.ndim == 3:
        valid = valid.unsqueeze(1)
    if valid.shape[-2:] != logits.shape[-2:]:
        valid = F.interpolate(valid, size=logits.shape[-2:], mode="area")
    valid = (valid > 0.5).to(dtype=logits.dtype)
    flat = valid.flatten(1)
    empty = flat.sum(dim=1) <= 0.0
    if bool(empty.any()):
        valid = valid.clone()
        valid[empty] = 1.0
    return valid


def mask_heatmap_logits(logits: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    if valid_mask is None or int(valid_mask.numel()) <= 0:
        return logits
    valid = _resize_mask_to_logits(valid_mask, logits)
    return logits.masked_fill(valid <= 0.0, -1.0e4)


def soft_argmax_2d(
    logits: torch.Tensor,
    *,
    temperature: float = 10.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = mask_heatmap_logits(logits, valid_mask)
    b, _, h, w = logits.shape
    flat = (logits.view(b, -1) * float(temperature)).softmax(dim=1)
    key = (_device_key(logits.device), logits.dtype, int(h), int(w))
    cached = _SOFTARGMAX_GRID_CACHE.get(key)
    if _has_inference_tensor(cached):
        _SOFTARGMAX_GRID_CACHE.pop(key, None)
        cached = None
    if cached is None:
        ys, xs = torch.meshgrid(
            torch.linspace(0.0, 1.0, h, device=logits.device, dtype=logits.dtype),
            torch.linspace(0.0, 1.0, w, device=logits.device, dtype=logits.dtype),
            indexing="ij",
        )
        cached = (ys.reshape(1, -1), xs.reshape(1, -1))
        if not torch.is_inference_mode_enabled():
            _SOFTARGMAX_GRID_CACHE[key] = cached
    ys_flat, xs_flat = cached
    x = (flat * xs_flat).sum(dim=1)
    y = (flat * ys_flat).sum(dim=1)
    return torch.stack([x, y], dim=1)


def peak_argmax_2d(logits: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    logits = mask_heatmap_logits(logits, valid_mask)
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
    distractor_centers: torch.Tensor | None = None,
    distractor_mask: torch.Tensor | None = None,
    distractor_weight: float = 0.0,
    distractor_sigma: float | None = None,
    land_mask: torch.Tensor | None = None,
    land_weight: float = 0.0,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_heatmaps = make_gaussian_heatmaps(targets, heatmap_size=heatmap_size, sigma=sigma)
    heatmap_logits = mask_heatmap_logits(outputs["heatmap_logits"], valid_mask)
    if valid_mask is not None and int(valid_mask.numel()) > 0:
        target_valid = _resize_mask_to_logits(valid_mask, outputs["heatmap_logits"]).detach()
        target_heatmaps = target_heatmaps * target_valid
    valid = targets[:, 2] > 0.5
    target_flat = target_heatmaps.flatten(1)
    target_probs = target_flat / target_flat.sum(dim=1, keepdim=True).clamp_min(1e-12)
    log_probs = F.log_softmax(heatmap_logits.flatten(1), dim=1)
    hm_per_sample = -(target_probs * log_probs).sum(dim=1)
    valid_weight = valid.to(dtype=hm_per_sample.dtype)
    valid_den = valid_weight.sum().clamp_min(1.0)
    hm_loss = (hm_per_sample * valid_weight).sum() / valid_den
    pred_xy = soft_argmax_2d(heatmap_logits, temperature=softargmax_temperature)
    coord_per_sample = F.smooth_l1_loss(pred_xy, targets[:, :2], reduction="none").mean(dim=1)
    coord_loss = (coord_per_sample * valid_weight).sum() / valid_den
    conf_loss = F.binary_cross_entropy_with_logits(outputs["conf_logits"], targets[:, 2])
    distractor_loss = outputs["heatmap_logits"].new_tensor(0.0)
    pred_probs = None
    if (
        distractor_centers is not None
        and distractor_mask is not None
        and float(distractor_weight) > 0.0
        and int(distractor_centers.numel()) > 0
    ):
        b, n, _ = distractor_centers.shape
        d_valid = distractor_mask.to(device=targets.device, dtype=targets.dtype).clamp(0.0, 1.0)
        d_targets = torch.cat(
            [
                distractor_centers.to(device=targets.device, dtype=targets.dtype).reshape(b * n, 2),
                d_valid.reshape(b * n, 1),
            ],
            dim=1,
        )
        d_heatmaps = make_gaussian_heatmaps(
            d_targets,
            heatmap_size=heatmap_size,
            sigma=float(distractor_sigma if distractor_sigma is not None else sigma),
        ).reshape(b, n, 1, int(heatmap_size), int(heatmap_size))
        d_mask = d_heatmaps.max(dim=1).values.flatten(1).detach()
        pred_probs = F.softmax(heatmap_logits.flatten(1), dim=1)
        d_per_sample = (pred_probs * d_mask).sum(dim=1)
        has_distractor = (d_valid.sum(dim=1) > 0.5).to(dtype=d_per_sample.dtype)
        d_den = (has_distractor * valid_weight).sum().clamp_min(1.0)
        distractor_loss = (d_per_sample * has_distractor * valid_weight).sum() / d_den
    land_loss = outputs["heatmap_logits"].new_tensor(0.0)
    if land_mask is not None and float(land_weight) > 0.0 and int(land_mask.numel()) > 0:
        logits = heatmap_logits
        if pred_probs is None:
            pred_probs = F.softmax(logits.flatten(1), dim=1)
        land = land_mask.to(device=logits.device, dtype=logits.dtype)
        if land.ndim == 3:
            land = land.unsqueeze(1)
        if land.shape[-2:] != logits.shape[-2:]:
            land = F.interpolate(land, size=logits.shape[-2:], mode="area")
        land = land.clamp(0.0, 1.0).flatten(1).detach()
        land_per_sample = (pred_probs * land).sum(dim=1)
        has_land = (land.sum(dim=1) > 0.0).to(dtype=land_per_sample.dtype)
        land_den = (has_land * valid_weight).sum().clamp_min(1.0)
        land_loss = (land_per_sample * has_land * valid_weight).sum() / land_den
    total = (
        float(heatmap_weight) * hm_loss
        + float(coord_weight) * coord_loss
        + float(conf_weight) * conf_loss
        + float(distractor_weight) * distractor_loss
        + float(land_weight) * land_loss
    )
    parts = {
        "heatmap_loss": float(hm_loss.detach().cpu().item()),
        "coord_loss": float(coord_loss.detach().cpu().item()),
        "conf_loss": float(conf_loss.detach().cpu().item()),
        "distractor_loss": float(distractor_loss.detach().cpu().item()),
        "land_loss": float(land_loss.detach().cpu().item()),
    }
    return total, parts
