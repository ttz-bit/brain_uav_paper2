from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from paper2.models.cnn_heatmap import HeatmapCNN


def test_heatmap_cnn_forward_shapes():
    model = HeatmapCNN(width=16)
    x = torch.zeros((2, 3, 256, 256), dtype=torch.float32)
    out = model(x)
    assert out["heatmap_logits"].shape[0] == 2
    assert out["conf_logits"].shape == (2,)
