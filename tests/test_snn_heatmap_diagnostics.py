from __future__ import annotations

import pytest


def test_heatmap_snn_can_return_diagnostics():
    torch = pytest.importorskip("torch")
    pytest.importorskip("snntorch")

    from paper2.models.snn_heatmap import HeatmapSNN

    model = HeatmapSNN(num_steps=2, train_encoding="direct", eval_encoding="direct")
    out = model(torch.zeros(1, 3, 64, 64), stochastic=False, return_diagnostics=True)

    assert "heatmap_logits" in out
    assert "conf_logits" in out
    assert "diagnostics" in out
    assert "spike_rate_l1" in out["diagnostics"]
    assert "heatmap_logit_std" in out["diagnostics"]
