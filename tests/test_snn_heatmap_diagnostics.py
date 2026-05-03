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


def test_heatmap_loss_penalizes_distractor_mass():
    torch = pytest.importorskip("torch")

    from paper2.models.snn_heatmap import heatmap_loss

    logits = torch.zeros(1, 1, 16, 16)
    logits[0, 0, 12, 12] = 8.0
    outputs = {
        "heatmap_logits": logits,
        "conf_logits": torch.tensor([8.0]),
    }
    targets = torch.tensor([[0.25, 0.25, 1.0]])
    distractors = torch.tensor([[[12.0 / 15.0, 12.0 / 15.0]]])
    dmask = torch.tensor([[1.0]])

    no_repel, no_parts = heatmap_loss(
        outputs,
        targets,
        heatmap_size=16,
        sigma=1.5,
        coord_weight=0.0,
        heatmap_weight=0.0,
        conf_weight=0.0,
        softargmax_temperature=20.0,
        distractor_centers=distractors,
        distractor_mask=dmask,
        distractor_weight=0.0,
    )
    with_repel, parts = heatmap_loss(
        outputs,
        targets,
        heatmap_size=16,
        sigma=1.5,
        coord_weight=0.0,
        heatmap_weight=0.0,
        conf_weight=0.0,
        softargmax_temperature=20.0,
        distractor_centers=distractors,
        distractor_mask=dmask,
        distractor_weight=2.0,
        distractor_sigma=1.5,
    )

    assert float(no_repel.item()) == 0.0
    assert no_parts["distractor_loss"] == 0.0
    assert float(with_repel.item()) > 0.0
    assert parts["distractor_loss"] > 0.0
