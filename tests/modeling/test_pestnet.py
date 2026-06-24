from __future__ import annotations

import torch

from pestscope.modeling import build_model, count_parameters


def test_pestnet_s_forward_shape_and_size_bound() -> None:
    model = build_model("pestnet_s", num_classes=12, width=16, dropout=0.1)
    logits = model(torch.randn(2, 3, 64, 64))

    assert logits.shape == (2, 12)
    assert 100_000 < count_parameters(model) < 2_000_000


def test_simple_cnn_forward_shape() -> None:
    model = build_model("simple_cnn", num_classes=3, width=8, dropout=0.0)

    assert model(torch.randn(4, 3, 32, 32)).shape == (4, 3)
