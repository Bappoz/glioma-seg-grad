"""
Testes de incerteza: MC-Dropout (incerteza epistêmica não-negativa) e TTA
(probabilidades de segmentação normalizadas). CPU, rápido.
"""
import torch

from src.models import SAM2SegGradeNet
from src.uncertainty import mc_dropout_predict, tta_predict, enable_mc_dropout


def test_mc_dropout_and_tta():
    net = SAM2SegGradeNet(backbone="stub", p_drop=0.3)
    x = torch.randn(2, 3, 64, 64)
    assert enable_mc_dropout(net) >= 1
    mc = mc_dropout_predict(net, x, n_samples=4)
    assert mc["seg_prob"].shape == (2, 4, 64, 64)
    assert mc["seg_mutual_info"].shape == (2, 64, 64)
    assert (mc["seg_mutual_info"] >= 0).all()              # MI não-negativa
    tta = tta_predict(net, x)
    assert tta["seg_pred"].shape == (2, 64, 64)
    assert torch.allclose(tta["seg_prob"].sum(1),
                          torch.ones(2, 64, 64), atol=1e-4)  # probs somam 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("OK:", name)
