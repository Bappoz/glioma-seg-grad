"""
Testes de explicabilidade: atenção intrínseca da graduação e saliência por
oclusão retornam mapas normalizados em [0,1] com o shape esperado. CPU, rápido.
"""
import torch

from src.models import SAM2SegGradeNet
from src.explain import grade_attention, occlusion_saliency


def test_explain_maps_normalized():
    net = SAM2SegGradeNet(backbone="stub").eval()
    x = torch.randn(1, 3, 48, 48)
    a = grade_attention(net, x)
    s = occlusion_saliency(net, x, patch=16, stride=16)
    assert a.shape == (1, 48, 48) and s.shape == (1, 48, 48)
    assert float(a.min()) >= 0 and float(a.max()) <= 1.0001
    assert float(s.min()) >= 0 and float(s.max()) <= 1.0001


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("OK:", name)
