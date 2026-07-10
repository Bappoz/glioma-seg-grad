"""
Testes do baseline U-Net (mesma interface do modelo principal) e do harness de
build_variant usado pela ablação. CPU, rápido.
"""
import torch

from src.models import SAM2SegGradeNet
from src.baseline import UNetSegGradeNet, build_variant


def test_unet_baseline_interface():
    net = UNetSegGradeNet(n_seg_classes=4, n_grades=2)
    out = net(torch.randn(2, 3, 96, 96))
    assert out["seg_logits"].shape == (2, 4, 96, 96)
    assert out["grade_logits"].shape == (2, 2)
    assert "grade_attn" in out and "feat" in out


def test_build_variant():
    from src.train import TrainConfig
    cfg = TrainConfig(backbone="stub")
    assert isinstance(build_variant(cfg, "unet"), UNetSegGradeNet)
    assert isinstance(build_variant(cfg, "stub_lora"), SAM2SegGradeNet)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("OK:", name)
