"""
Testes de arquitetura: shapes, skips multi-escala, LoRA (injeção + gradiente +
merge) e cabeça de graduação (atenção exposta). CPU, rápido.
"""
import torch

from src.models import SAM2SegGradeNet, build_model
from src.lora import count_trainable, merge_all_lora, lora_state_dict


def test_stub_shapes_and_skips():
    net = SAM2SegGradeNet(backbone="stub", n_seg_classes=4, n_grades=2)
    assert net.encoder.skip_channels == [128, 64, 32]      # skips multi-escala reais
    out = net(torch.randn(2, 3, 128, 128))
    assert out["seg_logits"].shape == (2, 4, 128, 128)
    assert out["grade_logits"].shape == (2, 2)
    assert out["grade_attn"].shape[0] == 2 and out["grade_attn"].shape[1] == 1


def test_arbitrary_input_size():
    net = SAM2SegGradeNet(backbone="stub").eval()
    for s in (64, 96, 160):
        out = net(torch.randn(1, 3, s, s), return_aux=False)
        assert out["seg_logits"].shape == (1, 4, s, s)


def test_lora_injection_and_gradient():
    net = SAM2SegGradeNet(backbone="stub", use_lora=True, lora_r=8)
    assert net.encoder.n_lora >= 1
    stats = count_trainable(net.encoder)
    assert 0 < stats["pct"] < 100                          # só parte treina
    out = net(torch.randn(2, 3, 64, 64))
    (out["seg_logits"].mean() + out["grade_logits"].mean()).backward()
    lora_grads = [p.grad for n, p in net.named_parameters()
                  if "lora_" in n and p.grad is not None]
    assert lora_grads and any(g.abs().sum() > 0 for g in lora_grads)
    # checkpoint LoRA é pequeno
    sd = lora_state_dict(net)
    assert all("lora_" in k for k in sd) and len(sd) >= 2


def test_lora_merge_equivalence():
    net = SAM2SegGradeNet(backbone="stub", use_lora=True, lora_r=8).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        before = net(x, return_aux=False)["seg_logits"]
        merge_all_lora(net)
        after = net(x, return_aux=False)["seg_logits"]
    assert torch.allclose(before, after, atol=1e-4)        # merge não muda a saída


def test_grade_head_uses_geometry():
    # a cabeça concatena descritores geométricos: frações (tumor total + K-1
    # sub-regiões) + 3 razões clínicas (ET/TC, TC/WT, NCR/TC) = K+3.
    net = SAM2SegGradeNet(backbone="stub", n_seg_classes=4)
    first_linear = net.grader.classifier[0]
    assert first_linear.in_features == 128 + (4 + 3)       # embed(128) + geom(K+3)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("OK:", name)
