"""
Testes do Trainer: wiring do peso de classe automático na graduação
(auto_grade_weight) e peso explícito via config. CPU, rápido.
"""
import torch
from torch.utils.data import DataLoader

from src.dataset import SyntheticBraTS
from src.train import Trainer, TrainConfig


def _cfg(**kw):
    base = dict(backbone="stub", epochs=1, batch_size=8, img_size=64, warmup_epochs=0,
                out_dir="/tmp/glioma_test_train")
    base.update(kw)
    return TrainConfig(**base)


def test_auto_grade_weight_injects_inverse_frequency():
    ds = SyntheticBraTS(60, img_size=64)
    labels = ds.grade_labels()
    tr = DataLoader(ds, batch_size=12)
    t = Trainer(_cfg(auto_grade_weight=True))
    assert t.criterion.grade.weight is None            # sem peso antes do fit
    t._maybe_auto_grade_weight(tr)
    w = t.criterion.grade.weight
    assert w is not None and w.numel() == 2
    # a classe minoritária no split recebe o maior peso
    minority = 0 if labels.count(0) < labels.count(1) else 1
    assert w[minority] == w.max()


def test_explicit_grade_class_weight_from_config():
    t = Trainer(_cfg(grade_class_weight=[2.0, 0.5]))
    assert t.criterion.grade.weight is not None
    assert torch.allclose(t.criterion.grade.weight.cpu(), torch.tensor([2.0, 0.5]))
    # explícito tem precedência: auto não sobrescreve
    tr = DataLoader(SyntheticBraTS(24, img_size=64), batch_size=8)
    t2 = Trainer(_cfg(grade_class_weight=[2.0, 0.5], auto_grade_weight=True))
    t2._maybe_auto_grade_weight(tr)
    assert torch.allclose(t2.criterion.grade.weight.cpu(), torch.tensor([2.0, 0.5]))


def test_weighted_loss_trains_one_epoch():
    tr = DataLoader(SyntheticBraTS(24, img_size=64), batch_size=8, shuffle=True)
    va = DataLoader(SyntheticBraTS(8, img_size=64), batch_size=8)
    hist = Trainer(_cfg(auto_grade_weight=True, region="tversky")).fit(tr, va)
    assert len(hist["val"]) == 1 and "grade_auc" in hist["val"][0]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("OK:", name)
