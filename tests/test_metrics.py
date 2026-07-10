"""
Testes de métricas: Dice/IoU/HD95 de segmentação, ROC/AUC e calibração
(ECE/Brier/reliability) da graduação. CPU, rápido.
"""
import numpy as np
import torch

from src.metrics import (seg_scores, seg_scores_hd95, grade_roc_auc,
                         expected_calibration_error, brier_score, grade_report,
                         reliability_bins)


def test_seg_scores_and_hd95():
    pred = torch.zeros(1, 16, 16, dtype=torch.long); pred[0, 4:10, 4:10] = 3
    tgt = torch.zeros(1, 16, 16, dtype=torch.long); tgt[0, 4:10, 4:10] = 3
    sc = seg_scores(pred, tgt)
    assert sc["dice_ET"] > 0.99 and sc["dice_WT"] > 0.99
    hd = seg_scores_hd95(pred, tgt)
    assert hd["hd95_ET"] < 1.0                             # bordas coincidem
    # HD95 nan quando GT vazio p/ a sub-região
    empty = torch.zeros(1, 16, 16, dtype=torch.long)
    assert np.isnan(seg_scores_hd95(pred, empty)["hd95_ET"])


def test_calibration_metrics():
    y = torch.randint(0, 2, (200,))
    logits = torch.stack([torch.where(y == 0, 3.0, -2.0),
                          torch.where(y == 1, 3.0, -2.0)], 1) + torch.randn(200, 2) * 0.5
    ece = expected_calibration_error(logits, y)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= brier_score(logits, y) <= 1.0
    rep = grade_report(logits, y)
    assert 0.0 <= rep["f1_macro"] <= 1.0 and len(rep["confusion"]) == 2
    b = reliability_bins(logits, y, n_bins=5)
    assert len(b["conf"]) == 5


def test_roc_auc_separable():
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    logits = torch.tensor([[3., -3.]] * 3 + [[-3., 3.]] * 3)
    roc = grade_roc_auc(logits, y)
    assert roc["auc"] > 0.99


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("OK:", name)
