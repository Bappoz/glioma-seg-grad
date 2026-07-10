"""
losses.py
=========
Funções de perda para o desbalanceamento extremo do BraTS (o tumor ocupa
tipicamente <2% dos voxels de uma fatia).

Segmentação:
    - DiceLoss (multi-classe, soft): otimiza overlap, robusta a desbalanço.
    - TverskyLoss / Focal-Tversky: generaliza o Dice com pesos assimétricos para
      FP e FN — `beta>alpha` prioriza RECALL (útil p/ ET pequeno). O termo focal
      (gamma) foca nas sub-regiões difíceis.
    - Cross-Entropy / Focal ponderada: gradiente estável por pixel.
    - Combo = região (Dice|Tversky) + pixel (CE|Focal) -> padrão-ouro em BraTS.

Graduação:
    - Cross-Entropy (opcionalmente ponderada + label smoothing p/ calibração).

Objetivo total multi-tarefa:
    L = w_seg * L_seg + w_grade * L_grade
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _one_hot(target: torch.Tensor, n_classes: int) -> torch.Tensor:
    return F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()


class DiceLoss(nn.Module):
    """Soft Dice multi-classe. Ignora opcionalmente o fundo (idx 0)."""

    def __init__(self, n_classes: int, ignore_background: bool = True, eps: float = 1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_background = ignore_background
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.softmax(logits, dim=1)
        tgt = _one_hot(target, self.n_classes)
        dims = (0, 2, 3)
        start = 1 if self.ignore_background else 0
        inter = (prob[:, start:] * tgt[:, start:]).sum(dims)
        card = prob[:, start:].sum(dims) + tgt[:, start:].sum(dims)
        dice = (2 * inter + self.eps) / (card + self.eps)
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    """Tversky (Salehi et al. 2017) / Focal-Tversky (Abraham & Khan 2019).

    TL = 1 - TP / (TP + alpha*FP + beta*FN);  Focal: (1-TI)^(1/gamma).
    beta>alpha penaliza mais os falsos-negativos -> mais recall no tumor pequeno.
    """

    def __init__(self, n_classes: int, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 1.0, ignore_background: bool = True, eps: float = 1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.ignore_background = ignore_background
        self.eps = eps

    def forward(self, logits, target):
        prob = torch.softmax(logits, dim=1)
        tgt = _one_hot(target, self.n_classes)
        dims = (0, 2, 3)
        start = 1 if self.ignore_background else 0
        p, g = prob[:, start:], tgt[:, start:]
        tp = (p * g).sum(dims)
        fp = (p * (1 - g)).sum(dims)
        fn = ((1 - p) * g).sum(dims)
        ti = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss = (1.0 - ti)
        if self.gamma != 1.0:
            loss = loss.pow(1.0 / self.gamma)
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal (Lin et al. 2017) para pixels difíceis; alternativa ao CE."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class SegLoss(nn.Module):
    """Combo região (Dice|Tversky) + pixel (CE|Focal) para segmentação."""

    def __init__(self, n_classes: int, ce_weight: torch.Tensor | None = None,
                 use_focal: bool = False, region: str = "dice",
                 tversky_alpha: float = 0.3, tversky_beta: float = 0.7,
                 tversky_gamma: float = 1.0, w_region: float = 1.0, w_ce: float = 1.0):
        super().__init__()
        if region == "tversky":
            self.region = TverskyLoss(n_classes, alpha=tversky_alpha,
                                      beta=tversky_beta, gamma=tversky_gamma)
        else:
            self.region = DiceLoss(n_classes)
        self.pixel = FocalLoss(weight=ce_weight) if use_focal else \
            nn.CrossEntropyLoss(weight=ce_weight)
        self.w_region, self.w_ce = w_region, w_ce

    def forward(self, logits, target):
        return self.w_region * self.region(logits, target) + self.w_ce * self.pixel(logits, target)


class MultiTaskLoss(nn.Module):
    """L = w_seg * SegLoss + w_grade * CE(grade)."""

    def __init__(self, n_seg_classes: int, seg_ce_weight=None, grade_weight=None,
                 w_seg: float = 1.0, w_grade: float = 0.5, use_focal: bool = False,
                 region: str = "dice", tversky_alpha: float = 0.3,
                 tversky_beta: float = 0.7, tversky_gamma: float = 1.0,
                 grade_label_smoothing: float = 0.05):
        super().__init__()
        self.seg = SegLoss(n_seg_classes, ce_weight=seg_ce_weight, use_focal=use_focal,
                           region=region, tversky_alpha=tversky_alpha,
                           tversky_beta=tversky_beta, tversky_gamma=tversky_gamma)
        self.grade = nn.CrossEntropyLoss(weight=grade_weight,
                                         label_smoothing=grade_label_smoothing)
        self.w_seg, self.w_grade = w_seg, w_grade

    def forward(self, out: dict, seg_target, grade_target):
        l_seg = self.seg(out["seg_logits"], seg_target)
        l_grade = self.grade(out["grade_logits"], grade_target)
        total = self.w_seg * l_seg + self.w_grade * l_grade
        return total, {"loss": total.item(), "loss_seg": l_seg.item(),
                       "loss_grade": l_grade.item()}
