"""
losses.py
=========
Funções de perda para o desbalanceamento extremo do BraTS (o tumor ocupa
tipicamente <2% dos voxels de uma fatia).

Segmentação:
    - DiceLoss (multi-classe, soft): otimiza overlap, robusta a desbalanço.
    - Cross-Entropy ponderada: gradiente estável por pixel.
    - Combo = Dice + CE  -> padrão-ouro em BraTS. Opcional: Focal no lugar do CE
      para focar em pixels difíceis (bordas necrose/edema).

Graduação:
    - Cross-Entropy (opcionalmente ponderada por frequência de classe).

O objetivo total é multi-tarefa:
    L = w_seg * L_seg + w_grade * L_grade
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice multi-classe. Ignora opcionalmente o fundo (idx 0)."""

    def __init__(self, n_classes: int, ignore_background: bool = True, eps: float = 1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_background = ignore_background
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits [B,K,H,W]; target [B,H,W]
        prob = torch.softmax(logits, dim=1)
        tgt = F.one_hot(target, self.n_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        start = 1 if self.ignore_background else 0
        inter = (prob[:, start:] * tgt[:, start:]).sum(dims)
        card = prob[:, start:].sum(dims) + tgt[:, start:].sum(dims)
        dice = (2 * inter + self.eps) / (card + self.eps)
        return 1.0 - dice.mean()


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
    """Combo Dice + (CE | Focal) para segmentação."""

    def __init__(self, n_classes: int, ce_weight: torch.Tensor | None = None,
                 use_focal: bool = False, w_dice: float = 1.0, w_ce: float = 1.0):
        super().__init__()
        self.dice = DiceLoss(n_classes)
        self.pixel = FocalLoss(weight=ce_weight) if use_focal else \
            nn.CrossEntropyLoss(weight=ce_weight)
        self.w_dice, self.w_ce = w_dice, w_ce

    def forward(self, logits, target):
        return self.w_dice * self.dice(logits, target) + self.w_ce * self.pixel(logits, target)


class MultiTaskLoss(nn.Module):
    """L = w_seg * SegLoss + w_grade * CE(grade)."""

    def __init__(self, n_seg_classes: int, seg_ce_weight=None, grade_weight=None,
                 w_seg: float = 1.0, w_grade: float = 0.5, use_focal: bool = False):
        super().__init__()
        self.seg = SegLoss(n_seg_classes, ce_weight=seg_ce_weight, use_focal=use_focal)
        self.grade = nn.CrossEntropyLoss(weight=grade_weight)
        self.w_seg, self.w_grade = w_seg, w_grade

    def forward(self, out: dict, seg_target, grade_target):
        l_seg = self.seg(out["seg_logits"], seg_target)
        l_grade = self.grade(out["grade_logits"], grade_target)
        total = self.w_seg * l_seg + self.w_grade * l_grade
        return total, {"loss": total.item(), "loss_seg": l_seg.item(),
                       "loss_grade": l_grade.item()}
