"""
baseline.py
===========
Baseline e HARNESS DE ABLAÇÃO (rigor científico exigido pelo professor).

- `UNetSegGradeNet`: U-Net clássica (encoder treinado do zero) + a MESMA cabeça de
  graduação guiada por máscara. Serve para isolar o efeito do encoder de fundação:
  a única coisa que muda vs. o modelo principal é o extrator de features.

- `run_ablation`: treina N variantes com o mesmo Trainer/loaders e devolve uma
  tabela comparável (Dice WT/TC/ET, dice_mean, grade_acc, AUC, params treináveis).

Variantes típicas:
    unet            -> U-Net do zero (sem fundação)
    stub_frozen     -> encoder de fundação (stub) CONGELADO, sem adaptação
    stub_lora       -> encoder de fundação (stub) adaptado com LoRA
    sam2_lora       -> SAM2/MedSAM real + LoRA (quando o checkpoint estiver plugado)
"""

from __future__ import annotations
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import MaskGuidedAttnClassifier


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, p_drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.GELU(),
            nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(8, out_ch), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNetSegGradeNet(nn.Module):
    """U-Net 2D (do zero) para segmentação + graduação guiada por máscara.
    Mesma interface de saída de `SAM2SegGradeNet` -> pluga no mesmo Trainer."""

    def __init__(self, n_seg_classes: int = 4, n_grades: int = 2, feat_ch: int = 256,
                 base: int = 32, p_drop: float = 0.1):
        super().__init__()
        self.enc1 = DoubleConv(3, base, p_drop)
        self.enc2 = DoubleConv(base, base * 2, p_drop)
        self.enc3 = DoubleConv(base * 2, base * 4, p_drop)
        self.enc4 = DoubleConv(base * 4, base * 8, p_drop)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 8, feat_ch, p_drop)
        self.dec4 = DoubleConv(feat_ch + base * 8, base * 8, p_drop)
        self.dec3 = DoubleConv(base * 8 + base * 4, base * 4, p_drop)
        self.dec2 = DoubleConv(base * 4 + base * 2, base * 2, p_drop)
        self.dec1 = DoubleConv(base * 2 + base, base, p_drop)
        self.head = nn.Conv2d(base, n_seg_classes, 1)
        self.grader = MaskGuidedAttnClassifier(feat_ch, n_seg_classes, n_grades,
                                               p_drop=max(p_drop, 0.2))

    @staticmethod
    def _up(x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x: torch.Tensor, return_aux: bool = True) -> Dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        feat = self.bottleneck(self.pool(e4))                 # [B,feat_ch,H/16,W/16]
        d = self.dec4(self._up(feat, e4))
        d = self.dec3(self._up(d, e3))
        d = self.dec2(self._up(d, e2))
        d = self.dec1(self._up(d, e1))
        seg_logits = self.head(d)
        grade_logits, grade_attn, _ = self.grader(feat, seg_logits, return_attn=True)
        out = {"seg_logits": seg_logits, "grade_logits": grade_logits}
        if return_aux:
            out["grade_attn"] = grade_attn; out["feat"] = feat
        return out


# ---------------------------------------------------------------------------
# Harness de ablação
# ---------------------------------------------------------------------------
_VARIANTS = {
    "unet":        dict(backbone="unet", use_lora=False),
    "stub_frozen": dict(backbone="stub", use_lora=False, freeze_encoder=True),
    "stub_lora":   dict(backbone="stub", use_lora=True),
    "sam2_lora":   dict(backbone="sam2", use_lora=True, freeze_encoder=True),
}


def build_variant(cfg, variant: str) -> nn.Module:
    """Instancia o modelo de uma variante de ablação a partir do TrainConfig."""
    if variant not in _VARIANTS:
        raise ValueError(f"variante '{variant}' desconhecida: {list(_VARIANTS)}")
    if variant == "unet":
        return UNetSegGradeNet(n_seg_classes=cfg.n_seg_classes, n_grades=cfg.n_grades,
                               feat_ch=cfg.feat_ch, p_drop=getattr(cfg, "p_drop", 0.1))
    from .models import build_model
    over = _VARIANTS[variant]
    return build_model(replace(cfg, **over))


def _trainable_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def run_ablation(cfg, train_dl, val_dl, variants: List[str],
                 verbose: bool = True) -> List[Dict]:
    """Treina cada variante com o mesmo Trainer e devolve a tabela comparativa
    (última época de validação). `cfg.backbone` é sobrescrito por variante."""
    from .train import Trainer
    rows = []
    for v in variants:
        if verbose:
            print(f"\n===== ABLAÇÃO: {v} =====")
        model = build_variant(cfg, v)
        vcfg = replace(cfg, out_dir=f"{cfg.out_dir}/ablacao_{v}",
                       use_lora=_VARIANTS[v].get("use_lora", cfg.use_lora))
        trainer = Trainer(vcfg, model=model)   # injeta o modelo da variante
        hist = trainer.fit(train_dl, val_dl)
        va = hist["val"][-1]
        rows.append({
            "variant": v,
            "params_M": round(_trainable_params(trainer.model), 3),
            "dice_mean": round(va["dice_mean"], 4),
            "dice_WT": round(va["dice_WT"], 4),
            "dice_TC": round(va["dice_TC"], 4),
            "dice_ET": round(va["dice_ET"], 4),
            "grade_acc": round(va["grade_acc"], 4),
            "grade_auc": round(va.get("grade_auc", float("nan")), 4),
            "best_dice_mean": round(trainer.best, 4),
        })
    return rows


if __name__ == "__main__":
    net = UNetSegGradeNet()
    x = torch.randn(2, 3, 128, 128)
    out = net(x)
    print("UNet seg:", tuple(out["seg_logits"].shape), "grade:", tuple(out["grade_logits"].shape))
    print("params treinaveis:", round(_trainable_params(net), 2), "M")
    print("BASELINE OK")
