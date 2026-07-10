"""
explain.py
==========
Explicabilidade da GRADUAÇÃO — entregável concreto: mostrar em CIMA da MRI qual
região sustentou a decisão de grau. Dois métodos complementares:

1. Atenção intrínseca (self-explaining): o `MaskGuidedAttnClassifier` já produz um
   mapa de atenção espacial usado no pooling. É a explicação "de graça", fiel ao
   que o modelo de fato agregou (restrito ao tumor segmentado).

2. Saliência por oclusão (Zeiler & Fergus, 2014): occlui patches da entrada e mede
   a queda na logit do grau predito. Model-agnostic, valida a atenção intrínseca.

Uso:
    from src.explain import grade_attention, occlusion_saliency
    attn = grade_attention(model, img[None])          # [1,H,W] em [0,1]
    sal  = occlusion_saliency(model, img[None])        # [1,H,W]
    # sobreponha com viz.overlay_heatmap(img, attn)
"""

from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F


def _normalize(a: torch.Tensor) -> torch.Tensor:
    a = a.float()
    flat = a.flatten(1)
    mn = flat.min(1).values.view(-1, *([1] * (a.dim() - 1)))
    mx = flat.max(1).values.view(-1, *([1] * (a.dim() - 1)))
    return (a - mn) / (mx - mn + 1e-8)


@torch.no_grad()
def grade_attention(model, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Mapa de atenção intrínseco da graduação, upsampled p/ [B,H,W] em [0,1].
    É a região que o masked-attention pooling ponderou ao decidir o grau."""
    was_training = model.training
    model.eval()
    out = model(x, return_aux=True)
    attn = out["grade_attn"]                                   # [B,1,h,w]
    attn = F.interpolate(attn, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    model.train(was_training)
    return _normalize(attn) if normalize else attn


@torch.no_grad()
def occlusion_saliency(model, x: torch.Tensor, patch: int = 16, stride: int = 8,
                       normalize: bool = True) -> torch.Tensor:
    """Saliência por oclusão para a classe de grau predita. Desliza um patch
    (preenchido com a média local) e mede quanto a logit do grau predito CAI —
    quanto maior a queda, mais aquela região sustenta a decisão.

    Retorna [B,H,W]. Custo ~ (H/stride)*(W/stride) forwards; use stride>=8."""
    was_training = model.training
    model.eval()
    B, C, H, W = x.shape
    base = model(x, return_aux=False)["grade_logits"]
    pred = base.argmax(1)                                      # [B]
    base_score = base.gather(1, pred[:, None])[:, 0]          # logit do grau predito
    sal = torch.zeros(B, H, W, device=x.device)
    cnt = torch.zeros(B, H, W, device=x.device)
    fill = x.mean(dim=(2, 3), keepdim=True)                    # média por canal
    for top in range(0, H, stride):
        for left in range(0, W, stride):
            b, r = min(top + patch, H), min(left + patch, W)
            xo = x.clone()
            xo[:, :, top:b, left:r] = fill
            score = model(xo, return_aux=False)["grade_logits"].gather(1, pred[:, None])[:, 0]
            drop = (base_score - score).clamp_min(0)          # queda na logit
            sal[:, top:b, left:r] += drop[:, None, None]
            cnt[:, top:b, left:r] += 1
    model.train(was_training)
    sal = sal / cnt.clamp_min(1)
    return _normalize(sal) if normalize else sal


@torch.no_grad()
def explain_case(model, x: torch.Tensor, patch: int = 16, stride: int = 8) -> Dict:
    """Pacote de explicação de um caso [1,3,H,W]: grau predito, prob, atenção e
    saliência por oclusão — pronto para a figura do relatório."""
    out = model(x, return_aux=True)
    prob = torch.softmax(out["grade_logits"], dim=1)[0]
    return {
        "grade_pred": int(prob.argmax()),
        "grade_prob": prob.detach().cpu(),
        "attention": grade_attention(model, x)[0].cpu(),
        "occlusion": occlusion_saliency(model, x, patch, stride)[0].cpu(),
    }


if __name__ == "__main__":
    from src.models import SAM2SegGradeNet
    net = SAM2SegGradeNet(backbone="stub")
    x = torch.randn(1, 3, 64, 64)
    a = grade_attention(net, x); s = occlusion_saliency(net, x, patch=16, stride=16)
    print("attention:", tuple(a.shape), "range", round(float(a.min()), 2), round(float(a.max()), 2))
    print("occlusion:", tuple(s.shape), "range", round(float(s.min()), 2), round(float(s.max()), 2))
    print("EXPLAIN OK")
