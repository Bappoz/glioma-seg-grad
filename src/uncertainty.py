"""
uncertainty.py
==============
Estimativa de incerteza — entregável concreto que agrega valor clínico:
"ONDE o modelo não confia na segmentação?" e "quão confiável é o grau?".

Dois mecanismos complementares (ambos rodam na T4, sem re-treino):

1. MC-Dropout (Gal & Ghahramani, 2016): mantém o dropout ATIVO na inferência e
   faz N passagens estocásticas. A variância entre passagens estima a incerteza
   EPISTÊMICA (do modelo). Decompomos:
       - entropia preditiva  H[E[p]]              (incerteza total)
       - entropia esperada   E[H[p]]              (aleatória/dados)
       - informação mútua    H[E[p]] - E[H[p]]    (epistêmica -> mapa de "dúvida")

2. TTA (test-time augmentation): média das predições sobre flips. Determinístico,
   costuma AUMENTAR o Dice e suavizar as bordas.

Requer dropout > 0 no modelo (TrainConfig.p_drop). Sem dropout, MC-Dropout vira
uma predição determinística (variância ~0) — use TTA nesse caso.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


def enable_mc_dropout(model: nn.Module) -> int:
    """Coloca SÓ as camadas de dropout em modo treino (o resto fica em eval).
    Retorna quantas camadas foram ativadas."""
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train(); n += 1
    return n


def predictive_entropy(prob: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """Entropia de Shannon ao longo das classes. `prob` [B,K,H,W] -> [B,H,W]."""
    return -(prob.clamp_min(eps) * prob.clamp_min(eps).log()).sum(dim=dim)


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: torch.Tensor, n_samples: int = 10,
                       verbose: bool = False) -> Dict[str, torch.Tensor]:
    """N passagens com dropout ativo. Retorna médias + mapas de incerteza.

    Chaves:
        seg_prob   [B,K,H,W] média das probabilidades de segmentação
        seg_pred   [B,H,W]   argmax da média
        seg_total_entropy [B,H,W]  incerteza total (H[E[p]])
        seg_mutual_info   [B,H,W]  incerteza EPISTÊMICA (mapa de dúvida do modelo)
        seg_sample_std    [B,H,W]  desvio-padrão de p(tumor) ENTRE as N passagens
                                   (diagnóstico: se ~0, o dropout não variou)
        grade_prob [B,G]     probabilidade média de grau
        grade_std  [B]       desvio-padrão de p(classe positiva) entre passagens
        n_dropout_active   int  quantas camadas de Dropout ficaram ativas

    `verbose=True` imprime um diagnóstico (camadas ativas + variabilidade entre
    passagens) — serve para confirmar que a incerteza baixa é genuína (modelo
    confiante) e não um bug de dropout inerte."""
    was_training = model.training
    model.eval()
    n_active = enable_mc_dropout(model)
    seg_probs, grade_probs = [], []
    entropies = []
    for _ in range(n_samples):
        out = model(x, return_aux=False)
        sp = torch.softmax(out["seg_logits"], dim=1)
        seg_probs.append(sp)
        entropies.append(predictive_entropy(sp))
        grade_probs.append(torch.softmax(out["grade_logits"], dim=1))
    model.train(was_training)

    seg_stack = torch.stack(seg_probs)                        # [N,B,K,H,W]
    seg_mean = seg_stack.mean(0)                              # [B,K,H,W]
    total_entropy = predictive_entropy(seg_mean)              # H[E[p]]
    expected_entropy = torch.stack(entropies).mean(0)         # E[H[p]]
    mutual_info = (total_entropy - expected_entropy).clamp_min(0)
    # variabilidade bruta entre passagens (p/ diagnóstico): desvio-padrão de
    # p(tumor)=1-p(fundo) ao longo das N amostras. Se for ~0, o dropout não está
    # perturbando o forward (bug) — se >0 mas a MI ainda é baixa, é confiança real.
    seg_sample_std = seg_stack[:, :, 0].std(0)                # [B,H,W]
    grade_stack = torch.stack(grade_probs)                    # [N,B,G]
    grade_mean = grade_stack.mean(0)
    pos = grade_stack[..., -1] if grade_stack.shape[-1] == 2 else grade_stack.max(-1).values
    grade_std = pos.std(0)
    if verbose:
        print(f"[MC-Dropout] camadas de Dropout ativas: {n_active} | "
              f"passagens: {n_samples} | "
              f"std entre passagens (seg): {float(seg_sample_std.mean()):.5f} | "
              f"std entre passagens (grau): {float(grade_std.mean()):.5f}")
        if n_active == 0:
            print("  ⚠ nenhuma camada de Dropout ativa — MC-Dropout degenera "
                  "para predição determinística (aumente p_drop no TrainConfig).")
    return {"seg_prob": seg_mean, "seg_pred": seg_mean.argmax(1),
            "seg_total_entropy": total_entropy, "seg_mutual_info": mutual_info,
            "seg_sample_std": seg_sample_std,
            "grade_prob": grade_mean, "grade_std": grade_std,
            "n_dropout_active": n_active}


_TTA_OPS = ("id", "flip_h", "flip_v", "flip_hv")


def _apply(x: torch.Tensor, op: str) -> torch.Tensor:
    if op == "flip_h":  return torch.flip(x, dims=[-1])
    if op == "flip_v":  return torch.flip(x, dims=[-2])
    if op == "flip_hv": return torch.flip(x, dims=[-2, -1])
    return x


@torch.no_grad()
def tta_predict(model: nn.Module, x: torch.Tensor,
                ops: Sequence[str] = _TTA_OPS) -> Dict[str, torch.Tensor]:
    """Média determinística das predições sobre flips (inversos aplicados à saída
    de segmentação). Costuma elevar o Dice e regularizar bordas."""
    was_training = model.training
    model.eval()
    seg_accum, grade_accum = None, None
    for op in ops:
        out = model(_apply(x, op), return_aux=False)
        sp = _apply(torch.softmax(out["seg_logits"], dim=1), op)  # desfaz o flip
        gp = torch.softmax(out["grade_logits"], dim=1)
        seg_accum = sp if seg_accum is None else seg_accum + sp
        grade_accum = gp if grade_accum is None else grade_accum + gp
    model.train(was_training)
    k = len(ops)
    seg_prob = seg_accum / k
    return {"seg_prob": seg_prob, "seg_pred": seg_prob.argmax(1),
            "grade_prob": grade_accum / k,
            "seg_entropy": predictive_entropy(seg_prob)}


@torch.no_grad()
def uncertainty_summary(unc: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Reduz os mapas a escalares para tabelas do relatório (média por batch)."""
    out = {}
    if "seg_mutual_info" in unc:
        out["mean_epistemic"] = float(unc["seg_mutual_info"].mean())
    if "seg_total_entropy" in unc:
        out["mean_total_entropy"] = float(unc["seg_total_entropy"].mean())
    if "seg_sample_std" in unc:
        out["mean_sample_std"] = float(unc["seg_sample_std"].mean())
    if "grade_std" in unc:
        out["mean_grade_std"] = float(unc["grade_std"].mean())
    return out


if __name__ == "__main__":
    from src.models import SAM2SegGradeNet
    net = SAM2SegGradeNet(backbone="stub", p_drop=0.2)
    x = torch.randn(2, 3, 128, 128)
    mc = mc_dropout_predict(net, x, n_samples=5)
    print("MC seg_prob:", tuple(mc["seg_prob"].shape),
          "| epistemic mean:", round(float(mc["seg_mutual_info"].mean()), 5),
          "| grade_std:", [round(v, 3) for v in mc["grade_std"].tolist()])
    tta = tta_predict(net, x)
    print("TTA seg_pred:", tuple(tta["seg_pred"].shape),
          "| entropy mean:", round(float(tta["seg_entropy"].mean()), 4))
