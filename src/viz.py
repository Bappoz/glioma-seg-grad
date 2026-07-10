"""
viz.py
======
Visualizações para a seção de RESULTADOS do relatório.

Funções (todas devolvem uma Figure do matplotlib, prontas para salvar):
    - overlay_mask         : sobrepõe a máscara colorida (necrose/edema/ET) na MRI
    - overlay_heatmap      : sobrepõe um mapa de calor (atenção/incerteza) na MRI
    - qualitative_panel    : painel lado-a-lado MRI | GT | Predição | Erro
    - plot_training_curves : loss + Dice(WT/TC/ET) + graduação por época
    - plot_roc             : curva ROC da graduação (com AUC)
    - plot_hd95_bars       : barras de HD95 por sub-região
    - plot_reliability     : reliability diagram + ECE (calibração da graduação)
    - plot_uncertainty_panel : MRI | predição | incerteza total | epistêmica (MC-Dropout)
    - plot_explanation_panel : MRI | atenção da graduação | saliência por oclusão
    - plot_biomarker_associations : AUC univariada de cada biomarcador vs grau
    - plot_biomarker_by_grade     : distribuição de um biomarcador por grau
    - plot_ablation        : comparação de variantes (Dice/AUC/params)

Convenção de rótulos (contígua): 0=fundo, 1=NCR/NET, 2=ED, 3=ET.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# cores fixas por classe (para consistência entre todas as figuras)
CLASS_COLORS = {
    0: (0.0, 0.0, 0.0, 0.0),      # fundo -> transparente
    1: (0.90, 0.10, 0.10, 0.55),  # NCR/NET -> vermelho
    2: (0.10, 0.60, 0.95, 0.45),  # edema   -> azul
    3: (1.00, 0.85, 0.10, 0.70),  # ET      -> amarelo
}
CLASS_NAMES = {1: "NCR/NET", 2: "Edema (ED)", 3: "Tumor ativo (ET)"}


def _to_numpy(x) -> np.ndarray:
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _norm_img(img: np.ndarray) -> np.ndarray:
    """Normaliza uma fatia MRI para [0,1] só para exibição."""
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, 1), np.percentile(img, 99)
    if hi - lo < 1e-6:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0, 1)


def _colorize(mask: np.ndarray) -> np.ndarray:
    """Converte máscara [H,W] de rótulos em RGBA [H,W,4]."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for cls, color in CLASS_COLORS.items():
        rgba[mask == cls] = color
    return rgba


def overlay_mask(ax, image2d, mask2d, title: str = ""):
    """Sobrepõe a máscara colorida na fatia MRI (grayscale) em um eixo dado."""
    img = _norm_img(_to_numpy(image2d))
    msk = _to_numpy(mask2d).astype(int)
    ax.imshow(img, cmap="gray")
    ax.imshow(_colorize(msk))
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    return ax


def qualitative_panel(
    image2d, gt_mask, pred_mask,
    grade_true: Optional[int] = None, grade_pred: Optional[int] = None,
    dice_txt: str = "", save_path: Optional[str] = None,
):
    """Painel 1x4: MRI | Ground Truth | Predição | Mapa de erro.

    O mapa de erro destaca falsos positivos (magenta) e falsos negativos (ciano)
    do tumor inteiro (WT), tornando visíveis os pontos fracos do modelo.
    """
    img = _norm_img(_to_numpy(image2d))
    gt = _to_numpy(gt_mask).astype(int)
    pr = _to_numpy(pred_mask).astype(int)

    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("MRI (FLAIR)", fontsize=11)
    overlay_mask(ax[1], img, gt, "Ground Truth")
    ptitle = "Predição"
    if grade_pred is not None:
        gp = {0: "LGG", 1: "HGG"}.get(int(grade_pred), str(grade_pred))
        gt_g = {0: "LGG", 1: "HGG"}.get(int(grade_true), "?") if grade_true is not None else "?"
        ptitle += f"  (grau pred={gp} / real={gt_g})"
    overlay_mask(ax[2], img, pr, ptitle)

    # mapa de erro (WT = qualquer tumor)
    gt_wt = gt > 0
    pr_wt = pr > 0
    err = np.zeros((*gt_wt.shape, 4), dtype=np.float32)
    err[pr_wt & ~gt_wt] = (1.0, 0.0, 1.0, 0.8)   # FP magenta
    err[~pr_wt & gt_wt] = (0.0, 1.0, 1.0, 0.8)   # FN ciano
    ax[3].imshow(img, cmap="gray"); ax[3].imshow(err)
    ax[3].set_title(f"Erro (FP=magenta, FN=ciano)\n{dice_txt}", fontsize=10)
    ax[3].axis("off")

    legend = [Patch(facecolor=CLASS_COLORS[c][:3], label=n) for c, n in CLASS_NAMES.items()]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_training_curves(history: Dict[str, List[dict]], save_path: Optional[str] = None):
    """3 painéis: loss (treino/val), Dice por sub-região, métricas de graduação."""
    tr, va = history["train"], history["val"]
    ep = range(1, len(tr) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))

    ax[0].plot(ep, [h["loss"] for h in tr], label="treino")
    ax[0].plot(ep, [h["loss"] for h in va], label="validação")
    ax[0].set_title("Loss total"); ax[0].set_xlabel("época"); ax[0].legend(); ax[0].grid(alpha=0.3)

    for reg in ("WT", "TC", "ET"):
        ax[1].plot(ep, [h[f"dice_{reg}"] for h in va], marker="o", ms=3, label=reg)
    ax[1].set_title("Dice — validação"); ax[1].set_xlabel("época")
    ax[1].set_ylim(0, 1); ax[1].legend(); ax[1].grid(alpha=0.3)

    ax[2].plot(ep, [h["grade_acc"] for h in va], label="accuracy")
    if "grade_sens" in va[0]:
        ax[2].plot(ep, [h["grade_sens"] for h in va], label="sensibilidade")
        ax[2].plot(ep, [h["grade_spec"] for h in va], label="especificidade")
    ax[2].set_title("Graduação — validação"); ax[2].set_xlabel("época")
    ax[2].set_ylim(0, 1); ax[2].legend(); ax[2].grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_roc(roc: Dict, save_path: Optional[str] = None):
    """Plota a curva ROC devolvida por metrics.grade_roc_auc."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if roc.get("fpr"):
        ax.plot(roc["fpr"], roc["tpr"], lw=2,
                label=f"AUC = {roc['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("1 - Especificidade (FPR)")
    ax.set_ylabel("Sensibilidade (TPR)")
    ax.set_title("ROC — Graduação (LGG vs HGG)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_hd95_bars(hd95: Dict[str, float], save_path: Optional[str] = None):
    """Barras de HD95 (mm) por sub-região. Menor é melhor."""
    regs = [k.replace("hd95_", "") for k in hd95 if k.startswith("hd95_")]
    vals = [hd95[f"hd95_{r}"] for r in regs]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(regs, vals, color=["#4C72B0", "#DD8452", "#C44E52"])
    ax.set_ylabel("HD95 (mm) — menor é melhor")
    ax.set_title("Hausdorff95 por sub-região")
    for b, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Mapas de calor (atenção / incerteza / saliência)
# ---------------------------------------------------------------------------
def overlay_heatmap(ax, image2d, heat2d, title: str = "", cmap: str = "inferno",
                    alpha: float = 0.5):
    """Sobrepõe um mapa de calor [0,1] na fatia MRI (grayscale) em um eixo dado."""
    img = _norm_img(_to_numpy(image2d))
    h = _to_numpy(heat2d).astype(np.float32)
    ax.imshow(img, cmap="gray")
    im = ax.imshow(h, cmap=cmap, alpha=alpha)
    ax.set_title(title, fontsize=11); ax.axis("off")
    return im


def plot_reliability(bins: Dict, ece: Optional[float] = None,
                     save_path: Optional[str] = None):
    """Reliability diagram: confiança média vs acurácia empírica por bin.
    A diagonal = calibração perfeita. Mostra o ECE quando fornecido."""
    conf = np.array(bins["conf"], dtype=float)
    acc = np.array(bins["acc"], dtype=float)
    cnt = np.array(bins["count"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="calibração perfeita")
    m = ~np.isnan(conf)
    ax.plot(conf[m], acc[m], "o-", color="#C44E52", lw=2, label="modelo")
    for c, a, n in zip(conf[m], acc[m], cnt[m]):
        ax.bar(c, a, width=0.03, color="#4C72B0", alpha=0.25)
    ax.set_xlabel("Confiança média (bin)"); ax.set_ylabel("Acurácia empírica")
    title = "Reliability diagram — Graduação"
    if ece is not None:
        title += f"   (ECE = {ece:.3f})"
    ax.set_title(title); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_uncertainty_panel(image2d, seg_pred, total_entropy, mutual_info,
                           save_path: Optional[str] = None):
    """Painel 1x4: MRI | Predição | Incerteza total | Incerteza epistêmica.
    A epistêmica (MC-Dropout) destaca ONDE o modelo hesita — normalmente as
    bordas das sub-regiões e casos fora da distribuição."""
    img = _norm_img(_to_numpy(image2d))
    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("MRI", fontsize=11); ax[0].axis("off")
    overlay_mask(ax[1], img, _to_numpy(seg_pred).astype(int), "Segmentação predita")
    im2 = overlay_heatmap(ax[2], img, total_entropy, "Incerteza total (entropia)", cmap="magma")
    im3 = overlay_heatmap(ax[3], img, mutual_info, "Incerteza epistêmica (MI)", cmap="viridis")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_explanation_panel(image2d, attention, occlusion,
                           grade_pred: Optional[int] = None,
                           grade_prob: Optional[float] = None,
                           save_path: Optional[str] = None):
    """Painel 1x3: MRI | Atenção da graduação | Saliência por oclusão.
    Evidencia QUAL região sustentou a decisão de grau (explicabilidade)."""
    img = _norm_img(_to_numpy(image2d))
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    sub = "MRI"
    if grade_pred is not None:
        g = {0: "LGG", 1: "HGG"}.get(int(grade_pred), str(grade_pred))
        sub = f"MRI  (grau predito = {g}" + (f", p={grade_prob:.2f})" if grade_prob is not None else ")")
    ax[0].imshow(img, cmap="gray"); ax[0].set_title(sub, fontsize=11); ax[0].axis("off")
    im1 = overlay_heatmap(ax[1], img, attention, "Atenção intrínseca (pooling)", cmap="jet")
    im2 = overlay_heatmap(ax[2], img, occlusion, "Saliência por oclusão", cmap="jet")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Biomarcadores e ablação
# ---------------------------------------------------------------------------
def plot_biomarker_associations(assoc: List[Dict], top: int = 8,
                                save_path: Optional[str] = None):
    """Barras horizontais da AUC univariada de cada biomarcador vs grau."""
    assoc = [a for a in assoc if not np.isnan(a.get("auc", float("nan")))][:top]
    feats = [a["feature"] for a in assoc][::-1]
    aucs = [a["auc"] for a in assoc][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(feats) + 1.5))
    bars = ax.barh(feats, aucs, color="#55A868")
    ax.axvline(0.5, color="gray", ls="--", lw=1)
    ax.set_xlim(0.4, 1.0); ax.set_xlabel("AUC univariada (grau)")
    ax.set_title("Poder discriminativo dos biomarcadores")
    for b, v in zip(bars, aucs):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_biomarker_by_grade(rows: List[Dict], feature: str = "vol_ET",
                            save_path: Optional[str] = None):
    """Boxplot de um biomarcador separado por grau (LGG vs HGG)."""
    rows = [r for r in rows if "grade" in r]
    groups = {0: [], 1: []}
    for r in rows:
        groups[int(r["grade"])].append(r[feature])
    fig, ax = plt.subplots(figsize=(5, 5))
    data = [groups[0], groups[1]]
    bp = ax.boxplot(data, patch_artist=True, showmeans=True)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["LGG", "HGG"])
    for patch, c in zip(bp["boxes"], ["#4C72B0", "#C44E52"]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_ylabel(feature); ax.set_title(f"{feature} por grau")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_ablation(rows: List[Dict], save_path: Optional[str] = None):
    """Barras agrupadas comparando variantes: Dice médio e AUC de graduação,
    anotando os parâmetros treináveis (M) de cada uma."""
    variants = [r["variant"] for r in rows]
    dice = [r["dice_mean"] for r in rows]
    auc = [r.get("grade_auc", float("nan")) for r in rows]
    params = [r.get("params_M", float("nan")) for r in rows]
    x = np.arange(len(variants)); w = 0.38
    fig, ax = plt.subplots(figsize=(1.8 * len(variants) + 3, 5))
    b1 = ax.bar(x - w / 2, dice, w, label="Dice médio", color="#4C72B0")
    b2 = ax.bar(x + w / 2, auc, w, label="AUC graduação", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(variants)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Métrica (↑ melhor)")
    ax.set_title("Ablação — encoder de fundação vs baseline")
    for xi, p in zip(x, params):
        ax.text(xi, 1.0, f"{p:.2f}M", ha="center", va="bottom", fontsize=8, color="gray")
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig
