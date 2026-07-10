"""
metrics.py
==========
Métricas de validação para RELATAR AO PROFESSOR.

Segmentação (por sub-região clínica BraTS: WT, TC, ET):
    - Dice Score  (principal métrica do desafio BraTS)
    - IoU / Jaccard
    - Sensibilidade (Recall/TPR) e Especificidade (TNR)
    - (opcional) Hausdorff95 — descomente se instalar medpy/scipy

Graduação:
    - Accuracy, Sensibilidade, Especificidade, e AUC (via sklearn se disponível)

As sub-regiões são derivadas dos rótulos contíguos {0=fundo,1=NCR,2=ED,3=ET}:
    WT = {1,2,3} | TC = {1,3} | ET = {3}
"""

from __future__ import annotations
from typing import Dict
import torch

SUBREGIONS = {
    "WT": (1, 2, 3),   # whole tumor
    "TC": (1, 3),      # tumor core
    "ET": (3,),        # enhancing tumor
}


def _binarize(mask: torch.Tensor, classes) -> torch.Tensor:
    out = torch.zeros_like(mask, dtype=torch.bool)
    for c in classes:
        out |= (mask == c)
    return out


@torch.no_grad()
def seg_scores(pred: torch.Tensor, target: torch.Tensor,
               eps: float = 1e-6) -> Dict[str, float]:
    """pred, target: LongTensor [B,H,W] com rótulos {0..3}.
    Retorna Dice/IoU/Sens/Espec por sub-região (média no batch)."""
    scores: Dict[str, float] = {}
    for name, cls in SUBREGIONS.items():
        p = _binarize(pred, cls)
        g = _binarize(target, cls)
        tp = (p & g).sum(dim=(1, 2)).float()
        fp = (p & ~g).sum(dim=(1, 2)).float()
        fn = (~p & g).sum(dim=(1, 2)).float()
        tn = (~p & ~g).sum(dim=(1, 2)).float()

        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        sens = (tp + eps) / (tp + fn + eps)          # recall
        spec = (tn + eps) / (tn + fp + eps)
        scores[f"dice_{name}"] = dice.mean().item()
        scores[f"iou_{name}"] = iou.mean().item()
        scores[f"sens_{name}"] = sens.mean().item()
        scores[f"spec_{name}"] = spec.mean().item()
    scores["dice_mean"] = sum(scores[f"dice_{n}"] for n in SUBREGIONS) / len(SUBREGIONS)
    return scores


@torch.no_grad()
def grade_scores(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """logits [B,K] ; target [B]. Accuracy + Sens/Espec (binário LGG/HGG)."""
    pred = logits.argmax(dim=1)
    acc = (pred == target).float().mean().item()
    out = {"grade_acc": acc}
    if logits.shape[1] == 2:
        tp = ((pred == 1) & (target == 1)).sum().float()
        fp = ((pred == 1) & (target == 0)).sum().float()
        fn = ((pred == 0) & (target == 1)).sum().float()
        tn = ((pred == 0) & (target == 0)).sum().float()
        out["grade_sens"] = (tp / (tp + fn + 1e-6)).item()
        out["grade_spec"] = (tn / (tn + fp + 1e-6)).item()
    return out


# ---------------------------------------------------------------------------
# Hausdorff 95 (HD95) — distância de borda robusta (percentil 95)
# ---------------------------------------------------------------------------
def _surface_distances(pred_mask, gt_mask, spacing=(1.0, 1.0)):
    """Distâncias das superfícies (bordas) de pred->gt e gt->pred, em mm.
    Usa transformada de distância euclidiana (scipy)."""
    from scipy.ndimage import distance_transform_edt, binary_erosion
    import numpy as np

    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return None  # indefinido se uma das máscaras é vazia

    # bordas = máscara menos sua erosão
    pred_border = pred ^ binary_erosion(pred)
    gt_border = gt ^ binary_erosion(gt)

    dt_gt = distance_transform_edt(~gt_border, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)

    d_pred_to_gt = dt_gt[pred_border]
    d_gt_to_pred = dt_pred[gt_border]
    return d_pred_to_gt, d_gt_to_pred


def hausdorff95(pred_mask, gt_mask, spacing=(1.0, 1.0)):
    """HD95 simétrico (mm). Percentil 95 das distâncias de borda em ambos os
    sentidos — menos sensível a outliers que o HD máximo puro.

    Retorna np.nan quando uma das máscaras é vazia (informe isso no relatório)."""
    import numpy as np
    d = _surface_distances(pred_mask, gt_mask, spacing)
    if d is None:
        return float("nan")
    d_pred_to_gt, d_gt_to_pred = d
    alld = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(alld, 95)) if alld.size else float("nan")


@torch.no_grad()
def seg_scores_hd95(pred: torch.Tensor, target: torch.Tensor,
                    spacing=(1.0, 1.0)) -> Dict[str, float]:
    """HD95 médio por sub-região (WT/TC/ET) no batch. Ignora casos vazios."""
    import numpy as np
    pred_np = pred.cpu().numpy()
    tgt_np = target.cpu().numpy()
    out: Dict[str, float] = {}
    for name, cls in SUBREGIONS.items():
        vals = []
        for b in range(pred_np.shape[0]):
            p = np.isin(pred_np[b], cls)
            g = np.isin(tgt_np[b], cls)
            hd = hausdorff95(p, g, spacing)
            if not np.isnan(hd):
                vals.append(hd)
        out[f"hd95_{name}"] = float(np.mean(vals)) if vals else float("nan")
    return out


# ---------------------------------------------------------------------------
# ROC / AUC para a graduação (probabilístico)
# ---------------------------------------------------------------------------
@torch.no_grad()
def grade_roc_auc(logits: torch.Tensor, target: torch.Tensor):
    """Curva ROC + AUC para graduação binária (LGG vs HGG).
    Retorna dict com fpr, tpr, thresholds (listas) e auc (float).
    Requer scikit-learn. Para multi-classe, calcula AUC macro one-vs-rest."""
    import numpy as np
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    y = target.cpu().numpy()
    try:
        from sklearn.metrics import roc_curve, roc_auc_score
    except ImportError:
        return {"auc": float("nan"), "error": "instale scikit-learn"}

    if probs.shape[1] == 2:
        if len(np.unique(y)) < 2:
            return {"auc": float("nan"), "fpr": [], "tpr": [], "thr": []}
        fpr, tpr, thr = roc_curve(y, probs[:, 1])
        auc = float(roc_auc_score(y, probs[:, 1]))
        return {"auc": auc, "fpr": fpr.tolist(), "tpr": tpr.tolist(),
                "thr": thr.tolist()}
    else:
        try:
            auc = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
        except ValueError:
            auc = float("nan")
        return {"auc": auc, "multiclass": True}


# ---------------------------------------------------------------------------
# Calibração da graduação (suporte à decisão clínica exige probabilidade honesta)
# ---------------------------------------------------------------------------
@torch.no_grad()
def reliability_bins(logits: torch.Tensor, target: torch.Tensor, n_bins: int = 10):
    """Agrupa as predições por confiança e mede a acurácia empírica em cada bin.
    Base do reliability diagram e do ECE. Retorna dict com listas por bin."""
    import numpy as np
    prob = torch.softmax(logits, dim=1).cpu().numpy()
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    y = target.cpu().numpy()
    edges = np.linspace(0, 1, n_bins + 1)
    bins = {"conf": [], "acc": [], "count": [], "edges": edges.tolist()}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            bins["conf"].append(float("nan")); bins["acc"].append(float("nan"))
            bins["count"].append(0); continue
        bins["conf"].append(float(conf[m].mean()))
        bins["acc"].append(float((pred[m] == y[m]).mean()))
        bins["count"].append(int(m.sum()))
    return bins


@torch.no_grad()
def expected_calibration_error(logits: torch.Tensor, target: torch.Tensor,
                               n_bins: int = 10) -> float:
    """ECE = média ponderada |confiança - acurácia| por bin. 0 = perfeitamente
    calibrado. Métrica-chave quando a probabilidade de grau vira apoio à decisão."""
    import numpy as np
    b = reliability_bins(logits, target, n_bins)
    n = sum(b["count"])
    if n == 0:
        return float("nan")
    ece = 0.0
    for c, a, cnt in zip(b["conf"], b["acc"], b["count"]):
        if cnt > 0 and not (np.isnan(c) or np.isnan(a)):
            ece += (cnt / n) * abs(c - a)
    return float(ece)


@torch.no_grad()
def brier_score(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Brier (binário) = MSE entre p(HGG) e o rótulo. Menor é melhor."""
    import numpy as np
    prob = torch.softmax(logits, dim=1).cpu().numpy()
    y = target.cpu().numpy()
    if prob.shape[1] != 2:
        oh = np.eye(prob.shape[1])[y]
        return float(((prob - oh) ** 2).sum(axis=1).mean())
    return float(((prob[:, 1] - y) ** 2).mean())


@torch.no_grad()
def grade_report(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Precision/Recall/F1 (macro) + matriz de confusão da graduação (sklearn)."""
    pred = logits.argmax(dim=1).cpu().numpy()
    y = target.cpu().numpy()
    out: Dict[str, float] = {}
    try:
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        p, r, f1, _ = precision_recall_fscore_support(
            y, pred, average="macro", zero_division=0)
        out.update({"precision_macro": float(p), "recall_macro": float(r),
                    "f1_macro": float(f1)})
        out["confusion"] = confusion_matrix(y, pred).tolist()
    except ImportError:
        out["error"] = "instale scikit-learn"
    return out


class MetricTracker:
    """Acumula médias ao longo de um epoch (para plotar depois)."""

    def __init__(self):
        self.sums: Dict[str, float] = {}
        self.n = 0

    def update(self, d: Dict[str, float], batch: int = 1):
        for k, v in d.items():
            if isinstance(v, (int, float)):
                self.sums[k] = self.sums.get(k, 0.0) + v * batch
        self.n += batch

    def average(self) -> Dict[str, float]:
        return {k: v / max(self.n, 1) for k, v in self.sums.items()}
