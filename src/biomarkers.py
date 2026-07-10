"""
biomarkers.py
=============
Biomarcadores tumorais quantitativos DERIVADOS DA SEGMENTAÇÃO.

Objetivo concreto (entregável ao professor): transformar a máscara predita em
medidas clínicas interpretáveis e usá-las como uma PONTE explícita seg->grau —
um classificador de grau transparente (baseado só em volume/geometria do tumor)
que serve de sanity-check e de baseline interpretável para a cabeça neural.

Sub-regiões (rótulos contíguos {0:fundo,1:NCR/NET,2:ED,3:ET}):
    WT (whole tumor)     = {1,2,3}
    TC (tumor core)      = {1,3}
    ET (enhancing tumor) = {3}
    NCR                  = {1}
    ED                   = {2}

Biomarcadores por caso:
    - volumes (em voxels ou mm^3 se `voxel_volume_mm3` for dado)
    - razões clínicas: ET/TC, TC/WT, fração de necrose (NCR/TC)
      -> ET alto e ET/TC alto correlacionam com alto grau (HGG).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence
import numpy as np

SUBREGIONS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,), "NCR": (1,), "ED": (2,)}


def _to_numpy(x) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def volume_biomarkers(mask, voxel_volume_mm3: float = 1.0) -> Dict[str, float]:
    """Biomarcadores de um caso. `mask` pode ser 2D [H,W] (uma fatia) ou 3D
    [S,H,W]/[H,W,S] (volume). Conta voxels por sub-região e deriva as razões.

    Se `voxel_volume_mm3` for informado (ex.: 1.0 no BraTS, já isotrópico 1mm),
    os volumes saem em mm^3; caso contrário, em voxels."""
    m = _to_numpy(mask).astype(np.int64)
    vox = {name: int(np.isin(m, cls).sum()) for name, cls in SUBREGIONS.items()}
    eps = 1e-6
    out: Dict[str, float] = {f"vol_{k}": v * voxel_volume_mm3 for k, v in vox.items()}
    out["ratio_ET_TC"] = vox["ET"] / (vox["TC"] + eps)
    out["ratio_TC_WT"] = vox["TC"] / (vox["WT"] + eps)
    out["frac_NCR_TC"] = vox["NCR"] / (vox["TC"] + eps)
    out["frac_ED_WT"] = vox["ED"] / (vox["WT"] + eps)
    out["has_ET"] = float(vox["ET"] > 0)
    return out


def case_biomarkers_from_slices(pred_masks, voxel_volume_mm3: float = 1.0) -> Dict[str, float]:
    """Agrega biomarcadores de um caso a partir de um stack de fatias preditas
    [S,H,W] (soma de voxels entre as fatias)."""
    return volume_biomarkers(pred_masks, voxel_volume_mm3)


def aggregate_biomarkers(
    case_masks: Dict[str, "np.ndarray"],
    grade_lookup: Optional[Dict[str, int]] = None,
    voxel_volume_mm3: float = 1.0,
) -> List[Dict]:
    """Constrói uma tabela (lista de dicts) de biomarcadores por caso, anexando o
    grau quando disponível. `case_masks[cid]` = máscara 2D/3D predita do caso."""
    rows = []
    for cid, mask in case_masks.items():
        rec = {"case_id": cid, **volume_biomarkers(mask, voxel_volume_mm3)}
        if grade_lookup is not None and cid in grade_lookup:
            rec["grade"] = int(grade_lookup[cid])
        rows.append(rec)
    return rows


def biomarker_grade_association(rows: List[Dict],
                                features: Optional[Sequence[str]] = None) -> List[Dict]:
    """Para cada biomarcador, mede a associação com o grau (binário):
        - AUC univariada (poder discriminativo isolado da feature)
        - correlação ponto-bisserial (sinal/força) + p-valor (se scipy)
    Requer que os `rows` tenham a chave 'grade'. Ordena por AUC desc."""
    rows = [r for r in rows if "grade" in r]
    if not rows:
        raise ValueError("nenhum caso com 'grade' — passe grade_lookup em aggregate_biomarkers")
    y = np.array([r["grade"] for r in rows])
    if features is None:
        features = [k for k in rows[0] if k not in ("case_id", "grade")]
    results = []
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        roc_auc_score = None
    try:
        from scipy.stats import pointbiserialr
    except ImportError:
        pointbiserialr = None
    for feat in features:
        x = np.array([r[feat] for r in rows], dtype=np.float64)
        rec = {"feature": feat}
        if roc_auc_score is not None and len(np.unique(y)) == 2 and np.ptp(x) > 0:
            try:
                auc = roc_auc_score(y, x)
                rec["auc"] = float(max(auc, 1 - auc))   # poder discriminativo
                rec["direction"] = "HGG↑" if auc >= 0.5 else "HGG↓"
            except ValueError:
                rec["auc"] = float("nan")
        else:
            rec["auc"] = float("nan")
        if pointbiserialr is not None and np.ptp(x) > 0 and len(np.unique(y)) == 2:
            r_pb, p = pointbiserialr(y, x)
            rec["r_pointbiserial"] = float(r_pb); rec["p_value"] = float(p)
        results.append(rec)
    results.sort(key=lambda d: (np.isnan(d.get("auc", float("nan"))), -(d.get("auc") or 0)))
    return results


def interpretable_grade_classifier(rows: List[Dict],
                                   features: Sequence[str] = ("vol_ET", "ratio_ET_TC", "vol_TC")):
    """Regressão logística SÓ sobre biomarcadores -> classificador de grau
    TRANSPARENTE (coeficientes lidos como importância clínica). Serve de baseline
    interpretável para comparar com a cabeça neural de graduação.

    Retorna dict com o modelo sklearn, AUC de validação cruzada e coeficientes."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score

    rows = [r for r in rows if "grade" in r]
    X = np.array([[r[f] for f in features] for r in rows], dtype=np.float64)
    y = np.array([r["grade"] for r in rows])
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000, class_weight="balanced"))
    n_splits = int(min(5, np.bincount(y).min())) if len(np.unique(y)) == 2 else 0
    cv_auc = float("nan")
    if n_splits >= 2:
        cv_auc = float(cross_val_score(clf, X, y, cv=n_splits, scoring="roc_auc").mean())
    clf.fit(X, y)
    coefs = dict(zip(features, clf.named_steps["logisticregression"].coef_[0].tolist()))
    return {"model": clf, "cv_auc": cv_auc, "coefficients": coefs, "features": list(features)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    case_masks, grades = {}, {}
    for i in range(30):
        hgg = i % 2 == 0
        m = np.zeros((16, 64, 64), np.int64)
        m[:, 20:44, 20:44] = 2; m[:, 26:38, 26:38] = 1
        if hgg:
            m[:, 30:34, 30:34] = 3   # ET presente -> HGG
        case_masks[f"c{i}"] = m; grades[f"c{i}"] = int(hgg)
    rows = aggregate_biomarkers(case_masks, grades)
    print("assoc top:", [(r["feature"], round(r["auc"], 2)) for r in biomarker_grade_association(rows)[:4]])
    res = interpretable_grade_classifier(rows)
    print("CV AUC biomarcadores:", round(res["cv_auc"], 3), "| coefs:", {k: round(v, 2) for k, v in res["coefficients"].items()})
