"""
Testes de biomarcadores: volumes/razões por sub-região, associação com o grau
(AUC univariada) e classificador logístico interpretável. CPU, rápido.
"""
import numpy as np

from src.biomarkers import (volume_biomarkers, aggregate_biomarkers,
                            biomarker_grade_association, interpretable_grade_classifier)


def test_biomarkers_and_association():
    m = np.zeros((8, 32, 32), np.int64)
    m[:, 8:24, 8:24] = 2; m[:, 12:20, 12:20] = 1; m[:, 14:18, 14:18] = 3
    bm = volume_biomarkers(m)
    assert bm["vol_ET"] > 0 and bm["vol_WT"] >= bm["vol_TC"] >= bm["vol_ET"]
    assert 0 <= bm["ratio_ET_TC"] <= 1

    case_masks, grades = {}, {}
    for i in range(24):
        hgg = i % 2 == 0
        mm = np.zeros((6, 32, 32), np.int64); mm[:, 8:24, 8:24] = 2; mm[:, 12:20, 12:20] = 1
        if hgg:
            mm[:, 14:18, 14:18] = 3
        case_masks[f"c{i}"] = mm; grades[f"c{i}"] = int(hgg)
    rows = aggregate_biomarkers(case_masks, grades)
    assoc = biomarker_grade_association(rows)
    top = assoc[0]
    assert top["auc"] > 0.9                                # vol_ET separa grau
    res = interpretable_grade_classifier(rows)
    assert res["cv_auc"] > 0.9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("OK:", name)
