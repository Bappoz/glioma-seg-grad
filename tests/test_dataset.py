"""
Testes do pipeline de dados: z-score por volume, augmentation, grade proxy
patient-level, split estratificado, pré-computação -> shards e ingestão por CSV
(name_mapping.csv do BraTS2020). Roda em CPU com NIfTI sintético.
"""
import os
import csv
import shutil
import tempfile

import numpy as np
import torch
import nibabel as nib

from src import dataset as D

SHAPE = (24, 24, 12)


def _make_case(case_dir: str, cid: str, has_et: bool) -> None:
    os.makedirs(case_dir, exist_ok=True)
    rng = np.random.default_rng(abs(hash(cid)) % (2**32))
    for suf in ["flair", "t1", "t1ce", "t2"]:
        vol = (rng.random(SHAPE).astype(np.float32) * 100)
        nib.save(nib.Nifti1Image(vol, np.eye(4)), os.path.join(case_dir, f"{cid}_{suf}.nii"))
    seg = np.zeros(SHAPE, np.float32)
    seg[6:18, 6:18, 3:9] = 2      # edema
    seg[9:15, 9:15, 4:8] = 1      # core
    if has_et:
        seg[11:13, 11:13, 5:7] = 4   # ET
    nib.save(nib.Nifti1Image(seg, np.eye(4)), os.path.join(case_dir, f"{cid}_seg.nii"))


def _make_root(n=6):
    tmp = tempfile.mkdtemp()
    root = os.path.join(tmp, "BraTS2020")
    for k in range(n):
        _make_case(os.path.join(root, f"BraTS20_Training_{k:03d}"),
                   f"BraTS20_Training_{k:03d}", has_et=(k % 2 == 0))
    return tmp, root


def test_zscore_per_volume():
    vol = np.random.default_rng(0).random((10, 10, 5)).astype(np.float32) * 50
    vol[vol < 5] = 0
    out = D.zscore_norm(vol)
    brain = out[vol > 0]
    assert abs(float(brain.mean())) < 1e-3          # média ~0 no cérebro
    assert abs(float(brain.std()) - 1.0) < 0.05     # desvio ~1
    assert np.all(out[vol == 0] == 0)               # fundo zerado


def test_grade_proxy_patient_level():
    seg = np.zeros((8, 8, 4), np.int8)
    assert D.grade_proxy_from_seg(seg) == 0
    seg[1, 1, 1] = 3
    assert D.grade_proxy_from_seg(seg) == 1


def test_augmentation_preserves_alignment():
    img = torch.zeros(3, 16, 16); seg = torch.zeros(16, 16, dtype=torch.long)
    img[:, 2:6, 2:6] = 1.0; seg[2:6, 2:6] = 3      # marcador co-localizado
    aug = D.SegAugmentation(p_flip=1.0, p_rot90=1.0, intensity_std=0.0, p_gamma=0.0, seed=0)
    ia, sa = aug(img, seg)
    assert ia.shape == img.shape and sa.shape == seg.shape
    # onde a máscara marca ET, a imagem também foi transformada de forma consistente
    assert torch.all(ia[:, sa == 3] > 0.5)


def test_synthetic_grade_labels_match_getitem():
    ds = D.SyntheticBraTS(30, img_size=64)
    assert ds.grade_labels() == [int(ds[i]["grade"]) for i in range(30)]


def test_stratified_split_no_leakage():
    tmp, root = _make_root(6)
    try:
        gl = {f"BraTS20_Training_{k:03d}": (1 if k % 2 == 0 else 0) for k in range(6)}
        tr, va = D.split_cases(root, grade_lookup=gl, val_frac=0.34)
        assert not (set(tr) & set(va))              # sem vazamento paciente
        assert set(tr) | set(va) == set(gl)         # cobre todos
        assert any(gl[c] == 1 for c in va) and any(gl[c] == 0 for c in va)  # ambos graus no val
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_csv_grade_lookup_brats2020():
    tmp, root = _make_root(4)
    try:
        csvp = os.path.join(root, "name_mapping.csv")
        with open(csvp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["Grade", "BraTS_2020_subject_ID"])
            w.writeheader()
            for k in range(4):
                w.writerow({"Grade": "HGG" if k % 2 == 0 else "LGG",
                            "BraTS_2020_subject_ID": f"BraTS20_Training_{k:03d}"})
        gl = D.build_grade_lookup_from_csv(csvp, root=root)
        assert gl["BraTS20_Training_000"] == 1 and gl["BraTS20_Training_001"] == 0
        assert len(gl) == 4
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_precompute_and_npy_dataset():
    tmp, root = _make_root(6)
    try:
        gl = {f"BraTS20_Training_{k:03d}": (1 if k % 2 == 0 else 0) for k in range(6)}
        shard = os.path.join(tmp, "shards")
        D.precompute_slices(root, shard, slices_per_case=3, img_size=32,
                            grade_lookup=gl, verbose=False)
        nd = D.NpySliceDataset(shard)
        assert len(nd) > 0 and len(nd.case_ids()) == 6
        it = nd[0]
        assert tuple(it["image"].shape) == (3, 32, 32)
        assert tuple(it["seg_mask"].shape) == (32, 32)
        assert it["grade"].item() == gl[it["case_id"]]
        tr, va = D.make_loaders_precomputed(shard, batch_size=2, num_workers=0,
                                            balance_grade=True)
        b = next(iter(tr))
        assert b["image"].shape[1:] == (3, 32, 32)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_volume_cache_bounded():
    tmp, root = _make_root(4)
    try:
        ds = D.BraTSSliceDataset(root=root, slices_per_case=2, img_size=32, cache_volumes=6)
        _ = [ds[i] for i in range(len(ds))]
        assert len(ds._cache.store) <= 6            # LRU respeita o limite
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_find_brats_root_handles_kaggle_nesting_and_skips_validation():
    # simula o unzip do Kaggle: um nível extra de pasta-wrapper + uma pasta de
    # validação irmã (sem *_seg*) que NÃO deve ser confundida com o treino.
    tmp = tempfile.mkdtemp()
    try:
        wrapper = os.path.join(tmp, "BraTS2020_TrainingData")
        train_root = os.path.join(wrapper, "MICCAI_BraTS2020_TrainingData")
        val_root = os.path.join(wrapper, "MICCAI_BraTS2020_ValidationData")
        _make_case(os.path.join(train_root, "BraTS20_Training_001"),
                  "BraTS20_Training_001", has_et=True)
        # validação: mesmas modalidades, SEM seg (replica o formato oficial)
        val_case = os.path.join(val_root, "BraTS20_Validation_001")
        os.makedirs(val_case, exist_ok=True)
        for suf in ["flair", "t1", "t1ce", "t2"]:
            nib.save(nib.Nifti1Image(np.zeros(SHAPE, np.float32), np.eye(4)),
                     os.path.join(val_case, f"BraTS20_Validation_001_{suf}.nii"))

        found = D.find_brats_root(tmp)
        assert os.path.normpath(found) == os.path.normpath(train_root)
        assert len(D.discover_cases(found)) == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("OK:", name)
