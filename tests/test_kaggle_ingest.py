"""
Valida a ingestão de um mirror Kaggle do BraTS2018 (pastas HGG/LGG, nomenclatura
canônica _flair/_t1/_t1ce/_t2/_seg -> normalize_tcia=False).
"""
import os
import shutil
import tempfile

import nibabel as nib
import numpy as np

from src.dataset import build_grade_dataset_from_folders, BraTSSliceDataset, make_loaders

SHAPE = (8, 8, 4)
SUFFIXES = ["flair", "t1", "t1ce", "t2", "seg"]


def _make_case(case_dir: str, cid: str, n_classes: int = 4) -> None:
    os.makedirs(case_dir, exist_ok=True)
    rng = np.random.default_rng(abs(hash(cid)) % (2**32))
    for suf in SUFFIXES:
        if suf == "seg":
            vol = rng.integers(0, 2, size=SHAPE).astype(np.float32)
            vol[vol == 1] = 4.0  # rotulo ET valido no BraTS
        else:
            vol = rng.random(SHAPE).astype(np.float32) * 100
        img = nib.Nifti1Image(vol, affine=np.eye(4))
        nib.save(img, os.path.join(case_dir, f"{cid}_{suf}.nii.gz"))


def test_build_grade_dataset_from_kaggle_mock():
    tmp = tempfile.mkdtemp()
    try:
        hgg_dir = os.path.join(tmp, "HGG")
        lgg_dir = os.path.join(tmp, "LGG")
        out_root = os.path.join(tmp, "merged")

        _make_case(os.path.join(hgg_dir, "Brats18_HGG_001"), "Brats18_HGG_001")
        _make_case(os.path.join(hgg_dir, "Brats18_HGG_002"), "Brats18_HGG_002")
        _make_case(os.path.join(lgg_dir, "Brats18_LGG_001"), "Brats18_LGG_001")

        grade_lookup = build_grade_dataset_from_folders(
            [(hgg_dir, 1), (lgg_dir, 0)], out_root, normalize_tcia=False)

        assert grade_lookup == {
            "Brats18_HGG_001": 1,
            "Brats18_HGG_002": 1,
            "Brats18_LGG_001": 0,
        }

        ds = BraTSSliceDataset(root=out_root, slices_per_case=2, img_size=32,
                               grade_lookup=grade_lookup)
        assert len(ds) > 0

        item = ds[0]
        cid = item["case_id"]
        assert item["grade"].item() == grade_lookup[cid]
        assert tuple(item["image"].shape) == (3, 32, 32)
        assert tuple(item["seg_mask"].shape) == (32, 32)

        train_dl, val_dl = make_loaders(out_root, batch_size=1, img_size=32,
                                        slices_per_case=2, num_workers=0,
                                        grade_lookup=grade_lookup)
        assert len(train_dl) > 0
        assert len(val_dl) > 0
        batch = next(iter(train_dl))
        assert batch["image"].shape[1:] == (3, 32, 32)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_build_grade_dataset_from_kaggle_mock()
    print("OK: build_grade_dataset_from_folders + BraTSSliceDataset + make_loaders")
