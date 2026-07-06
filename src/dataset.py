"""
dataset.py
==========
Carregamento e pré-processamento do BraTS (Brain Tumor Segmentation).

Convenções BraTS
----------------
Cada caso tem 4 modalidades MRI co-registradas (240x240x155):
    T1, T1ce (contraste), T2, FLAIR
E uma máscara `seg` com rótulos:
    0 = fundo/tecido saudável
    1 = necrose + tumor não-realçante (NCR/NET)
    2 = edema peritumoral (ED)
    4 = tumor ativo/realçante (ET)   <- note: rótulo 4, não 3

Sub-regiões clínicas avaliadas no desafio (derivadas dos rótulos):
    WT (Whole Tumor)     = 1 U 2 U 4
    TC (Tumor Core)      = 1 U 4
    ET (Enhancing Tumor) = 4

Aqui remapeamos para 4 canais contíguos {0,1,2,3} para a CrossEntropy:
    0 = fundo | 1 = NCR/NET | 2 = ED | 3 = ET

Graduação (Estágio B)
---------------------
BraTS histórico rotula LGG vs HGG por pasta. Se seu subset trouxer o grau OMS,
ajuste `grade_from_case`. Sem rótulo de grau, dá para usar um proxy fraco
(presença/volume de ET), mas avise o professor que é proxy, não ground-truth.
"""

from __future__ import annotations
import os
import glob
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import nibabel as nib
except ImportError:
    nib = None  # instale com: pip install nibabel


# ---------------------------------------------------------------------------
# Pré-processamento
# ---------------------------------------------------------------------------
def zscore_norm(vol: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalização z-score usando SÓ os voxels do cérebro (ignora fundo=0).
    Passo padrão em MRI, pois a intensidade não é calibrada entre scanners."""
    brain = vol[vol > 0]
    if brain.size == 0:
        return vol.astype(np.float32)
    mu, sd = brain.mean(), brain.std() + eps
    out = (vol - mu) / sd
    out[vol == 0] = 0.0
    return out.astype(np.float32)


def remap_labels(seg: np.ndarray) -> np.ndarray:
    """BraTS {0,1,2,4} -> contíguo {0,1,2,3}."""
    out = np.zeros_like(seg, dtype=np.int64)
    out[seg == 1] = 1
    out[seg == 2] = 2
    out[seg == 4] = 3
    return out


def pick_informative_slices(seg: np.ndarray, k: int = 32,
                            axis: int = 2) -> List[int]:
    """Escolhe as `k` fatias axiais com maior área tumoral. Reduz o desbalanço
    fundo/tumor e o custo de treino (evita centenas de fatias vazias)."""
    tumor_area = (seg > 0).sum(axis=tuple(i for i in range(3) if i != axis))
    order = np.argsort(tumor_area)[::-1]
    chosen = [int(i) for i in order[:k] if tumor_area[i] > 0]
    return sorted(chosen) if chosen else [seg.shape[axis] // 2]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BraTSSliceDataset(Dataset):
    """Dataset 2D por fatia axial.

    Cada item:
        image        : FloatTensor [3, H, W]  (3 modalidades -> pseudo-RGB p/ SAM2)
        seg_mask      : LongTensor  [H, W]     rótulos {0..3}
        grade         : LongTensor  []         0=LGG, 1=HGG (ou grau OMS)
        case_id, slice: metadados

    Parametros
    ----------
    root         : pasta com subpastas por caso (cada uma com *_t1ce.nii.gz etc.)
    modalities   : quais 3 modalidades empilhar como canais (SAM2 espera 3).
                   Default (FLAIR, T1ce, T2) é forte para tumor + edema.
    slices_per_case, img_size, transform, split.
    """

    MOD_SUFFIX = {"t1": "t1", "t1ce": "t1ce", "t2": "t2", "flair": "flair"}

    def __init__(
        self,
        root: str,
        modalities: Tuple[str, str, str] = ("flair", "t1ce", "t2"),
        slices_per_case: int = 32,
        img_size: int = 256,
        grade_lookup: Optional[Dict[str, int]] = None,
        transform: Optional[Callable] = None,
        case_ids: Optional[List[str]] = None,
    ):
        assert len(modalities) == 3, "SAM2 espera 3 canais -> passe 3 modalidades."
        self.root = root
        self.modalities = modalities
        self.img_size = img_size
        self.transform = transform
        self.grade_lookup = grade_lookup or {}

        self.cases = case_ids or self._discover_cases(root)
        # index plano: (case_id, slice_idx)
        self.index: List[Tuple[str, int]] = []
        self._seg_cache: Dict[str, np.ndarray] = {}
        for cid in self.cases:
            seg = self._load_seg(cid)
            for s in pick_informative_slices(seg, k=slices_per_case):
                self.index.append((cid, s))

    # ---- descoberta de arquivos ----
    def _discover_cases(self, root: str) -> List[str]:
        cases = []
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if os.path.isdir(d) and glob.glob(os.path.join(d, "*_flair*.nii*")):
                cases.append(os.path.basename(d))
        if not cases:
            raise FileNotFoundError(f"Nenhum caso BraTS encontrado em {root}")
        return cases

    def _path(self, cid: str, suffix: str) -> str:
        hits = glob.glob(os.path.join(self.root, cid, f"*_{suffix}.nii*"))
        if not hits:
            raise FileNotFoundError(f"{cid}: modalidade '{suffix}' ausente")
        return hits[0]

    def _load_vol(self, cid: str, suffix: str) -> np.ndarray:
        if nib is None:
            raise ImportError("nibabel nao instalado: pip install nibabel")
        return nib.load(self._path(cid, suffix)).get_fdata().astype(np.float32)

    def _load_seg(self, cid: str) -> np.ndarray:
        if cid not in self._seg_cache:
            self._seg_cache[cid] = self._load_vol(cid, "seg").astype(np.int64)
        return self._seg_cache[cid]

    # ---- grade ----
    def grade_from_case(self, cid: str, seg_slice: np.ndarray) -> int:
        if cid in self.grade_lookup:
            return int(self.grade_lookup[cid])
        # proxy fraco: presenca de tumor ativo (ET) sugere alto grau
        return 1 if (seg_slice == 3).sum() > 0 else 0

    # ---- API Dataset ----
    def __len__(self) -> int:
        return len(self.index)

    def _resize(self, arr: np.ndarray, is_mask: bool) -> np.ndarray:
        t = torch.from_numpy(arr)[None, None].float()
        mode = "nearest" if is_mask else "bilinear"
        kw = {} if is_mask else {"align_corners": False}
        t = torch.nn.functional.interpolate(t, size=(self.img_size, self.img_size),
                                            mode=mode, **kw)
        return t[0, 0].numpy()

    def __getitem__(self, i: int):
        cid, s = self.index[i]
        chans = []
        for m in self.modalities:
            vol = self._load_vol(cid, self.MOD_SUFFIX[m])
            chans.append(zscore_norm(vol[:, :, s]))
        img = np.stack(chans, axis=0)                       # [3,H,W]

        seg = remap_labels(self._load_seg(cid)[:, :, s])    # [H,W]
        grade = self.grade_from_case(cid, seg)

        # resize consistente
        img = np.stack([self._resize(c, is_mask=False) for c in img], axis=0)
        seg = self._resize(seg.astype(np.float32), is_mask=True).astype(np.int64)

        img_t = torch.from_numpy(img).float()
        seg_t = torch.from_numpy(seg).long()

        if self.transform is not None:
            img_t, seg_t = self.transform(img_t, seg_t)

        return {
            "image": img_t,
            "seg_mask": seg_t,
            "grade": torch.tensor(grade, dtype=torch.long),
            "case_id": cid,
            "slice": s,
        }


# ---------------------------------------------------------------------------
# Split por PACIENTE (evita vazamento entre treino/val)
# ---------------------------------------------------------------------------
def split_cases(root: str, val_frac: float = 0.2, seed: int = 42):
    import random
    cases = [os.path.basename(d) for d in sorted(glob.glob(os.path.join(root, "*")))
             if os.path.isdir(d)]
    rng = random.Random(seed)
    rng.shuffle(cases)
    n_val = max(1, int(len(cases) * val_frac))
    return cases[n_val:], cases[:n_val]      # train_ids, val_ids


def make_loaders(root: str, batch_size: int = 8, img_size: int = 256,
                 slices_per_case: int = 32, num_workers: int = 2,
                 grade_lookup: Optional[Dict[str, int]] = None):
    train_ids, val_ids = split_cases(root)
    common = dict(root=root, img_size=img_size, slices_per_case=slices_per_case,
                  grade_lookup=grade_lookup)
    train_ds = BraTSSliceDataset(case_ids=train_ids, **common)
    val_ds = BraTSSliceDataset(case_ids=val_ids, **common)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl


class SyntheticBraTS(Dataset):
    """Dataset sintético (sem baixar nada) para validar o pipeline no Colab.
    Gera blobs gaussianos como 'tumor'. Use antes de plugar o BraTS real."""

    def __init__(self, n: int = 64, img_size: int = 256, n_seg_classes: int = 4):
        self.n, self.s, self.k = n, img_size, n_seg_classes

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        img = torch.randn(3, self.s, self.s, generator=g)
        seg = torch.zeros(self.s, self.s, dtype=torch.long)
        yy, xx = torch.meshgrid(torch.arange(self.s), torch.arange(self.s), indexing="ij")
        cx, cy = torch.randint(60, self.s - 60, (2,), generator=g).tolist()
        # ~40% dos casos são "LGG-like": edema + core, SEM tumor ativo (ET).
        # Isso faz o rótulo de grau se dividir em 2 classes (LGG=0 / HGG=1),
        # essencial para a curva ROC/AUC de graduação funcionar no demo.
        is_lgg = (torch.rand(1, generator=g).item() < 0.4)
        rings = [(45, 2), (28, 1)] if is_lgg else [(45, 2), (28, 1), (14, 3)]
        for r, cls in rings:   # edema, core, (ET) aninhados
            m = ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
            seg[m] = cls
            img[:, m] += cls * 0.7
        grade = 1 if (seg == 3).sum() > 0 else 0
        return {"image": img, "seg_mask": seg,
                "grade": torch.tensor(grade), "case_id": f"synt{i}", "slice": 0}


if __name__ == "__main__":
    ds = SyntheticBraTS(n=4)
    b = ds[0]
    print("image   :", tuple(b["image"].shape))
    print("seg_mask:", tuple(b["seg_mask"].shape), "classes:", b["seg_mask"].unique().tolist())
    print("grade   :", b["grade"].item())
