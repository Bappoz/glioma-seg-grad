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
BraTS histórico (2018 e anteriores) rotula LGG vs HGG por pasta. Mirrors do
BraTS2020 costumam trazer o grau em `name_mapping.csv` (coluna `Grade`) —
`build_grade_lookup_from_csv` lê isso, mas a coluna nem sempre está presente
(varia por mirror); sem rótulo de grau, usa-se um proxy fraco a NÍVEL DE
PACIENTE (presença de ET no volume inteiro) — nunca por fatia, para não gerar
rótulos inconsistentes dentro do mesmo paciente.

Caminhos de eficiência
----------------------
- `BraTSSliceDataset`: leitura on-the-fly com z-score POR VOLUME e cache LRU
  limitado (correto e cômodo para subsets pequenos).
- `precompute_slices` + `NpySliceDataset`: extrai as fatias informativas UMA vez
  para shards `.npz` compactos; o treino na T4 passa a ler arquivos minúsculos
  (sem decodificar NIfTI no loop) — muito mais rápido e com RAM previsível.
"""

from __future__ import annotations
import os
import glob
import json
import csv
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import nibabel as nib
except ImportError:
    nib = None  # instale com: pip install nibabel

# rótulos contíguos -> nomes (fonte única de verdade p/ viz/metrics)
CLASS_NAMES = {0: "fundo", 1: "NCR/NET", 2: "ED", 3: "ET"}
GRADE_NAMES = {0: "LGG", 1: "HGG"}


# ---------------------------------------------------------------------------
# Pré-processamento
# ---------------------------------------------------------------------------
def zscore_norm(vol: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalização z-score usando SÓ os voxels do cérebro (ignora fundo=0).

    Aplica sobre o array recebido — passe o VOLUME 3D inteiro (não uma fatia) para
    obter estatísticas estáveis por cérebro, já que a intensidade MRI não é
    calibrada entre scanners."""
    brain = vol[vol > 0]
    if brain.size == 0:
        return vol.astype(np.float32)
    mu, sd = brain.mean(), brain.std() + eps
    out = (vol - mu) / sd
    out[vol == 0] = 0.0
    return out.astype(np.float32)


def remap_labels(seg: np.ndarray) -> np.ndarray:
    """BraTS {0,1,2,4} -> contíguo {0,1,2,3} (int8, econômico em memória)."""
    out = np.zeros_like(seg, dtype=np.int8)
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


def grade_proxy_from_seg(seg_vol: np.ndarray) -> int:
    """Proxy fraco de grau a NÍVEL DE PACIENTE: presença de tumor ativo (ET=3)
    em qualquer fatia do volume sugere alto grau (HGG). Determinístico e
    consistente por paciente (nunca varia entre fatias do mesmo caso)."""
    return 1 if (seg_vol == 3).sum() > 0 else 0


# ---------------------------------------------------------------------------
# Augmentation (espacial + intensidade) — aplicada só no treino
# ---------------------------------------------------------------------------
class SegAugmentation:
    """Augmentation leve e barata para MRI 2D. Espaciais são aplicadas
    IGUALMENTE a imagem e máscara; intensidade só à imagem.

    - flip horizontal/vertical
    - rotação múltipla de 90°
    - jitter de intensidade (escala + deslocamento gaussiano)
    - gamma aleatório (contraste não-linear)
    """

    def __init__(self, p_flip: float = 0.5, p_rot90: float = 0.5,
                 intensity_std: float = 0.1, p_gamma: float = 0.3,
                 gamma_range: Tuple[float, float] = (0.7, 1.5), seed: Optional[int] = None):
        self.p_flip = p_flip
        self.p_rot90 = p_rot90
        self.intensity_std = intensity_std
        self.p_gamma = p_gamma
        self.gamma_range = gamma_range
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def _rand(self) -> float:
        return torch.rand(1, generator=self.rng).item()

    def __call__(self, img: torch.Tensor, seg: torch.Tensor):
        # img [C,H,W], seg [H,W]
        if self._rand() < self.p_flip:
            img = torch.flip(img, dims=[2]); seg = torch.flip(seg, dims=[1])
        if self._rand() < self.p_flip:
            img = torch.flip(img, dims=[1]); seg = torch.flip(seg, dims=[0])
        if self._rand() < self.p_rot90:
            k = int(torch.randint(1, 4, (1,), generator=self.rng).item())
            img = torch.rot90(img, k, dims=[1, 2]); seg = torch.rot90(seg, k, dims=[0, 1])
        if self.intensity_std > 0:
            scale = 1.0 + self.intensity_std * torch.randn(1, generator=self.rng).item()
            shift = self.intensity_std * torch.randn(1, generator=self.rng).item()
            img = img * scale + shift
        if self._rand() < self.p_gamma:
            lo, hi = self.gamma_range
            g = lo + (hi - lo) * self._rand()
            mn = img.amin()
            rng = (img.amax() - mn).clamp_min(1e-6)
            img = ((img - mn) / rng).clamp(0, 1).pow(g) * rng + mn
        return img.contiguous(), seg.contiguous()


# ---------------------------------------------------------------------------
# Dataset on-the-fly (z-score por volume + cache LRU limitado)
# ---------------------------------------------------------------------------
class _VolumeLRU:
    """Cache LRU simples keyed por (case_id, suffix). Guarda volumes normalizados
    em float16 (modalidades) / int8 (seg) para limitar RAM."""

    def __init__(self, maxsize: int = 24):
        self.maxsize = maxsize
        self.store: "OrderedDict[Tuple[str, str], np.ndarray]" = OrderedDict()

    def get(self, key):
        if key in self.store:
            self.store.move_to_end(key)
            return self.store[key]
        return None

    def put(self, key, value):
        self.store[key] = value
        self.store.move_to_end(key)
        while len(self.store) > self.maxsize:
            self.store.popitem(last=False)


class BraTSSliceDataset(Dataset):
    """Dataset 2D por fatia axial (leitura on-the-fly).

    Cada item:
        image        : FloatTensor [3, H, W]  (3 modalidades -> pseudo-RGB p/ SAM2)
        seg_mask      : LongTensor  [H, W]     rótulos {0..3}
        grade         : LongTensor  []         0=LGG, 1=HGG (ou grau OMS)
        case_id, slice: metadados

    Correções vs. versão anterior:
        - z-score calculado no VOLUME inteiro (não por fatia).
        - grade proxy a nível de paciente (consistente entre fatias).
        - cache LRU limitado (evita OOM ao cachear todos os volumes).
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
        cache_volumes: int = 24,
    ):
        assert len(modalities) == 3, "SAM2 espera 3 canais -> passe 3 modalidades."
        self.root = root
        self.modalities = modalities
        self.img_size = img_size
        self.transform = transform
        self.grade_lookup = grade_lookup or {}

        self.cases = case_ids or discover_cases(root)
        self._cache = _VolumeLRU(maxsize=max(cache_volumes, 4))
        self.index: List[Tuple[str, int]] = []
        self._grade: Dict[str, int] = {}
        for cid in self.cases:
            seg = self._load_seg(cid)
            self._grade[cid] = self._resolve_grade(cid, seg)
            for s in pick_informative_slices(seg, k=slices_per_case):
                self.index.append((cid, s))
        if not self.index:
            raise RuntimeError(f"Nenhuma fatia informativa encontrada em {root}")

    # ---- descoberta / io ----
    def _path(self, cid: str, suffix: str) -> str:
        hits = sorted(glob.glob(os.path.join(self.root, cid, f"*_{suffix}.nii*")))
        if not hits and suffix == "seg":
            # BraTS2020 tem 1 caso com a máscara fora do padrão (*Segm*.nii).
            for pat in ("*[Ss]egm*.nii*", "*[Ss]eg*.nii*"):
                hits = sorted(glob.glob(os.path.join(self.root, cid, pat)))
                if hits:
                    break
        if not hits:
            raise FileNotFoundError(f"{cid}: modalidade '{suffix}' ausente")
        return hits[0]

    def _load_norm_vol(self, cid: str, suffix: str) -> np.ndarray:
        key = (cid, suffix)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if nib is None:
            raise ImportError("nibabel nao instalado: pip install nibabel")
        raw = nib.load(self._path(cid, suffix)).get_fdata().astype(np.float32)
        vol = zscore_norm(raw).astype(np.float16)   # normaliza o volume inteiro
        self._cache.put(key, vol)
        return vol

    def _load_seg(self, cid: str) -> np.ndarray:
        key = (cid, "seg")
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if nib is None:
            raise ImportError("nibabel nao instalado: pip install nibabel")
        raw = nib.load(self._path(cid, "seg")).get_fdata()
        seg = remap_labels(np.asarray(raw))
        self._cache.put(key, seg)
        return seg

    def _resolve_grade(self, cid: str, seg_vol: np.ndarray) -> int:
        if cid in self.grade_lookup:
            return int(self.grade_lookup[cid])
        return grade_proxy_from_seg(seg_vol)

    # ---- API Dataset ----
    def __len__(self) -> int:
        return len(self.index)

    def _resize(self, arr: np.ndarray, is_mask: bool) -> np.ndarray:
        t = torch.from_numpy(np.ascontiguousarray(arr))[None, None].float()
        mode = "nearest" if is_mask else "bilinear"
        kw = {} if is_mask else {"align_corners": False}
        t = torch.nn.functional.interpolate(t, size=(self.img_size, self.img_size),
                                            mode=mode, **kw)
        return t[0, 0].numpy()

    def __getitem__(self, i: int):
        cid, s = self.index[i]
        chans = [self._load_norm_vol(cid, self.MOD_SUFFIX[m])[:, :, s].astype(np.float32)
                 for m in self.modalities]
        img = np.stack([self._resize(c, is_mask=False) for c in chans], axis=0)

        seg2d = self._load_seg(cid)[:, :, s].astype(np.float32)
        seg = self._resize(seg2d, is_mask=True).astype(np.int64)

        img_t = torch.from_numpy(img).float()
        seg_t = torch.from_numpy(seg).long()
        if self.transform is not None:
            img_t, seg_t = self.transform(img_t, seg_t)

        return {
            "image": img_t,
            "seg_mask": seg_t,
            "grade": torch.tensor(self._grade[cid], dtype=torch.long),
            "case_id": cid,
            "slice": s,
        }

    def grade_labels(self) -> List[int]:
        """Grau (0/1) de cada item do índice — útil p/ amostragem balanceada."""
        return [self._grade[cid] for cid, _ in self.index]


# ---------------------------------------------------------------------------
# Pré-computação de fatias -> shards .npz (caminho rápido para a T4)
# ---------------------------------------------------------------------------
def precompute_slices(
    root: str,
    out_dir: str,
    modalities: Tuple[str, str, str] = ("flair", "t1ce", "t2"),
    slices_per_case: int = 32,
    img_size: int = 256,
    grade_lookup: Optional[Dict[str, int]] = None,
    case_ids: Optional[List[str]] = None,
    verbose: bool = True,
) -> str:
    """Extrai as fatias informativas de cada caso UMA vez e grava `.npz` compactos
    (`img` float16 [3,H,W], `seg` int8 [H,W], `grade`, `case_id`). Escreve também
    `index.json`. Retorna `out_dir`. Leia depois com `NpySliceDataset`.

    Vantagem: o treino deixa de decodificar NIfTI a cada passo (gargalo na T4) e o
    uso de RAM fica previsível (cada shard é minúsculo)."""
    os.makedirs(out_dir, exist_ok=True)
    ds = BraTSSliceDataset(root, modalities=modalities, slices_per_case=slices_per_case,
                           img_size=img_size, grade_lookup=grade_lookup,
                           case_ids=case_ids, cache_volumes=8)
    index = []
    for i in range(len(ds)):
        item = ds[i]
        cid, s = item["case_id"], int(item["slice"])
        fname = f"{cid}__s{s:03d}.npz"
        fpath = os.path.join(out_dir, fname)
        np.savez_compressed(
            fpath,
            img=item["image"].numpy().astype(np.float16),
            seg=item["seg_mask"].numpy().astype(np.int8),
            grade=np.int8(int(item["grade"])),
        )
        index.append({"file": fname, "case_id": cid, "slice": s,
                      "grade": int(item["grade"])})
        if verbose and (i + 1) % 100 == 0:
            print(f"  precompute {i + 1}/{len(ds)} fatias...")
    meta = {"modalities": list(modalities), "img_size": img_size,
            "n_slices": len(index), "index": index}
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(meta, f)
    if verbose:
        print(f"OK: {len(index)} fatias -> {out_dir}")
    return out_dir


class NpySliceDataset(Dataset):
    """Lê shards `.npz` produzidos por `precompute_slices`. Rápido e leve."""

    def __init__(self, shard_dir: str, transform: Optional[Callable] = None,
                 case_ids: Optional[Sequence[str]] = None):
        self.dir = shard_dir
        self.transform = transform
        with open(os.path.join(shard_dir, "index.json")) as f:
            meta = json.load(f)
        idx = meta["index"]
        if case_ids is not None:
            keep = set(case_ids)
            idx = [e for e in idx if e["case_id"] in keep]
        if not idx:
            raise RuntimeError(f"Nenhum shard em {shard_dir} p/ os case_ids dados.")
        self.entries = idx
        self.img_size = meta.get("img_size")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        with np.load(os.path.join(self.dir, e["file"])) as z:
            img = torch.from_numpy(z["img"].astype(np.float32))
            seg = torch.from_numpy(z["seg"].astype(np.int64))
            grade = int(z["grade"])
        if self.transform is not None:
            img, seg = self.transform(img, seg)
        return {"image": img, "seg_mask": seg,
                "grade": torch.tensor(grade, dtype=torch.long),
                "case_id": e["case_id"], "slice": e["slice"]}

    def grade_labels(self) -> List[int]:
        return [e["grade"] for e in self.entries]

    def case_ids(self) -> List[str]:
        return sorted({e["case_id"] for e in self.entries})


# ---------------------------------------------------------------------------
# Descoberta / Split por PACIENTE (evita vazamento) — estratificado por grau
# ---------------------------------------------------------------------------
def discover_cases(root: str) -> List[str]:
    """Casos = subpastas com pelo menos um `*_flair*.nii*`."""
    cases = [os.path.basename(d) for d in sorted(glob.glob(os.path.join(root, "*")))
             if os.path.isdir(d) and glob.glob(os.path.join(d, "*_flair*.nii*"))]
    if not cases:
        raise FileNotFoundError(f"Nenhum caso BraTS encontrado em {root}")
    return cases


def find_brats_root(base: str, max_depth: int = 4, require_seg: bool = True) -> str:
    """Localiza automaticamente a raiz do BraTS sob `base` (ex.: `/content`),
    robusto a variações de nesting do `unzip` (o Kaggle às vezes envolve os
    dados num nível extra de pasta, e o nome exato do wrapper varia por mirror).

    Retorna o primeiro diretório cujos filhos incluem ao menos um caso completo
    (subpasta com `*_flair*.nii*` e, se `require_seg=True`, também uma máscara
    `*seg*.nii*`). Exigir `seg` exclui automaticamente pastas de VALIDAÇÃO (que
    têm FLAIR/T1/T1ce/T2 mas não trazem ground-truth) quando o dataset baixado
    inclui treino+validação lado a lado.
    """
    base = os.path.abspath(base)
    for root, dirs, _ in os.walk(base):
        depth = root[len(base):].count(os.sep)
        if depth >= max_depth:
            dirs[:] = []   # poda a busca (evita descer nas pastas de caso em si)
            continue
        dirs.sort()
        for d in dirs:
            case_dir = os.path.join(root, d)
            has_flair = bool(glob.glob(os.path.join(case_dir, "*_flair*.nii*")))
            has_seg = bool(glob.glob(os.path.join(case_dir, "*[Ss]eg*.nii*")))
            if has_flair and (has_seg or not require_seg):
                return root
    raise FileNotFoundError(
        f"Nenhuma pasta de caso BraTS completa (flair{'+seg' if require_seg else ''}) "
        f"encontrada sob {base}. Confira se o download/unzip terminou.")


def split_cases(root: str, val_frac: float = 0.2, seed: int = 42,
                grade_lookup: Optional[Dict[str, int]] = None):
    """Split treino/val a nível de PACIENTE. Se `grade_lookup` for dado,
    estratifica por grau (mantém a proporção LGG/HGG nos dois lados)."""
    import random
    cases = discover_cases(root)
    rng = random.Random(seed)
    if grade_lookup:
        by_grade: Dict[int, List[str]] = {}
        for c in cases:
            by_grade.setdefault(int(grade_lookup.get(c, 0)), []).append(c)
        train, val = [], []
        for g, group in by_grade.items():
            rng.shuffle(group)
            n_val = max(1, int(len(group) * val_frac))
            val += group[:n_val]; train += group[n_val:]
        rng.shuffle(train); rng.shuffle(val)
        return train, val
    rng.shuffle(cases)
    n_val = max(1, int(len(cases) * val_frac))
    return cases[n_val:], cases[:n_val]


def _make_grade_sampler(grade_labels: List[int]):
    """WeightedRandomSampler que equilibra as classes de grau por época."""
    from torch.utils.data import WeightedRandomSampler
    counts = np.bincount(np.asarray(grade_labels), minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0
    w = (1.0 / counts)[np.asarray(grade_labels)]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                 num_samples=len(grade_labels), replacement=True)


def grade_class_weights(labels: List[int], n_grades: int = 2) -> "torch.Tensor":
    """Pesos inverso-frequência (estilo sklearn 'balanced') para a CE de
    graduação, compensando o desbalanço LGG/HGG do BraTS (tipicamente ~75-80%
    HGG). Normalizados para média ~1, mantendo a escala da loss estável.

        w_c = N / (n_grades * count_c)

    Prefira isto a `balance_grade` (resampling) quando há poucos casos da classe
    minoritária: reponderar a loss é mais estável que reamostrar (não revê o
    mesmo punhado de LGG dezenas de vezes por época). Não use os dois ao mesmo
    tempo — seria correção dupla."""
    arr = np.asarray(labels)
    counts = np.bincount(arr, minlength=n_grades).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_grades * counts)
    return torch.tensor(w, dtype=torch.float32)


def _patient_grades(ds) -> Dict[str, int]:
    """Mapa case_id -> grau para um dataset (nível de PACIENTE). Usa `entries`
    (NpySliceDataset) quando disponível; senão cai para pares case_id/grade do
    índice on-the-fly. Retorna {} se o dataset não expõe essa informação."""
    entries = getattr(ds, "entries", None)
    if entries is not None:                       # NpySliceDataset
        return {e["case_id"]: int(e["grade"]) for e in entries}
    grade_map = getattr(ds, "_grade", None)
    if isinstance(grade_map, dict):               # BraTSSliceDataset
        return {c: int(g) for c, g in grade_map.items()}
    return {}


def grade_split_report(train_ds, val_ds, n_grades: int = 2, verbose: bool = True) -> Dict:
    """Sanity-check PERMANENTE do split: distribuição de grau em treino e validação,
    a nível de PACIENTE e de FATIA. Detecta split não-estratificado (ex.: 100% de
    uma classe num dos lados). Retorna um dict com as contagens/frações e, com
    `verbose`, imprime uma tabela no formato dos demais prints do notebook."""
    names = GRADE_NAMES

    def _counts(labels: List[int]) -> np.ndarray:
        return np.bincount(np.asarray(labels, dtype=int), minlength=n_grades)

    tr_slice = _counts(train_ds.grade_labels())
    va_slice = _counts(val_ds.grade_labels())
    tr_pat = _counts(list(_patient_grades(train_ds).values()))
    va_pat = _counts(list(_patient_grades(val_ds).values()))
    has_patient = int((tr_pat + va_pat).sum()) > 0     # datasets on-the-fly/npz têm; sintético não
    total_pat = tr_pat + va_pat

    def _frac(c: np.ndarray) -> List[float]:
        s = int(c.sum())
        return [float(x) / s if s else float("nan") for x in c]

    report = {
        "patient": {"train": tr_pat.tolist(), "val": va_pat.tolist()},
        "slice": {"train": tr_slice.tolist(), "val": va_slice.tolist()},
        "val_frac_by_grade": _frac(va_pat if has_patient else va_slice),
        "dataset_frac_by_grade": _frac(total_pat if has_patient else tr_slice + va_slice),
    }
    if verbose:
        print("Verificação do split (estratificação por grau):")
        if has_patient:
            print(f"{'':10s} {'PACIENTES':>22s}   {'FATIAS':>22s}")
            print(f"{'grau':10s} {'treino':>7s} {'val':>7s} {'%val':>6s}   "
                  f"{'treino':>7s} {'val':>7s} {'%val':>6s}")
            for g in range(n_grades):
                pv, sv = tr_pat[g] + va_pat[g], tr_slice[g] + va_slice[g]
                p_pct = 100 * va_pat[g] / pv if pv else float("nan")
                s_pct = 100 * va_slice[g] / sv if sv else float("nan")
                print(f"{names.get(g, g):10s} {tr_pat[g]:7d} {va_pat[g]:7d} {p_pct:5.1f}%   "
                      f"{tr_slice[g]:7d} {va_slice[g]:7d} {s_pct:5.1f}%")
            lgg_all = _frac(total_pat)[0] if n_grades >= 2 else float("nan")
            print(f"→ LGG = {100*lgg_all:.1f}% do dataset; ambos os graus devem aparecer "
                  f"em treino E validação (senão o split não está estratificado).")
            miss = min(va_pat.tolist()) == 0 or min(tr_pat.tolist()) == 0
        else:  # sem info a nível de paciente (ex.: SyntheticBraTS) -> só fatias
            print(f"{'grau':10s} {'treino':>7s} {'val':>7s} {'%val':>6s}   (nível de fatia)")
            for g in range(n_grades):
                sv = tr_slice[g] + va_slice[g]
                s_pct = 100 * va_slice[g] / sv if sv else float("nan")
                print(f"{names.get(g, g):10s} {tr_slice[g]:7d} {va_slice[g]:7d} {s_pct:5.1f}%")
            miss = min(va_slice.tolist()) == 0 or min(tr_slice.tolist()) == 0
        if miss:
            print("  ⚠ um dos lados ficou SEM uma das classes — split NÃO estratificado.")
    return report


def make_loaders(root: str, batch_size: int = 8, img_size: int = 256,
                 slices_per_case: int = 32, num_workers: int = 2,
                 grade_lookup: Optional[Dict[str, int]] = None,
                 augment: bool = True, balance_grade: bool = False):
    """Loaders on-the-fly com split estratificado + augmentation no treino."""
    train_ids, val_ids = split_cases(root, grade_lookup=grade_lookup)
    common = dict(root=root, img_size=img_size, slices_per_case=slices_per_case,
                  grade_lookup=grade_lookup)
    aug = SegAugmentation() if augment else None
    train_ds = BraTSSliceDataset(case_ids=train_ids, transform=aug, **common)
    val_ds = BraTSSliceDataset(case_ids=val_ids, **common)
    return _wrap_loaders(train_ds, val_ds, batch_size, num_workers, balance_grade)


def make_loaders_precomputed(shard_dir: str, batch_size: int = 8, num_workers: int = 2,
                             val_frac: float = 0.2, seed: int = 42,
                             augment: bool = True, balance_grade: bool = False):
    """Loaders a partir de shards `.npz` (rápido). Split por paciente estratificado
    pelo grau lido do index.json."""
    with open(os.path.join(shard_dir, "index.json")) as f:
        meta = json.load(f)
    grade_lookup = {e["case_id"]: e["grade"] for e in meta["index"]}
    cases = sorted(grade_lookup)
    import random
    rng = random.Random(seed)
    by_grade: Dict[int, List[str]] = {}
    for c in cases:
        by_grade.setdefault(grade_lookup[c], []).append(c)
    train_ids, val_ids = [], []
    for group in by_grade.values():
        rng.shuffle(group)
        n_val = max(1, int(len(group) * val_frac))
        val_ids += group[:n_val]; train_ids += group[n_val:]
    aug = SegAugmentation() if augment else None
    train_ds = NpySliceDataset(shard_dir, transform=aug, case_ids=train_ids)
    val_ds = NpySliceDataset(shard_dir, case_ids=val_ids)
    return _wrap_loaders(train_ds, val_ds, batch_size, num_workers, balance_grade)


def _wrap_loaders(train_ds, val_ds, batch_size, num_workers, balance_grade):
    pin = torch.cuda.is_available()
    if balance_grade:
        sampler = _make_grade_sampler(train_ds.grade_labels())
        train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=pin, drop_last=True)
    else:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin)
    return train_dl, val_dl


# ---------------------------------------------------------------------------
# Ingestão de fontes externas
# ---------------------------------------------------------------------------
_TCIA_ALIASES = {
    "flair": ["flair", "FLAIR"],
    "t1ce":  ["t1Gd", "t1ce", "T1Gd", "T1CE", "t1_gd"],
    "t1":    ["t1", "T1"],
    "t2":    ["t2", "T2"],
    "seg":   ["GlistrBoost", "GlistrBoost_ManuallyCorrected", "seg", "SegBinRHO", "segmentation"],
}


def normalize_tcia_case(case_dir: str) -> None:
    """Cria symlinks *_flair.nii.gz / *_t1ce.nii.gz / ... apontando para os
    arquivos reais, quaisquer que sejam seus nomes originais (coleções TCIA)."""
    cid = os.path.basename(os.path.normpath(case_dir))
    existing = glob.glob(os.path.join(case_dir, "*.nii*"))
    for canon, aliases in _TCIA_ALIASES.items():
        dst = os.path.join(case_dir, f"{cid}_{canon}.nii.gz")
        if os.path.lexists(dst):
            continue
        hit = None
        for alias in aliases:
            matches = [f for f in existing
                       if alias.lower() in os.path.basename(f).lower()]
            if matches:
                hit = matches[0]
                break
        if hit is None:
            raise FileNotFoundError(
                f"{cid}: nao encontrei arquivo para modalidade '{canon}' "
                f"(aliases tentados: {aliases}) em {case_dir}")
        os.symlink(os.path.abspath(hit), dst)


def build_grade_dataset_from_folders(
    dir_grade_pairs: List[Tuple[str, int]],
    out_root: str,
    normalize_tcia: bool = False,
) -> Dict[str, int]:
    """Une N pastas (cada uma = um grau) em out_root/<case_id>/... via symlink
    e devolve o grade_lookup pronto para make_loaders(..., grade_lookup=...).
    Funciona com TCIA (normalize_tcia=True) ou mirrors Kaggle BraTS2018 HGG/LGG
    (nomenclatura canônica -> normalize_tcia=False)."""
    os.makedirs(out_root, exist_ok=True)
    grade_lookup: Dict[str, int] = {}
    for src_dir, grade in dir_grade_pairs:
        case_dirs = [d for d in sorted(glob.glob(os.path.join(src_dir, "*")))
                     if os.path.isdir(d)]
        for d in case_dirs:
            cid = os.path.basename(os.path.normpath(d))
            dst = os.path.join(out_root, cid)
            if not os.path.lexists(dst):
                os.symlink(os.path.abspath(d), dst)
            if normalize_tcia:
                normalize_tcia_case(dst)
            grade_lookup[cid] = grade
    return grade_lookup


def build_tcia_grade_dataset(lgg_dir: str, gbm_dir: str, out_root: str) -> Dict[str, int]:
    """Wrapper de compatibilidade p/ TCIA."""
    return build_grade_dataset_from_folders(
        [(lgg_dir, 0), (gbm_dir, 1)], out_root, normalize_tcia=True)


def build_grade_lookup_from_csv(
    csv_path: str,
    root: Optional[str] = None,
    id_col: str = "BraTS_2020_subject_ID",
    grade_col: str = "Grade",
    grade_map: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """Lê o grau de um CSV (ex.: `name_mapping.csv` do BraTS2020) -> grade_lookup.

    BraTS2020: cada linha traz `Grade` (HGG/LGG) e `BraTS_2020_subject_ID`
    (= nome da pasta do caso). Ex.:
        grade_lookup = build_grade_lookup_from_csv(
            '/content/.../MICCAI_BraTS2020_TrainingData/name_mapping.csv',
            root='/content/.../MICCAI_BraTS2020_TrainingData')
    Passa direto para make_loaders(..., grade_lookup=grade_lookup).
    """
    grade_map = grade_map or {"HGG": 1, "LGG": 0, "GBM": 1, "gbm": 1, "lgg": 0}
    lookup: Dict[str, int] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or id_col not in reader.fieldnames:
            raise KeyError(f"coluna '{id_col}' ausente em {csv_path}. "
                           f"Colunas: {reader.fieldnames}")
        for row in reader:
            cid = (row.get(id_col) or "").strip()
            graw = (row.get(grade_col) or "").strip()
            if not cid or graw in ("", "NA", "nan"):
                continue
            if graw in grade_map:
                lookup[cid] = grade_map[graw]
    if root is not None:  # mantém só casos realmente presentes em disco
        present = set(discover_cases(root))
        lookup = {c: g for c, g in lookup.items() if c in present}
    if not lookup:
        raise RuntimeError(f"Nenhum grau resolvido de {csv_path} (confira id_col/grade_col).")
    return lookup


# ---------------------------------------------------------------------------
# Dataset sintético (valida o pipeline sem baixar nada)
# ---------------------------------------------------------------------------
class SyntheticBraTS(Dataset):
    """Dataset sintético para validar o pipeline no Colab. Gera anéis aninhados
    (edema -> core -> ET) como 'tumor'. ~40% dos casos são LGG-like (sem ET),
    dividindo o rótulo de grau em 2 classes p/ a curva ROC/AUC funcionar."""

    def __init__(self, n: int = 64, img_size: int = 256, n_seg_classes: int = 4,
                 transform: Optional[Callable] = None):
        self.n, self.s, self.k = n, img_size, n_seg_classes
        self.transform = transform

    def __len__(self):
        return self.n

    def _draw(self, i):
        """Reproduz a sequência de RNG de um caso (mesma ordem do __getitem__)."""
        g = torch.Generator().manual_seed(i)
        img = torch.randn(3, self.s, self.s, generator=g) * 0.4
        margin = max(int(0.25 * self.s), 8)              # margem proporcional ao tamanho
        cx, cy = torch.randint(margin, self.s - margin, (2,), generator=g).tolist()
        is_lgg = (torch.rand(1, generator=g).item() < 0.4)
        return img, cx, cy, is_lgg

    def __getitem__(self, i):
        img, cx, cy, is_lgg = self._draw(i)
        seg = torch.zeros(self.s, self.s, dtype=torch.long)
        yy, xx = torch.meshgrid(torch.arange(self.s), torch.arange(self.s), indexing="ij")
        base = self.s / 256.0
        rings = [(int(45 * base), 2), (int(28 * base), 1)] if is_lgg \
            else [(int(45 * base), 2), (int(28 * base), 1), (int(14 * base), 3)]
        for r, cls in rings:
            m = ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
            seg[m] = cls
            img[:, m] += cls * 0.7
        grade = 1 if (seg == 3).sum() > 0 else 0
        item = {"image": img, "seg_mask": seg, "grade": torch.tensor(grade),
                "case_id": f"synt{i}", "slice": 0}
        if self.transform is not None:
            item["image"], item["seg_mask"] = self.transform(item["image"], item["seg_mask"])
        return item

    def grade_labels(self) -> List[int]:
        return [0 if self._draw(i)[3] else 1 for i in range(self.n)]


if __name__ == "__main__":
    ds = SyntheticBraTS(n=4)
    b = ds[0]
    print("image   :", tuple(b["image"].shape))
    print("seg_mask:", tuple(b["seg_mask"].shape), "classes:", b["seg_mask"].unique().tolist())
    print("grade   :", b["grade"].item())
    aug = SegAugmentation(seed=0)
    ia, sa = aug(b["image"], b["seg_mask"])
    print("aug ok  :", tuple(ia.shape), tuple(sa.shape))
