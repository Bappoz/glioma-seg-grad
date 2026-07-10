# Glioma: Segmentação de Sub-regiões + Graduação (SAM 2 / MedSAM + Atenção + LoRA)

Projeto final de Visão Computacional Avançada — FCTE/UnB.
Pipeline modular em PyTorch de **duas etapas acopladas**: **(A) segmenta** as
sub-regiões do tumor (necrose/NCR, edema/ED, tumor ativo/ET) e **(B) gradua** a
lesão (LGG × HGG) **a partir da máscara segmentada**. A graduação depende
explicitamente da segmentação — o projeto **não é** um classificador de imagem isolado.

> **Comece por aqui:** abra `notebooks/colab_glioma.ipynb` no Colab (T4). Ele roda
> ponta-a-ponta em dados **sintéticos** (sem download) e, com uma flag, treina no
> **BraTS real**. Todas as figuras/tabelas do relatório saem de lá.

## Arquitetura (dois estágios acoplados)

```
MRI(FLAIR,T1ce,T2) → Encoder Hiera (SAM2/MedSAM, ❄ congelado + LoRA)
                         ├─► Attention U-Net (decoder + gates, skips multi-escala) ─► máscara NCR/ED/ET
                         └─► Masked-Attention Pooling + geometria ──(guiado pela máscara)──► grau LGG×HGG
```

- **Encoder de fundação** Hiera do SAM 2, pré-treinado em >1 bilhão de máscaras →
  priors de bordas/objetidade que uma U-Net do zero não aprende.
- **Attention U-Net** (Oktay et al., 2018): usa as features **multi-escala** do
  encoder como skip-connections gateadas → bordas nítidas entre NCR/ED/ET.
- **Acoplamento seg→grau**: *masked-attention pooling* guiado pela máscara +
  **descritores geométricos** (frações de área por sub-região) → decisão de grau
  interpretável e alinhada à clínica.
- **MedSAM2** acrescenta memória-atenção entre fatias vizinhas (contexto 3D).

### Fine-tuning eficiente com LoRA (Hu et al., 2021)
Encoder **congelado**; adaptadores de baixo posto (LoRA) nas projeções de atenção.
Treinam ~1–5% dos pesos → cabe na T4 **e** preserva os priors (sem *catastrophic
forgetting*). `lora_B=0` no início ⇒ parte exatamente do encoder pré-treinado;
`merge_all_lora()` funde para inferência sem custo. `lora.pt` versiona só os
adaptadores (poucos MB).

## Estrutura

| Arquivo | Papel |
|---------|-------|
| `src/dataset.py` | BraTS: z-score **por volume**, seleção de fatias, augmentation, split estratificado, **pré-computação em shards `.npz`**, ingestão TCIA/Kaggle/CSV, `SyntheticBraTS` |
| `src/models.py` | Encoder SAM2 (+skips multi-escala), Attention U-Net, graduação guiada por máscara + geometria |
| `src/lora.py` | Injeção/merge de LoRA, checkpoint leve |
| `src/losses.py` | Dice, **Focal-Tversky**, CE/Focal, multi-tarefa (+label smoothing) |
| `src/metrics.py` | Dice/IoU/Sens/Spec/**HD95**, ROC/AUC, **ECE/Brier/reliability**, F1/confusão |
| `src/train.py` | Trainer: AMP, warmup+cosine, grad clip, acumulação, early stopping, AUC/época |
| `src/biomarkers.py` | **Biomarcadores tumorais** (volumes, razões) → classificador interpretável seg→grau |
| `src/uncertainty.py` | **MC-Dropout** (incerteza epistêmica) + **TTA** |
| `src/explain.py` | **Explicabilidade** da graduação (atenção intrínseca + saliência por oclusão) |
| `src/baseline.py` | U-Net do zero + **harness de ablação** |
| `src/viz.py` | Figuras do relatório (curvas, ROC, HD95, incerteza, calibração, biomarcadores, ablação, qualitativo) |
| `tests/` | 25 testes em CPU com NIfTI mock (`pytest tests/`) |

## Uso rápido (Colab T4)

```python
# 1) valida o pipeline SEM baixar dados (sintético)
from src.train import Trainer, TrainConfig
from src.dataset import SyntheticBraTS, SegAugmentation
from torch.utils.data import DataLoader
cfg = TrainConfig(backbone="stub", epochs=5, batch_size=16, region="tversky")
tr = DataLoader(SyntheticBraTS(160, transform=SegAugmentation()), batch_size=16, shuffle=True)
va = DataLoader(SyntheticBraTS(48), batch_size=16)
Trainer(cfg).fit(tr, va)
```

## Dados: BraTS

- 4 modalidades co-registradas (T1, T1ce, T2, FLAIR), 240×240×155.
- Rótulos `1`=NCR/NET, `2`=ED, `4`=ET → remapeados para `{0,1,2,3}` em `dataset.py`.
- Sub-regiões avaliadas: **WT** (1∪2∪4), **TC** (1∪4), **ET** (4).

### Opção A — Kaggle · BraTS2020 (recomendada, grau real)

Mirror estável `awsaf49/brats20-dataset-training-validation` (369 casos), com a
estrutura oficial `BraTS20_Training_XXX/*_flair.nii …` **e** o grau real em
`name_mapping.csv` (coluna `Grade`). Gere `kaggle.json` em **Kaggle → Account →
Create New API Token**.

```bash
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!pip install kaggle
!kaggle datasets download -d awsaf49/brats20-dataset-training-validation -p /content --unzip
```
```python
from src.dataset import build_grade_lookup_from_csv, precompute_slices, make_loaders_precomputed
root = "/content/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
grade_lookup = build_grade_lookup_from_csv(f"{root}/name_mapping.csv", root=root)   # grau REAL
precompute_slices(root, "/content/shards", grade_lookup=grade_lookup)               # 1x -> shards rápidos
tr, va = make_loaders_precomputed("/content/shards", batch_size=4, balance_grade=True)
```

> **Procedência:** mirror de terceiros — declare no relatório.

### Opção B — Kaggle · BraTS2018 (pastas HGG/LGG)

```python
from src.dataset import build_grade_dataset_from_folders
grade_lookup = build_grade_dataset_from_folders(
    [("/content/MICCAI_BraTS_2018_Data_Training/HGG", 1),
     ("/content/MICCAI_BraTS_2018_Data_Training/LGG", 0)],
    out_root="/content/BraTS_merged")   # grau pela pasta
```

### Opção C — TCIA (BRATS-TCGA-LGG/GBM)

Nomenclatura diferente (`t1Gd`, `GlistrBoost`), download via **Aspera Connect**.
Use `build_tcia_grade_dataset(lgg_dir, gbm_dir, out_root)` (normaliza por symlink).

## Ligando o backbone SAM2/MedSAM real

```bash
pip install "git+https://github.com/facebookresearch/segment-anything-2.git"
# baixe um checkpoint Hiera (ex.: sam2_hiera_tiny.pt)
```
```python
cfg = TrainConfig(backbone="sam2", sam2_cfg="sam2_hiera_t.yaml",
                  sam2_ckpt="checkpoints/sam2_hiera_tiny.pt",
                  freeze_encoder=True, use_lora=True, lora_r=4)
```

## Métricas relatadas ao professor

| Tarefa | Métricas |
|--------|----------|
| Segmentação | **Dice** (WT/TC/ET), IoU, Sensibilidade, Especificidade, **Hausdorff95** |
| Graduação | Accuracy, Sens/Espec, **ROC/AUC**, F1, **ECE + Brier** (calibração) |
| Ablação | Dice/AUC × parâmetros treináveis (encoder de fundação × congelado × U-Net) |
| Interpretação | Biomarcadores (AUC univariada), incerteza epistêmica, mapas de atenção |

## Entregáveis "além do básico" (seções 8–12 do notebook)

- **Ablação científica** isolando o ganho do encoder de fundação + LoRA.
- **Biomarcadores** volumétricos (ET/TC ratio etc.) como ponte interpretável
  seg→grau, com classificador logístico transparente vs. cabeça neural.
- **Incerteza** (MC-Dropout + TTA): mapas de "onde o modelo hesita" e ganho de
  Dice com TTA.
- **Explicabilidade**: atenção intrínseca da graduação + saliência por oclusão.

## Notas de rigor científico

- **Split por paciente** estratificado por grau (evita vazamento e desbalanço).
- **z-score por cérebro** (volume inteiro) — intensidade MRI não calibrada entre scanners.
- **HD95** retorna `nan` quando a máscara (pred ou GT) é vazia p/ a sub-região
  (ex.: ET ausente em LGG) — reporte a média só sobre casos válidos.
- Sem o grau OMS no subset, o rótulo de grau vira **proxy fraco** (presença de ET
  a nível de paciente) — declare no relatório. Com BraTS2020 (Opção A) o grau é **real**.

## Testes

```bash
pytest tests/ -q       # 25 testes, CPU, ~4s (NIfTI mock + sintético)
```
