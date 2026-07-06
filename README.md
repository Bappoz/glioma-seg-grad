# Segmentação e Graduação de Glioblastoma via SAM 2 / MedSAM + Redes de Atenção

Projeto final de Visão Computacional Avançada — FCTE/UnB.
Pipeline modular em PyTorch que **segmenta** as sub-regiões do tumor (necrose,
tumor ativo, edema) e **gradua** a severidade a partir da máscara segmentada.
A graduação depende explicitamente da segmentação — o projeto **não é** um
classificador de imagens isolado.

## Arquitetura (dois estágios acoplados)

```
        MRI (3 modalidades: FLAIR, T1ce, T2)  ->  pseudo-RGB [3,H,W]
                              │
                    ┌─────────▼──────────┐
                    │  Encoder SAM2/MedSAM│  (Hiera, congelado)
                    │  priors de fundação │
                    └─────────┬──────────┘
                       features densas [C,h,w]
                    ┌─────────┼─────────────────────────┐
        Estágio A   │                                   │  Estágio B
     (SEGMENTAÇÃO)  ▼                                   ▼ (GRADUAÇÃO)
        AttnDecoder + Attention Gates        MaskGuidedAttnClassifier
        -> seg_logits [4,H,W]  ───(máscara)──►  pooling de atenção
           {fundo,NCR,ED,ET}                     SÓ na região do tumor
                                                 -> grade_logits [K]
```

**Por que SAM2/MedSAM + atenção supera a U-Net pura?**
- O encoder Hiera do SAM2 chega **pré-treinado em >1 bilhão de máscaras**;
  traz priors de bordas/objetidade que uma U-Net treinada do zero em poucas
  centenas de volumes BraTS não aprende.
- **Attention Gates** (Oktay et al., 2018) suprimem crânio e edema difuso,
  afinando as bordas entre necrose / tumor ativo / edema.
- O **acoplamento seg→graduação** (masked attention pooling) força a decisão de
  grau a se basear na região segmentada — interpretável e alinhado à clínica.
- MedSAM2 acrescenta um bloco de memória-atenção que dá contexto entre fatias
  vizinhas do volume MRI, algo que o SAM2 puro (frame a frame) não faz.

**Fine-tuning eficiente com LoRA** (Hu et al., 2021)
- O encoder de fundação fica **congelado**; injetamos adaptadores de baixo posto
  (LoRA) nas projeções de atenção. Só os adaptadores treinam (~1–5% dos pesos).
- Vantagem dupla: cabe na memória do Colab (T4) **e** preserva os priors da
  fundação, evitando o *catastrophic forgetting* de um fine-tuning completo em
  poucos volumes BraTS.
- `lora_B` inicia em zero (Δ=0), então o modelo parte exatamente do encoder
  pré-treinado. `merge_all_lora()` funde os adaptadores para inferência sem custo.

## Estrutura do repositório

```
glioma_seg_grade/
├── src/
│   ├── models.py     # SAM2Encoder(+LoRA), AttnDecoder, MaskGuidedAttnClassifier, SAM2SegGradeNet
│   ├── dataset.py    # BraTSSliceDataset, pré-proc MRI, SyntheticBraTS (teste)
│   ├── losses.py     # DiceLoss, Focal, SegLoss (Dice+CE), MultiTaskLoss
│   ├── metrics.py    # Dice/IoU/Sens/Espec por sub-região + Hausdorff95 + ROC/AUC
│   ├── lora.py       # LoRALinear/Conv2d, inject_lora, merge_all_lora (PEFT)
│   ├── viz.py        # overlays qualitativos, curvas de treino, ROC, barras HD95
│   └── train.py      # Trainer, TrainConfig, loop AMP + cosine LR
├── configs/default.yaml
├── notebooks/colab_glioma.ipynb
├── requirements.txt
└── README.md
```

## Uso rápido (Google Colab)

```bash
!git clone <seu-repo> && cd glioma_seg_grade
!pip install -r requirements.txt
```

```python
# 1) valida o pipeline SEM baixar dados (dataset sintético)
from src.train import Trainer, TrainConfig
from src.dataset import SyntheticBraTS
from torch.utils.data import DataLoader

cfg = TrainConfig(backbone="stub", epochs=3, batch_size=8)
tr = DataLoader(SyntheticBraTS(64), batch_size=8, shuffle=True)
va = DataLoader(SyntheticBraTS(16), batch_size=8)
Trainer(cfg).fit(tr, va)
```

```python
# 2) BraTS real (após montar o Drive / baixar do TCIA/Synapse)
from src.dataset import make_loaders
cfg = TrainConfig(root="/content/BraTS2021", backbone="stub", epochs=30)
tr, va = make_loaders(cfg.root, batch_size=cfg.batch_size)
Trainer(cfg).fit(tr, va)
```

## Ligando o backbone SAM2/MedSAM real

```bash
pip install "git+https://github.com/facebookresearch/segment-anything-2.git"
# baixe um checkpoint Hiera (ex.: sam2_hiera_small.pt)
```
```python
cfg = TrainConfig(backbone="sam2",
                  sam2_cfg="sam2_hiera_s.yaml",
                  sam2_ckpt="checkpoints/sam2_hiera_small.pt",
                  freeze_encoder=True)
```

## Dados: BraTS

- BraTS 2021/2023 via **Synapse** (registro) ou **TCIA**.
- 4 modalidades co-registradas (T1, T1ce, T2, FLAIR), 240×240×155.
- Rótulos: `1`=NCR/NET, `2`=ED, `4`=ET → remapeados para `{0,1,2,3}` em `dataset.py`.
- Sub-regiões avaliadas: **WT** (1∪2∪4), **TC** (1∪4), **ET** (4).

## Métricas relatadas ao professor

| Tarefa | Métricas |
|--------|----------|
| Segmentação | **Dice** (WT/TC/ET), IoU, Sensibilidade, Especificidade, **Hausdorff95** |
| Graduação | Accuracy, Sensibilidade, Especificidade, **ROC / AUC** |

O `train.py` grava `runs/exp1/logs/history.json` — use o notebook para plotar as
curvas de loss e Dice por época.

### Módulo de visualização (`src/viz.py`)

Para a seção **Resultados & Conclusão** exigida pelo professor:

```python
from src import viz
# 1) painel qualitativo: MRI | Ground Truth | Predição | Mapa de erro (FP/FN)
viz.qualitative_panel(img, gt_mask, pred_mask, grade_true=1, grade_pred=1)
# 2) curvas de treino (loss, Dice por sub-região, métricas de graduação)
viz.plot_training_curves(history)          # history = runs/.../history.json
# 3) curva ROC da graduação  +  barras de Hausdorff95 por sub-região
viz.plot_roc(roc_dict); viz.plot_hd95_bars(hd95_dict)
```

O **mapa de erro** destaca falsos-positivos (magenta) e falsos-negativos (ciano)
sobre a MRI — evidência qualitativa direta da qualidade da segmentação.

> **Nota HD95:** retorna `nan` quando a máscara predita ou a de referência está
> vazia para aquela sub-região (ex.: ET ausente em caso LGG). Reporte a média
> apenas sobre casos válidos.

## Notas de rigor científico

- **Split por paciente** (não por fatia) — evita vazamento de dados.
- **z-score por cérebro** — a intensidade MRI não é calibrada entre scanners.
- Se o seu subset não trouxer o grau OMS, o rótulo de grau vira **proxy fraco**
  (presença de ET). Deixe isso explícito no relatório.
