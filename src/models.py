"""
models.py
=========
Arquitetura para "Segmentação + Graduação de Glioblastoma".

Design em DOIS ESTÁGIOS, com a SEGMENTAÇÃO no centro (requisito do projeto):

    Estágio A  -  SEGMENTAÇÃO
        Um encoder de fundação (SAM 2 / MedSAM2, backbone Hiera) — congelado —
        extrai um mapa de features denso. Um decoder leve com *Attention Gates*
        reconstrói a máscara multi-classe do tumor (ET / TC / WT).

    Estágio B  -  GRADUAÇÃO (o que satisfaz a regra "não pode ser só classificação")
        As features do encoder são agregadas com um *masked attention pooling*
        GUIADO pela máscara predita no Estágio A. Ou seja: o classificador só
        "olha" para os pixels que a segmentação disse serem tumor. A saída é o
        grau/severidade do tecido (ex.: LGG vs HGG, ou grau OMS).

Por que SAM2/MedSAM + atenção > U-Net pura?
  - O encoder Hiera do SAM2 vem pré-treinado em >1 bilhão de máscaras, trazendo
    priors de "objetidade" e bordas que uma U-Net treinada do zero em ~poucas
    centenas de volumes BraTS não aprende.
  - Attention Gates suprimem ativações irrelevantes (crânio, edema difuso) e
    focam a decodificação nas sub-regiões do tumor -> bordas mais nítidas entre
    necrose / tumor ativo / edema.
  - O acoplamento seg->classificação obriga o modelo a graduar A PARTIR da
    região segmentada, tornando a decisão interpretável e alinhada à clínica.

O código roda mesmo SEM os pesos do SAM2 instalados: há um encoder CNN de
fallback (`TinyHieraStub`) para você validar o pipeline no Colab antes de plugar
os checkpoints reais. Troque `backbone="sam2"` quando tiver o repositório
`segment-anything-2` + checkpoint.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. ENCODER
# ---------------------------------------------------------------------------
class SAM2Encoder(nn.Module):
    """
    Wrapper sobre o image-encoder do SAM 2 / MedSAM2.

    Estratégia de fine-tuning: CONGELAR o encoder (default). Isso preserva os
    priors de fundação e cabe na memória do Colab (T4/L4). Para adaptação mais
    forte, use LoRA/adapters em vez de descongelar tudo.

    Retorna um tensor de features denso [B, C, h, w] (backbone stride ~16).
    """

    def __init__(
        self,
        backbone: str = "stub",              # "sam2" | "stub"
        sam2_cfg: Optional[str] = None,       # ex.: "sam2_hiera_s.yaml"
        sam2_ckpt: Optional[str] = None,      # ex.: "checkpoints/sam2_hiera_small.pt"
        freeze: bool = True,
        out_channels: int = 256,
        use_lora: bool = False,              # injeta adaptadores LoRA no encoder
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_targets: tuple = ("qkv", "proj", "q_proj", "v_proj", "k_proj"),
    ):
        super().__init__()
        self.backbone = backbone
        self.out_channels = out_channels
        self.use_lora = use_lora
        self.n_lora = 0

        if backbone == "sam2":
            # Import tardio: só exige a lib se você realmente usar o SAM2.
            from sam2.build_sam import build_sam2                       # type: ignore
            model = build_sam2(sam2_cfg, sam2_ckpt, device="cpu")
            self.image_encoder = model.image_encoder
            self._native_dim = self._infer_native_dim()
        elif backbone == "stub":
            # Encoder CNN mínimo só para o pipeline "andar" sem os pesos SAM2.
            self.image_encoder = TinyHieraStub(out_dim=out_channels)
            self._native_dim = out_channels
        else:
            raise ValueError(f"backbone desconhecido: {backbone}")

        # Projeta a dimensão nativa do backbone para `out_channels` (canal comum
        # usado pelo decoder e pela cabeça de classificação).
        self.neck = nn.Conv2d(self._native_dim, out_channels, kernel_size=1)

        # ---- Estratégia de adaptação do encoder ----
        # 1) LoRA: congela os pesos base e injeta adaptadores de baixo rank.
        #    Só ~<1% dos parâmetros treinam; preserva os priors de fundação.
        # 2) freeze puro: congela tudo (encoder vira extrator fixo de features).
        # 3) nenhum: fine-tuning completo (caro; use só com muitos dados/GPU).
        if use_lora:
            from .lora import inject_lora, mark_only_lora_as_trainable
            self.n_lora = inject_lora(
                self.image_encoder, r=lora_r, alpha=lora_alpha,
                dropout=lora_dropout, target_names=lora_targets,
                include_conv=(backbone == "stub"),   # stub usa conv 'proj' de teste
            )
            mark_only_lora_as_trainable(self.image_encoder)
            if self.n_lora == 0:
                import warnings
                warnings.warn(
                    "LoRA: nenhuma camada casou com lora_targets. Imprima os "
                    "nomes com [n for n,_ in encoder.image_encoder.named_modules()] "
                    "e ajuste `lora_targets` ao seu backbone."
                )
        elif freeze and backbone == "sam2":
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)

    def _infer_native_dim(self) -> int:
        # Heurística: a maioria dos Hiera-S/B expõe 256 no FPN de saída.
        return 256

    def _forward_encoder(self, x):
        # Com LoRA, o gradiente PRECISA fluir pelo encoder (os adaptadores
        # treinam). Sem LoRA e congelado, não há grad a propagar de qualquer
        # forma, então um único caminho serve aos dois casos.
        return self.image_encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] normalizado. -> feats [B, out_channels, h, w]."""
        if self.backbone == "sam2":
            feats = self._forward_encoder(x)
            # SAM2 retorna dict/lista dependendo da versão; pegue o mapa denso.
            if isinstance(feats, dict):
                feats = feats.get("vision_features", list(feats.values())[0])
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]
        else:
            feats = self.image_encoder(x)
        return self.neck(feats)


class TinyHieraStub(nn.Module):
    """Encoder CNN de fallback (~stride 16). NÃO é o SAM2, apenas um substituto
    para desenvolvimento/CI. Facilita testar shapes e loop de treino."""

    def __init__(self, in_ch: int = 3, out_dim: int = 256):
        super().__init__()

        def blk(i, o, s):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, stride=s, padding=1),
                nn.GroupNorm(8, o), nn.GELU(),
                nn.Conv2d(o, o, 3, padding=1),
                nn.GroupNorm(8, o), nn.GELU(),
            )

        self.stem = blk(in_ch, 32, 2)     # /2
        self.s1 = blk(32, 64, 2)          # /4
        self.s2 = blk(64, 128, 2)         # /8
        self.s3 = blk(128, out_dim, 2)    # /16
        # projeção nomeada 'proj' -> serve de alvo LoRA quando testando no stub
        self.proj = nn.Conv2d(out_dim, out_dim, 1)

    def forward(self, x):
        x = self.stem(x); x = self.s1(x); x = self.s2(x)
        return self.proj(self.s3(x))


# ---------------------------------------------------------------------------
# 2. ATTENTION GATE  (Oktay et al., 2018 - "Attention U-Net")
# ---------------------------------------------------------------------------
class AttentionGate(nn.Module):
    """Filtra a skip-connection `x` usando o sinal de gating `g` (mais profundo).
    Produz coeficientes de atenção alpha in [0,1] que realçam regiões do tumor."""

    def __init__(self, f_x: int, f_g: int, f_int: int):
        super().__init__()
        self.theta_x = nn.Conv2d(f_x, f_int, 1)
        self.phi_g = nn.Conv2d(f_g, f_int, 1)
        self.psi = nn.Conv2d(f_int, 1, 1)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        q = F.relu(self.theta_x(x) + self.phi_g(g), inplace=True)
        alpha = torch.sigmoid(self.psi(q))          # [B,1,h,w]
        return x * alpha, alpha


# ---------------------------------------------------------------------------
# 3. DECODER DE SEGMENTAÇÃO  (Estágio A)
# ---------------------------------------------------------------------------
class AttnDecoder(nn.Module):
    """Decoder progressivo com Attention Gates. Entra a feature densa do encoder,
    sai o logit de segmentação multi-classe na resolução da imagem."""

    def __init__(self, in_ch: int = 256, n_classes: int = 4, widths=(256, 128, 64, 32)):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.gates = nn.ModuleList()
        c = in_ch
        for w in widths:
            self.gates.append(AttentionGate(f_x=c, f_g=c, f_int=max(w // 2, 16)))
            self.blocks.append(nn.Sequential(
                nn.Conv2d(c, w, 3, padding=1), nn.GroupNorm(8, w), nn.GELU(),
                nn.Conv2d(w, w, 3, padding=1), nn.GroupNorm(8, w), nn.GELU(),
            ))
            c = w
        self.head = nn.Conv2d(c, n_classes, 1)

    def forward(self, feat: torch.Tensor, out_size: Tuple[int, int]):
        x = feat
        for gate, block in zip(self.gates, self.blocks):
            x, _ = gate(x, x)                       # self-gating no nível
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(x)
        logits = self.head(x)
        logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits                               # [B, n_classes, H, W]


# ---------------------------------------------------------------------------
# 4. CABEÇA DE GRADUAÇÃO  (Estágio B) - guiada pela máscara
# ---------------------------------------------------------------------------
class MaskGuidedAttnClassifier(nn.Module):
    """
    Agrega as features do encoder SÓ nas regiões que a segmentação apontou como
    tumor (masked attention pooling) e classifica o grau/severidade.

    Isto é o elo seg->classificação: a graduação depende explicitamente da
    máscara predita, não da imagem inteira.
    """

    def __init__(self, feat_ch: int = 256, n_seg_classes: int = 4,
                 n_grades: int = 2, embed: int = 128):
        super().__init__()
        # pesos de atenção espacial condicionados às features do tumor
        self.attn = nn.Sequential(
            nn.Conv2d(feat_ch, embed, 1), nn.GELU(),
            nn.Conv2d(embed, 1, 1),
        )
        self.proj = nn.Sequential(nn.Linear(feat_ch, embed), nn.GELU(), nn.Dropout(0.2))
        self.classifier = nn.Linear(embed, n_grades)
        self.n_seg_classes = n_seg_classes

    def forward(self, feat: torch.Tensor, seg_logits: torch.Tensor) -> torch.Tensor:
        # máscara de tumor (qualquer classe != fundo), reduzida à grade das feats
        seg_prob = torch.softmax(seg_logits, dim=1)
        tumor_prob = 1.0 - seg_prob[:, 0:1]                        # [B,1,H,W]
        tumor_prob = F.interpolate(tumor_prob, size=feat.shape[-2:], mode="bilinear",
                                   align_corners=False)

        attn = self.attn(feat)                                    # [B,1,h,w]
        attn = attn.masked_fill(tumor_prob < 1e-4, float("-inf")) # zera fora do tumor
        B, C, h, w = feat.shape
        a = torch.softmax(attn.view(B, 1, -1), dim=-1).view(B, 1, h, w)
        a = torch.nan_to_num(a)                                   # imagens sem tumor
        pooled = (feat * a).sum(dim=(2, 3))                       # [B, C]
        emb = self.proj(pooled)
        return self.classifier(emb)                               # [B, n_grades]


# ---------------------------------------------------------------------------
# 5. MODELO COMPLETO
# ---------------------------------------------------------------------------
class SAM2SegGradeNet(nn.Module):
    """Une encoder de fundação + decoder de segmentação + graduação por atenção.

    forward -> dict {'seg_logits': [B,Kseg,H,W], 'grade_logits': [B,Kgrade]}"""

    def __init__(
        self,
        backbone: str = "stub",
        sam2_cfg: Optional[str] = None,
        sam2_ckpt: Optional[str] = None,
        freeze_encoder: bool = True,
        n_seg_classes: int = 4,    # fundo, ET, TC-core, WT-edema (ver dataset.py)
        n_grades: int = 2,         # LGG vs HGG (ajuste p/ graus OMS se tiver rótulo)
        feat_ch: int = 256,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = SAM2Encoder(backbone, sam2_cfg, sam2_ckpt,
                                   freeze=freeze_encoder, out_channels=feat_ch,
                                   use_lora=use_lora, lora_r=lora_r,
                                   lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.decoder = AttnDecoder(in_ch=feat_ch, n_classes=n_seg_classes)
        self.grader = MaskGuidedAttnClassifier(feat_ch, n_seg_classes, n_grades)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encoder(x)                        # [B,C,h,w]
        seg_logits = self.decoder(feat, out_size=x.shape[-2:])
        grade_logits = self.grader(feat, seg_logits)
        return {"seg_logits": seg_logits, "grade_logits": grade_logits}


def build_model(cfg) -> SAM2SegGradeNet:
    """Fábrica a partir de um objeto de config (ver configs/default.yaml)."""
    return SAM2SegGradeNet(
        backbone=getattr(cfg, "backbone", "stub"),
        sam2_cfg=getattr(cfg, "sam2_cfg", None),
        sam2_ckpt=getattr(cfg, "sam2_ckpt", None),
        freeze_encoder=getattr(cfg, "freeze_encoder", True),
        n_seg_classes=getattr(cfg, "n_seg_classes", 4),
        n_grades=getattr(cfg, "n_grades", 2),
        feat_ch=getattr(cfg, "feat_ch", 256),
        use_lora=getattr(cfg, "use_lora", False),
        lora_r=getattr(cfg, "lora_r", 8),
        lora_alpha=getattr(cfg, "lora_alpha", 16),
        lora_dropout=getattr(cfg, "lora_dropout", 0.0),
    )


if __name__ == "__main__":
    # Smoke test: valida shapes sem precisar dos pesos SAM2.
    net = SAM2SegGradeNet(backbone="stub", n_seg_classes=4, n_grades=2)
    x = torch.randn(2, 3, 256, 256)
    out = net(x)
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("seg_logits :", tuple(out["seg_logits"].shape))
    print("grade_logits:", tuple(out["grade_logits"].shape))
    print("parametros treinaveis:", f"{n_train/1e6:.2f}M")
