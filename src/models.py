"""
models.py
=========
Arquitetura para "Segmentação + Graduação de Glioblastoma".

Design em DOIS ESTÁGIOS ACOPLADOS, com a SEGMENTAÇÃO no centro:

    Estágio A  -  SEGMENTAÇÃO (Attention U-Net sobre encoder de fundação)
        Um encoder de fundação (SAM 2 / MedSAM2, backbone Hiera) — congelado —
        extrai features multi-escala. Um decoder com *Attention Gates* (Oktay
        et al., 2018) funde essas escalas via skip-connections e reconstrói a
        máscara multi-classe do tumor (NCR/ED/ET).

    Estágio B  -  GRADUAÇÃO (satisfaz a regra "não pode ser só classificação")
        As features do encoder são agregadas por um *masked attention pooling*
        GUIADO pela máscara predita no Estágio A e concatenadas a descritores
        geométricos (frações de área por sub-região). A saída é o grau (LGG×HGG
        ou grau OMS). A decisão só "olha" para os pixels que a segmentação
        apontou como tumor -> interpretável e alinhado à clínica.

Por que SAM2/MedSAM + atenção > U-Net pura?
  - Encoder Hiera pré-treinado em >1 bilhão de máscaras -> priors de bordas e
    "objetidade" que uma U-Net do zero em poucas centenas de volumes não aprende.
  - Attention Gates suprimem crânio/edema difuso -> bordas mais nítidas.
  - O acoplamento seg->graduação torna a decisão de grau interpretável.

O código roda SEM os pesos do SAM2: há um encoder CNN de fallback multi-escala
(`TinyHieraStub`) para validar o pipeline. Troque `backbone="sam2"` quando tiver
o repositório `segment-anything-2` + checkpoint.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, List
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. ENCODER (retorna feature profunda + skips multi-escala)
# ---------------------------------------------------------------------------
class TinyHieraStub(nn.Module):
    """Encoder CNN de fallback (~stride 16) que EXPÕE features multi-escala para
    as skip-connections. NÃO é o SAM2 — apenas um substituto testável.

    forward -> (feat[/16, out_dim], skips=[s2/8, s1/4, stem/2])"""

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
        self.proj = nn.Conv2d(out_dim, out_dim, 1)   # alvo LoRA no modo stub
        self.skip_channels = [128, 64, 32]           # s2, s1, stem (deep->shallow)

    def forward(self, x):
        stem = self.stem(x)
        s1 = self.s1(stem)
        s2 = self.s2(s1)
        feat = self.proj(self.s3(s2))
        return feat, [s2, s1, stem]


class SAM2Encoder(nn.Module):
    """Wrapper sobre o image-encoder do SAM 2 / MedSAM2.

    Estratégia de fine-tuning: CONGELAR o encoder (default) ou adaptá-lo com LoRA.
    Retorna `(feat[B,out_ch,h,w], skips)`. Para o backbone `sam2`, tenta extrair a
    pirâmide FPN como skips; se a introspecção falhar, degrada para sem-skips
    (o decoder usa self-gating), preservando robustez entre versões da lib.
    """

    def __init__(
        self,
        backbone: str = "stub",
        sam2_cfg: Optional[str] = None,
        sam2_ckpt: Optional[str] = None,
        freeze: bool = True,
        out_channels: int = 256,
        use_lora: bool = False,
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
        self.skip_channels: List[int] = []

        if backbone == "sam2":
            from sam2.build_sam import build_sam2                       # type: ignore
            model = build_sam2(sam2_cfg, sam2_ckpt, device="cpu")
            self.image_encoder = model.image_encoder
            self._native_dim = 256
            self._skip_projs = nn.ModuleList()
            self._probe_sam2_skips(out_channels)
        elif backbone == "stub":
            self.image_encoder = TinyHieraStub(out_dim=out_channels)
            self._native_dim = out_channels
            self.skip_channels = list(self.image_encoder.skip_channels)
        else:
            raise ValueError(f"backbone desconhecido: {backbone}")

        self.neck = nn.Conv2d(self._native_dim, out_channels, kernel_size=1)

        if use_lora:
            from .lora import inject_lora, mark_only_lora_as_trainable
            self.n_lora = inject_lora(
                self.image_encoder, r=lora_r, alpha=lora_alpha,
                dropout=lora_dropout, target_names=lora_targets,
                include_conv=(backbone == "stub"),
            )
            mark_only_lora_as_trainable(self.image_encoder)
            if self.n_lora == 0:
                warnings.warn(
                    "LoRA: nenhuma camada casou com lora_targets. Imprima os nomes "
                    "com [n for n,_ in encoder.image_encoder.named_modules()] e "
                    "ajuste `lora_targets` ao seu backbone.")
        elif freeze and backbone == "sam2":
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)

    # ---- SAM2: descobre a pirâmide de skips com um forward de sondagem ----
    @torch.no_grad()
    def _probe_sam2_skips(self, out_channels: int) -> None:
        try:
            dummy = torch.zeros(1, 3, 256, 256)
            raw = self.image_encoder(dummy)
            pyr = self._extract_pyramid(raw)
            # pyr: lista da mais profunda p/ mais rasa (exclui o mapa mais fundo=feat)
            self.skip_channels = [p.shape[1] for p in pyr]
            self._skip_projs = nn.ModuleList(
                [nn.Conv2d(c, out_channels, 1) for c in self.skip_channels])
            self.skip_channels = [out_channels] * len(self.skip_channels)
        except Exception as e:  # degrada com elegância
            warnings.warn(f"SAM2: sem skips multi-escala ({e}); usando self-gating.")
            self.skip_channels = []
            self._skip_projs = nn.ModuleList()

    @staticmethod
    def _extract_pyramid(raw) -> List[torch.Tensor]:
        """Normaliza a saída do image_encoder do SAM2 numa lista de mapas
        (profundo->raso), retornando os skips (todos menos o mais profundo)."""
        maps: List[torch.Tensor] = []
        if isinstance(raw, dict):
            for key in ("backbone_fpn", "fpn", "features", "vision_features"):
                if key in raw and isinstance(raw[key], (list, tuple)):
                    maps = list(raw[key]); break
            if not maps:
                vals = [v for v in raw.values() if torch.is_tensor(v) and v.dim() == 4]
                maps = vals
        elif isinstance(raw, (list, tuple)):
            maps = [m for m in raw if torch.is_tensor(m) and m.dim() == 4]
        elif torch.is_tensor(raw):
            maps = [raw]
        # ordena por resolução crescente (profundo primeiro)
        maps = sorted(maps, key=lambda m: m.shape[-1])
        return maps[1:] if len(maps) > 1 else []

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.backbone == "sam2":
            raw = self.image_encoder(x)
            maps = self._extract_pyramid_full(raw)
            feat = self.neck(maps[0])
            skips = []
            for proj, m in zip(self._skip_projs, maps[1:]):
                skips.append(proj(m))
            return feat, skips
        feat, skips = self.image_encoder(x)
        return self.neck(feat), skips

    def _extract_pyramid_full(self, raw) -> List[torch.Tensor]:
        """Como _extract_pyramid, mas inclui o mapa mais profundo em maps[0]."""
        if isinstance(raw, dict):
            for key in ("backbone_fpn", "fpn", "features"):
                if key in raw and isinstance(raw[key], (list, tuple)):
                    maps = sorted(raw[key], key=lambda m: m.shape[-1]); return maps
            vf = raw.get("vision_features")
            if torch.is_tensor(vf):
                return [vf]
            vals = [v for v in raw.values() if torch.is_tensor(v) and v.dim() == 4]
            return sorted(vals, key=lambda m: m.shape[-1]) or [next(iter(raw.values()))]
        if isinstance(raw, (list, tuple)):
            return sorted([m for m in raw if torch.is_tensor(m)], key=lambda m: m.shape[-1])
        return [raw]


# ---------------------------------------------------------------------------
# 2. ATTENTION GATE  (Oktay et al., 2018 - "Attention U-Net")
# ---------------------------------------------------------------------------
class AttentionGate(nn.Module):
    """Filtra a skip-connection `x` usando o sinal de gating `g` (mais profundo).
    Produz coeficientes alpha in [0,1] que realçam regiões do tumor."""

    def __init__(self, f_x: int, f_g: int, f_int: int):
        super().__init__()
        self.theta_x = nn.Conv2d(f_x, f_int, 1)
        self.phi_g = nn.Conv2d(f_g, f_int, 1)
        self.psi = nn.Conv2d(f_int, 1, 1)

    def forward(self, x: torch.Tensor, g: torch.Tensor):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        q = F.relu(self.theta_x(x) + self.phi_g(g), inplace=True)
        alpha = torch.sigmoid(self.psi(q))          # [B,1,h,w]
        return x * alpha, alpha


# ---------------------------------------------------------------------------
# 3. DECODER DE SEGMENTAÇÃO  (Estágio A) — Attention U-Net com skips reais
# ---------------------------------------------------------------------------
class AttnDecoder(nn.Module):
    """Decoder progressivo com Attention Gates. Quando o encoder fornece skips
    multi-escala, cada estágio faz gate(skip, sinal_do_decoder) e concatena —
    Attention U-Net de fato. Sem skips, cai no self-gating (fallback robusto)."""

    def __init__(self, in_ch: int = 256, n_classes: int = 4,
                 widths=(256, 128, 64, 32), skip_channels: Optional[List[int]] = None,
                 p_drop: float = 0.0):
        super().__init__()
        skip_channels = list(skip_channels or [])
        self.blocks = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.use_skip: List[bool] = []
        c = in_ch
        for i, w in enumerate(widths):
            sk = skip_channels[i] if i < len(skip_channels) else None
            if sk is not None:
                self.gates.append(AttentionGate(f_x=sk, f_g=c, f_int=max(w // 2, 16)))
                in_block = c + sk
                self.use_skip.append(True)
            else:
                self.gates.append(AttentionGate(f_x=c, f_g=c, f_int=max(w // 2, 16)))
                in_block = c
                self.use_skip.append(False)
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_block, w, 3, padding=1), nn.GroupNorm(8, w), nn.GELU(),
                nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity(),
                nn.Conv2d(w, w, 3, padding=1), nn.GroupNorm(8, w), nn.GELU(),
            ))
            c = w
        self.head = nn.Conv2d(c, n_classes, 1)

    def forward(self, feat: torch.Tensor, skips: Optional[List[torch.Tensor]],
                out_size: Tuple[int, int]):
        skips = skips or []
        x = feat
        for i, (gate, block) in enumerate(zip(self.gates, self.blocks)):
            if self.use_skip[i] and i < len(skips):
                skip = skips[i]
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                gated, _ = gate(skip, x)               # gate a skip com o sinal do decoder
                x = torch.cat([x, gated], dim=1)
            else:
                x, _ = gate(x, x)                      # self-gating (fallback)
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(x)
        logits = self.head(x)
        logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits                                  # [B, n_classes, H, W]


# ---------------------------------------------------------------------------
# 4. CABEÇA DE GRADUAÇÃO  (Estágio B) — guiada pela máscara + geometria
# ---------------------------------------------------------------------------
class MaskGuidedAttnClassifier(nn.Module):
    """Agrega as features do encoder SÓ nas regiões que a segmentação apontou como
    tumor (masked attention pooling) e concatena descritores geométricos (frações
    de área por sub-região) antes de classificar o grau.

    Isto é o elo seg->graduação: a decisão depende explicitamente da máscara. O
    mapa de atenção espacial é exposto (`grade_attn`) para explicabilidade."""

    def __init__(self, feat_ch: int = 256, n_seg_classes: int = 4,
                 n_grades: int = 2, embed: int = 128, p_drop: float = 0.2):
        super().__init__()
        self.n_seg_classes = n_seg_classes
        self.attn = nn.Sequential(
            nn.Conv2d(feat_ch, embed, 1), nn.GELU(),
            nn.Conv2d(embed, 1, 1),
        )
        self.proj = nn.Sequential(nn.Linear(feat_ch, embed), nn.GELU(), nn.Dropout(p_drop))
        n_geom = n_seg_classes                     # frações: tumor total + por classe (1..K-1)
        self.classifier = nn.Sequential(
            nn.Linear(embed + n_geom, embed), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(embed, n_grades),
        )

    def forward(self, feat: torch.Tensor, seg_logits: torch.Tensor,
                return_attn: bool = False):
        seg_prob = torch.softmax(seg_logits, dim=1)
        tumor_prob = 1.0 - seg_prob[:, 0:1]                        # [B,1,H,W]
        tumor_prob_ds = F.interpolate(tumor_prob, size=feat.shape[-2:],
                                      mode="bilinear", align_corners=False)

        attn = self.attn(feat)                                    # [B,1,h,w]
        attn = attn.masked_fill(tumor_prob_ds < 1e-4, float("-inf"))
        B, C, h, w = feat.shape
        a = torch.softmax(attn.view(B, 1, -1), dim=-1).view(B, 1, h, w)
        a = torch.nan_to_num(a)
        pooled = (feat * a).sum(dim=(2, 3))                       # [B, C]
        emb = self.proj(pooled)

        # descritores geométricos (interpretáveis, robustos): fração de área
        # do tumor total e de cada sub-região tumoral na fatia.
        frac_total = tumor_prob.mean(dim=(2, 3))                  # [B,1]
        frac_cls = seg_prob[:, 1:].mean(dim=(2, 3))               # [B,K-1]
        geom = torch.cat([frac_total, frac_cls], dim=1)          # [B,K]
        logits = self.classifier(torch.cat([emb, geom], dim=1))
        if return_attn:
            return logits, a, geom
        return logits


# ---------------------------------------------------------------------------
# 5. MODELO COMPLETO
# ---------------------------------------------------------------------------
class SAM2SegGradeNet(nn.Module):
    """Encoder de fundação + Attention U-Net (seg) + graduação guiada por máscara.

    forward -> dict {
        'seg_logits'  : [B,Kseg,H,W],
        'grade_logits': [B,Kgrade],
        'grade_attn'  : [B,1,h,w]   (atenção da graduação, p/ explicabilidade),
        'feat'        : [B,C,h,w]   (features do encoder; p/ uncertainty/explain)
    }"""

    def __init__(
        self,
        backbone: str = "stub",
        sam2_cfg: Optional[str] = None,
        sam2_ckpt: Optional[str] = None,
        freeze_encoder: bool = True,
        n_seg_classes: int = 4,
        n_grades: int = 2,
        feat_ch: int = 256,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        p_drop: float = 0.1,
    ):
        super().__init__()
        self.encoder = SAM2Encoder(backbone, sam2_cfg, sam2_ckpt,
                                   freeze=freeze_encoder, out_channels=feat_ch,
                                   use_lora=use_lora, lora_r=lora_r,
                                   lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.decoder = AttnDecoder(in_ch=feat_ch, n_classes=n_seg_classes,
                                   skip_channels=self.encoder.skip_channels, p_drop=p_drop)
        self.grader = MaskGuidedAttnClassifier(feat_ch, n_seg_classes, n_grades, p_drop=max(p_drop, 0.2))

    def forward(self, x: torch.Tensor, return_aux: bool = True) -> Dict[str, torch.Tensor]:
        feat, skips = self.encoder(x)
        seg_logits = self.decoder(feat, skips, out_size=x.shape[-2:])
        grade_logits, grade_attn, _ = self.grader(feat, seg_logits, return_attn=True)
        out = {"seg_logits": seg_logits, "grade_logits": grade_logits}
        if return_aux:
            out["grade_attn"] = grade_attn
            out["feat"] = feat
        return out


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
        p_drop=getattr(cfg, "p_drop", 0.1),
    )


if __name__ == "__main__":
    net = SAM2SegGradeNet(backbone="stub", n_seg_classes=4, n_grades=2)
    x = torch.randn(2, 3, 256, 256)
    out = net(x)
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("seg_logits  :", tuple(out["seg_logits"].shape))
    print("grade_logits:", tuple(out["grade_logits"].shape))
    print("grade_attn  :", tuple(out["grade_attn"].shape))
    print("skip_channels:", net.encoder.skip_channels)
    print("parametros treinaveis:", f"{n_train/1e6:.2f}M")
