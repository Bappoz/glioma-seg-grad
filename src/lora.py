"""
lora.py
=======
Low-Rank Adaptation (LoRA, Hu et al. 2021) para o encoder de fundação SAM2/MedSAM2.

Motivação
---------
O encoder Hiera do SAM2 tem dezenas de milhões de parâmetros. Fazer fine-tuning
completo em poucas centenas de volumes BraTS (a) estoura a memória do Colab e
(b) destrói os priors de fundação (catastrophic forgetting). LoRA resolve isso:

    W_efetivo = W_congelado + (alpha / r) * (B @ A)

onde A e B são matrizes de baixa dimensão (rank r << dim). Só A e B treinam;
W permanece congelado. Tipicamente <1% dos parâmetros ficam treináveis, mas a
adaptação ao domínio médico é suficiente.

Como usar
---------
    from src.lora import inject_lora, mark_only_lora_as_trainable, lora_parameters

    n = inject_lora(encoder, r=8, alpha=16, target_names=("qkv", "proj"))
    mark_only_lora_as_trainable(encoder)          # congela o resto
    opt = torch.optim.AdamW(lora_parameters(encoder), lr=3e-4)

O `inject_lora` percorre o módulo e substitui `nn.Linear` (e, opcionalmente,
`nn.Conv2d` 1x1) cujo NOME contenha um dos `target_names`. Isso é robusto a
diferenças de versão do SAM2, pois não depende do caminho exato das camadas.
"""

from __future__ import annotations
from typing import Iterable, Sequence, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Camadas LoRA (wrappers que embrulham a camada congelada original)
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Envolve um `nn.Linear` congelado e soma um caminho de baixo rank.

    A camada base fica intacta (pesos originais preservados); só `lora_A` e
    `lora_B` treinam. `merge()` funde os pesos para inferência sem overhead.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_f, out_f = base.in_features, base.out_features
        # A: [r, in] inicializada com Kaiming; B: [out, r] começa em zero
        # -> no início o caminho LoRA é nulo, preservando o modelo de fundação.
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r > 0 and not self.merged:
            delta = self.drop(x) @ self.lora_A.t() @ self.lora_B.t()
            out = out + self.scaling * delta
        return out

    @torch.no_grad()
    def merge(self):
        """Funde B@A nos pesos base (para exportar/inferir sem o ramo extra)."""
        if self.r > 0 and not self.merged:
            self.base.weight.data += self.scaling * (self.lora_B @ self.lora_A)
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.r > 0 and self.merged:
            self.base.weight.data -= self.scaling * (self.lora_B @ self.lora_A)
            self.merged = False


class LoRAConv2d(nn.Module):
    """LoRA para `nn.Conv2d` (útil se o backbone usa projeções convolucionais).
    Suporta qualquer kernel; o ramo de baixo rank usa duas convs em sequência."""

    def __init__(self, base: nn.Conv2d, r: int = 8, alpha: int = 16):
        super().__init__()
        assert isinstance(base, nn.Conv2d)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0

        self.lora_A = nn.Conv2d(base.in_channels, r, base.kernel_size,
                                stride=base.stride, padding=base.padding, bias=False)
        self.lora_B = nn.Conv2d(r, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.merged = False

    def forward(self, x):
        out = self.base(x)
        if self.r > 0 and not self.merged:
            out = out + self.scaling * self.lora_B(self.lora_A(x))
        return out


# ---------------------------------------------------------------------------
# Injeção: percorre o modelo e troca camadas-alvo por wrappers LoRA
# ---------------------------------------------------------------------------
def _get_parent(root: nn.Module, dotted: str):
    """Retorna (modulo_pai, nome_do_filho) para um caminho 'a.b.c'."""
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_names: Sequence[str] = ("qkv", "proj", "q_proj", "v_proj", "k_proj"),
    include_conv: bool = False,
) -> int:
    """Substitui in-place `nn.Linear`/`nn.Conv2d` cujo nome contenha um alvo.

    Retorna o número de camadas adaptadas. Se 0, revise `target_names` para o
    seu backbone (imprima `[n for n,_ in model.named_modules()]`).
    """
    to_patch: List[str] = []
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if not any(t in name for t in target_names):
            continue
        if isinstance(module, nn.Linear):
            to_patch.append((name, "linear"))
        elif include_conv and isinstance(module, nn.Conv2d):
            to_patch.append((name, "conv"))

    count = 0
    for name, kind in to_patch:
        parent, child = _get_parent(model, name)
        base = getattr(parent, child)
        if kind == "linear":
            setattr(parent, child, LoRALinear(base, r=r, alpha=alpha, dropout=dropout))
        else:
            setattr(parent, child, LoRAConv2d(base, r=r, alpha=alpha))
        count += 1
    return count


def mark_only_lora_as_trainable(model: nn.Module, train_bias: bool = False) -> None:
    """Congela tudo, exceto os parâmetros LoRA (e opcionalmente os bias)."""
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
        elif train_bias and n.endswith(".bias"):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)


def lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    """Iterador só com os parâmetros treináveis (para o otimizador)."""
    return (p for n, p in model.named_parameters() if p.requires_grad)


def lora_state_dict(model: nn.Module) -> dict:
    """Extrai SÓ os pesos LoRA — checkpoint leve (poucos MB) para versionar."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def count_trainable(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": train,
            "pct": 100.0 * train / max(total, 1)}


def merge_all_lora(model: nn.Module) -> None:
    """Funde todos os ramos LoRA para inferência (chame antes de exportar)."""
    for m in model.modules():
        if isinstance(m, (LoRALinear, LoRAConv2d)) and hasattr(m, "merge"):
            m.merge()
