"""
train.py
========
Loop de treino/validação multi-tarefa (segmentação + graduação).

Recursos:
    - AMP (mixed precision) p/ caber na T4
    - warmup linear + cosine annealing
    - gradient clipping e acumulação de gradiente (batch efetivo maior na T4)
    - early stopping por paciência (melhor dice_mean de validação)
    - AUC de graduação por época (acumulada no loader de validação)
    - checkpoint completo (best.pt) + checkpoint LoRA leve (lora.pt, poucos MB)

Uso no Colab:
    from src.train import Trainer, TrainConfig
    from src.dataset import make_loaders_precomputed
    tr, va = make_loaders_precomputed(shard_dir, batch_size=8, balance_grade=True)
    hist = Trainer(cfg).fit(tr, va)

Linha de comando:
    python -m src.train --synthetic --epochs 5 --backbone stub
"""

from __future__ import annotations
import os, json, time, math, argparse
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

import torch
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    def GradScaler(enabled=True): return _GradScaler("cuda", enabled=enabled)
    def autocast(enabled=True):   return _autocast("cuda", enabled=enabled)
except Exception:                                    # PyTorch < 2.3
    from torch.cuda.amp import autocast, GradScaler

from .models import build_model
from .losses import MultiTaskLoss
from .metrics import seg_scores, grade_scores, grade_roc_auc, MetricTracker


@dataclass
class TrainConfig:
    # dados
    root: str = ""
    img_size: int = 256
    batch_size: int = 8
    slices_per_case: int = 32
    # modelo
    backbone: str = "stub"                 # "sam2" | "stub"
    sam2_cfg: Optional[str] = None
    sam2_ckpt: Optional[str] = None
    freeze_encoder: bool = True
    p_drop: float = 0.1
    # --- LoRA ---
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    n_seg_classes: int = 4
    n_grades: int = 2
    feat_ch: int = 256
    # perdas
    w_seg: float = 1.0
    w_grade: float = 0.5
    use_focal: bool = False
    region: str = "dice"                   # "dice" | "tversky"
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    tversky_gamma: float = 1.0
    grade_label_smoothing: float = 0.05
    seg_ce_weight: Optional[List[float]] = None
    # otimização
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    min_lr_ratio: float = 0.02
    grad_clip: float = 1.0
    accum_steps: int = 1                   # acumulação de gradiente (batch efetivo)
    amp: bool = True
    early_stop_patience: int = 0           # 0 = desativado
    # infra
    out_dir: str = "runs/exp1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    def __init__(self, cfg: TrainConfig, model: Optional[torch.nn.Module] = None):
        self.cfg = cfg
        os.makedirs(os.path.join(cfg.out_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(cfg.out_dir, "logs"), exist_ok=True)

        self.device = torch.device(cfg.device)
        # `model` permite injetar variantes de ablação (ex.: U-Net) sem passar por
        # build_model; se None, constrói o modelo principal a partir da config.
        self.model = (model if model is not None else build_model(cfg)).to(self.device)

        w = torch.tensor(cfg.seg_ce_weight, device=self.device) if cfg.seg_ce_weight else None
        self.criterion = MultiTaskLoss(
            cfg.n_seg_classes, seg_ce_weight=w, w_seg=cfg.w_seg, w_grade=cfg.w_grade,
            use_focal=cfg.use_focal, region=cfg.region, tversky_alpha=cfg.tversky_alpha,
            tversky_beta=cfg.tversky_beta, tversky_gamma=cfg.tversky_gamma,
            grade_label_smoothing=cfg.grade_label_smoothing)

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, self._lr_lambda)
        self.scaler = GradScaler(enabled=cfg.amp and self.device.type == "cuda")
        self.history: Dict[str, List] = {"train": [], "val": []}
        self.best = -1.0
        self.best_epoch = 0

    def _lr_lambda(self, epoch: int) -> float:
        """Warmup linear -> cosine annealing até min_lr_ratio."""
        wu = max(self.cfg.warmup_epochs, 0)
        if epoch < wu:
            return (epoch + 1) / max(wu, 1)
        prog = (epoch - wu) / max(self.cfg.epochs - wu, 1)
        cos = 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
        return self.cfg.min_lr_ratio + (1 - self.cfg.min_lr_ratio) * cos

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        tracker = MetricTracker()
        grade_logits_all, grade_tgt_all = [], []
        self.opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            img = batch["image"].to(self.device, non_blocking=True)
            seg = batch["seg_mask"].to(self.device, non_blocking=True)
            grade = batch["grade"].to(self.device, non_blocking=True)

            with torch.set_grad_enabled(train), autocast(enabled=self.scaler.is_enabled()):
                out = self.model(img, return_aux=False)
                loss, parts = self.criterion(out, seg, grade)
                loss_scaled = loss / self.cfg.accum_steps

            if train:
                self.scaler.scale(loss_scaled).backward()
                if (step + 1) % self.cfg.accum_steps == 0:
                    if self.cfg.grad_clip > 0:
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(
                            (p for p in self.model.parameters() if p.requires_grad),
                            self.cfg.grad_clip)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                pred = out["seg_logits"].argmax(1)
                m = {**parts, **seg_scores(pred, seg),
                     **grade_scores(out["grade_logits"], grade)}
                if not train:
                    grade_logits_all.append(out["grade_logits"].float().cpu())
                    grade_tgt_all.append(grade.cpu())
            tracker.update(m, batch=img.size(0))

        avg = tracker.average()
        if not train and grade_logits_all:
            roc = grade_roc_auc(torch.cat(grade_logits_all), torch.cat(grade_tgt_all))
            avg["grade_auc"] = roc.get("auc", float("nan"))
        return avg

    def fit(self, train_dl, val_dl):
        for ep in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            tr = self._run_epoch(train_dl, train=True)
            va = self._run_epoch(val_dl, train=False)
            self.sched.step()
            self.history["train"].append(tr)
            self.history["val"].append(va)

            auc = va.get("grade_auc", float("nan"))
            print(f"[{ep:03d}/{self.cfg.epochs}] "
                  f"loss {tr['loss']:.3f}/{va['loss']:.3f} | "
                  f"Dice(WT/TC/ET) {va['dice_WT']:.3f}/{va['dice_TC']:.3f}/{va['dice_ET']:.3f} "
                  f"| grade_acc {va['grade_acc']:.3f} AUC {auc:.3f} | "
                  f"lr {self.opt.param_groups[0]['lr']:.2e} | {time.time()-t0:.0f}s")

            self._save_history()
            if va["dice_mean"] > self.best:
                self.best, self.best_epoch = va["dice_mean"], ep
                self._save_ckpt("best.pt", ep, va)
                if self.cfg.use_lora:
                    self._save_lora("lora.pt", ep, va)
            elif self.cfg.early_stop_patience and \
                    (ep - self.best_epoch) >= self.cfg.early_stop_patience:
                print(f"early stopping: sem melhora ha {self.cfg.early_stop_patience} epocas "
                      f"(melhor dice_mean={self.best:.3f} @ ep {self.best_epoch})")
                break
        return self.history

    def _save_ckpt(self, name, ep, metrics):
        path = os.path.join(self.cfg.out_dir, "checkpoints", name)
        torch.save({"model": self.model.state_dict(), "epoch": ep,
                    "metrics": metrics, "cfg": asdict(self.cfg)}, path)

    def _save_lora(self, name, ep, metrics):
        from .lora import lora_state_dict
        path = os.path.join(self.cfg.out_dir, "checkpoints", name)
        torch.save({"lora": lora_state_dict(self.model), "epoch": ep,
                    "metrics": metrics, "cfg": asdict(self.cfg)}, path)

    def _save_history(self):
        with open(os.path.join(self.cfg.out_dir, "logs", "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--backbone", type=str, default="stub")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--region", type=str, default="dice")
    p.add_argument("--synthetic", action="store_true", help="usa SyntheticBraTS")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    cfg = TrainConfig(root=args.root, epochs=args.epochs, backbone=args.backbone,
                      batch_size=args.batch_size, region=args.region)
    from torch.utils.data import DataLoader
    if args.synthetic or not args.root:
        from .dataset import SyntheticBraTS, SegAugmentation
        tr = DataLoader(SyntheticBraTS(64, transform=SegAugmentation()),
                        batch_size=cfg.batch_size, shuffle=True)
        va = DataLoader(SyntheticBraTS(16), batch_size=cfg.batch_size)
    else:
        from .dataset import make_loaders
        tr, va = make_loaders(cfg.root, batch_size=cfg.batch_size,
                              img_size=cfg.img_size, slices_per_case=cfg.slices_per_case)
    Trainer(cfg).fit(tr, va)
