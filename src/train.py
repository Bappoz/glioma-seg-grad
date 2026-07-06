"""
train.py
========
Loop de treino/validação multi-tarefa (segmentação + graduação).

Uso no Colab:
    from src.train import Trainer, TrainConfig
    from src.dataset import make_loaders, SyntheticBraTS
    ...
    trainer = Trainer(cfg)
    trainer.fit(train_dl, val_dl)

Ou linha de comando:
    python -m src.train --root /content/BraTS2021 --epochs 30 --backbone stub

Salva:
    - checkpoints/best.pt  (melhor dice_mean de validação)
    - logs/history.json    (curvas p/ plotar no notebook)
"""

from __future__ import annotations
import os, json, time, argparse
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List

import torch
# AMP: usa a API nova (torch.amp, PyTorch>=2.3) com fallback p/ a antiga.
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    def GradScaler(enabled=True): return _GradScaler("cuda", enabled=enabled)
    def autocast(enabled=True):   return _autocast("cuda", enabled=enabled)
except Exception:                                    # PyTorch < 2.3
    from torch.cuda.amp import autocast, GradScaler

from .models import build_model
from .losses import MultiTaskLoss
from .metrics import seg_scores, grade_scores, MetricTracker


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
    # --- LoRA (fine-tuning eficiente do encoder congelado) ---
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    n_seg_classes: int = 4
    n_grades: int = 2
    feat_ch: int = 256
    # otimização
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    w_seg: float = 1.0
    w_grade: float = 0.5
    use_focal: bool = False
    amp: bool = True
    # infra
    out_dir: str = "runs/exp1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seg_ce_weight: Optional[List[float]] = None   # ex.: [0.1,1,1,1] realça tumor


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        os.makedirs(os.path.join(cfg.out_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(cfg.out_dir, "logs"), exist_ok=True)

        self.device = torch.device(cfg.device)
        self.model = build_model(cfg).to(self.device)

        w = torch.tensor(cfg.seg_ce_weight, device=self.device) if cfg.seg_ce_weight else None
        self.criterion = MultiTaskLoss(cfg.n_seg_classes, seg_ce_weight=w,
                                       w_seg=cfg.w_seg, w_grade=cfg.w_grade,
                                       use_focal=cfg.use_focal)
        # só parâmetros treináveis (encoder congelado fica de fora)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=cfg.epochs)
        self.scaler = GradScaler(enabled=cfg.amp and self.device.type == "cuda")
        self.history: Dict[str, List] = {"train": [], "val": []}
        self.best = -1.0

    # ---------------- treino de 1 epoch ----------------
    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        tracker = MetricTracker()
        for batch in loader:
            img = batch["image"].to(self.device, non_blocking=True)
            seg = batch["seg_mask"].to(self.device, non_blocking=True)
            grade = batch["grade"].to(self.device, non_blocking=True)

            with torch.set_grad_enabled(train), autocast(enabled=self.scaler.is_enabled()):
                out = self.model(img)
                loss, parts = self.criterion(out, seg, grade)

            if train:
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()

            with torch.no_grad():
                pred = out["seg_logits"].argmax(1)
                m = {**parts, **seg_scores(pred, seg), **grade_scores(out["grade_logits"], grade)}
            tracker.update(m, batch=img.size(0))
        return tracker.average()

    # ---------------- fit ----------------
    def fit(self, train_dl, val_dl):
        for ep in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            tr = self._run_epoch(train_dl, train=True)
            va = self._run_epoch(val_dl, train=False)
            self.sched.step()
            self.history["train"].append(tr)
            self.history["val"].append(va)

            print(f"[{ep:03d}/{self.cfg.epochs}] "
                  f"loss {tr['loss']:.3f}/{va['loss']:.3f} | "
                  f"Dice(WT/TC/ET) {va['dice_WT']:.3f}/{va['dice_TC']:.3f}/{va['dice_ET']:.3f} "
                  f"| grade_acc {va['grade_acc']:.3f} | {time.time()-t0:.0f}s")

            self._save_history()
            if va["dice_mean"] > self.best:
                self.best = va["dice_mean"]
                self._save_ckpt("best.pt", ep, va)
        return self.history

    def _save_ckpt(self, name, ep, metrics):
        path = os.path.join(self.cfg.out_dir, "checkpoints", name)
        torch.save({"model": self.model.state_dict(), "epoch": ep,
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
    p.add_argument("--synthetic", action="store_true", help="usa SyntheticBraTS")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    cfg = TrainConfig(root=args.root, epochs=args.epochs,
                      backbone=args.backbone, batch_size=args.batch_size)
    from torch.utils.data import DataLoader
    if args.synthetic or not args.root:
        from .dataset import SyntheticBraTS
        tr = DataLoader(SyntheticBraTS(n=64), batch_size=cfg.batch_size, shuffle=True)
        va = DataLoader(SyntheticBraTS(n=16), batch_size=cfg.batch_size)
    else:
        from .dataset import make_loaders
        tr, va = make_loaders(cfg.root, batch_size=cfg.batch_size,
                              img_size=cfg.img_size, slices_per_case=cfg.slices_per_case)
    Trainer(cfg).fit(tr, va)
