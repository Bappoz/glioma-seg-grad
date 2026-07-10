"""Config compartilhada dos testes: CPU determinística e leve."""
import os
import torch

torch.manual_seed(0)
torch.set_num_threads(min(4, os.cpu_count() or 1))
os.environ.setdefault("MPLBACKEND", "Agg")   # matplotlib headless nos testes
