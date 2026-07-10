"""
Testes das perdas: Dice/Tversky recompensam a predição correta, beta>alpha do
Tversky pesa mais os falsos-negativos, e a perda multi-tarefa é finita nos dois
modos de região. CPU, rápido.
"""
import torch

from src.losses import DiceLoss, TverskyLoss, MultiTaskLoss


def _binary_logits(mask: torch.Tensor, hi: float = 10.0):
    """Logits [1,2,H,W] fortemente one-hot p/ uma máscara binária [H,W] (0/1)."""
    H, W = mask.shape
    lg = torch.full((1, 2, H, W), -hi)
    lg[0, 0][mask == 0] = hi
    lg[0, 1][mask == 1] = hi
    return lg


def test_dice_and_tversky_reward_correct_prediction():
    target = torch.zeros(8, 8, dtype=torch.long); target[2:5, 2:5] = 1
    good = _binary_logits(target)                          # prediz a lesão certa
    bad = _binary_logits(torch.zeros_like(target))         # prevê tudo fundo (perde a lesão)
    for loss_fn in (DiceLoss(2), TverskyLoss(2, alpha=0.3, beta=0.7)):
        l_good = loss_fn(good, target[None]).item()
        l_bad = loss_fn(bad, target[None]).item()
        assert l_good < 0.1                                # acerto -> loss baixa
        assert l_bad > 0.9                                 # perder a lesão -> loss alta


def test_tversky_beta_penalizes_false_negatives():
    # overlap parcial (metade da lesão): TP=FN. beta maior pune mais os FN.
    target = torch.zeros(8, 8, dtype=torch.long); target[2:6, 2:6] = 1
    pred = torch.zeros(8, 8, dtype=torch.long); pred[2:4, 2:6] = 1   # cobre metade
    logits = _binary_logits(pred)
    lo = TverskyLoss(2, alpha=0.7, beta=0.3)(logits, target[None]).item()
    hi = TverskyLoss(2, alpha=0.3, beta=0.7)(logits, target[None]).item()
    assert hi > lo                                         # beta>alpha -> mais peso no recall


def test_multitask_loss_dict():
    net_out = {"seg_logits": torch.randn(2, 4, 8, 8),
               "grade_logits": torch.randn(2, 2)}
    seg = torch.randint(0, 4, (2, 8, 8)); grade = torch.randint(0, 2, (2,))
    for region in ("dice", "tversky"):
        loss, parts = MultiTaskLoss(4, region=region)(net_out, seg, grade)
        assert torch.isfinite(loss) and set(parts) == {"loss", "loss_seg", "loss_grade"}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("OK:", name)
