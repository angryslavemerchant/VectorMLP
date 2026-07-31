"""End-to-end small CNN on CIFAR-10 pixels: the co-adapting host test.

The last live branch of the drop-in claim. Unlike the frozen-DINO test, the
backbone here trains JOINTLY with the head, so it can learn to feed channel
structure into the vector head's reshape interface (128 x 16 from the conv
feature map). If co-adaptation is real, the vector head should not be
wiring-starved the way it was on frozen features.

Arms (identical backbone, ~same total params):
    cnn-vec   SmallCNN -> reshape 2048 = 128 neurons x 16 ch -> vector head
              (pure rank-4 mixing)
    cnn-mlp   SmallCNN -> param-matched plain MLP head

Sizes x 5 seeds, vmap-stacked. Heavier than the head-only grids (conv
activations for 15 stacked models) — meant for a box, not the laptop.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from torchvision.datasets import CIFAR10

from vector_mlp import VectorMLP, PlainMLP, count_params, matched_mlp_width
from experiments.mnist_grid import balanced_subset, train_stack, eval_stack
from experiments.cifar_features import rotated  # also patches the HF mirror

OUT_JSON = ROOT / 'results' / 'cifar_e2e_results.json'
DIM = 16
HEAD_HIDDEN = [64, 64]
SIZES = [2000, 10000, 50000]
SEEDS = 5
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def make_models(factory):
    """One model per (size, seed) cell for THIS grid's sizes (mnist_grid's
    make_models is bound to its own 4-size grid)."""
    models = []
    for si in range(len(SIZES)):
        for seed in range(SEEDS):
            torch.manual_seed(1000 + 97 * si + seed)
            models.append(factory())
    return models


class SmallCNN(nn.Module):
    """3 conv blocks: [B, 3072] flat pixels -> [B, 2048] features."""

    def __init__(self):
        super().__init__()
        chans = [3, 32, 64, 128]
        self.blocks = nn.Sequential(*[
            m for a, b in zip(chans[:-1], chans[1:])
            for m in (nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))])

    def forward(self, x):
        x = x.reshape(-1, 3, 32, 32)
        return self.blocks(x).flatten(1)          # 128 * 4 * 4 = 2048


class E2E(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.backbone = SmallCNN()
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


def main():
    tr = CIFAR10(ROOT / 'data', train=True, download=True)
    te = CIFAR10(ROOT / 'data', train=False, download=True)
    to_t = lambda d: torch.from_numpy(d).permute(0, 3, 1, 2).float() / 255.0
    tr_x, te_x = to_t(tr.data), to_t(te.data)
    te_rot = rotated(te_x)
    norm = lambda x: ((x - MEAN) / STD).flatten(1)
    tx, ex, ex_rot = norm(tr_x), norm(te_x), norm(te_rot)
    ty, ey = torch.tensor(tr.targets), torch.tensor(te.targets)

    gen = torch.Generator().manual_seed(42)
    subsets = [[balanced_subset(ty, n, gen) for _ in range(SEEDS)]
               for n in SIZES]
    flat_subsets = [s for group in subsets for s in group]

    def vec_head():
        return VectorMLP(2048 // DIM, HEAD_HIDDEN, 10, DIM,
                         channel_mix='lowrank_pure', rank=4, vector_in=True)

    head_target = count_params(vec_head())
    mlp_w, mlp_par = matched_mlp_width(head_target, 2048, 10, len(HEAD_HIDDEN))
    print(f'vec head {head_target:,} | mlp head width {mlp_w}, {mlp_par:,} | '
          f'backbone {count_params(SmallCNN()):,}', flush=True)

    arms = {
        'cnn-vec': lambda: E2E(vec_head()),
        'cnn-mlp': lambda: E2E(PlainMLP(2048, [mlp_w] * len(HEAD_HIDDEN), 10)),
    }

    results = {}
    for name, factory in arms.items():
        models = make_models(factory)
        print(f'--- {name}: {count_params(models[0])} params x {len(models)} models',
              flush=True)
        params, buffers, base = train_stack(models, flat_subsets, tx, ty)
        acc = eval_stack(params, buffers, base, ex, ey,
                         chunk=500).reshape(len(SIZES), SEEDS)
        acc_r = eval_stack(params, buffers, base, ex_rot, ey,
                           chunk=500).reshape(len(SIZES), SEEDS)
        results[name] = {'params': count_params(models[0]),
                         'clean': acc.tolist(), 'rot45': acc_r.tolist()}
        for si, n in enumerate(SIZES):
            print(f'  n={n:<6} clean {acc[si].mean():.4f}+-{acc[si].std():.4f}'
                  f'   rot45 {acc_r[si].mean():.4f}+-{acc_r[si].std():.4f}',
                  flush=True)

    with open(OUT_JSON, 'w') as f:
        json.dump({'sizes': SIZES, 'seeds': SEEDS, 'results': results}, f,
                  indent=1)
    print(f'\nsaved -> {OUT_JSON}')


if __name__ == '__main__':
    main()
