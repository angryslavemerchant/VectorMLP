"""MNIST sample-efficiency grids, all rounds in one runner.

    python experiments/mnist_grid.py <round>     # 1, 2, 3, or 4

Every (size x seed) model of an arm trains as ONE stacked computation via
torch.func.vmap — 20 models in lockstep on the GPU. All rounds share the
same seeded data subsets, so results are paired per-seed across rounds.
Raw accuracies land in results/mnist_results<round>.json; analysis and
conclusions in results/experiment_results.md.

Rounds:
  1  channel-mix variants (matrix/ring/shared) vs param-matched plain MLPs
  2  ring-kernel ladder (K=3/16) + matrix D-sweep down (8, 4)
  3  D-sweep up (32, 64) + residual low-rank mixing (y = g + UVg)
  4  pure low-rank factorization (y = UVg)

Rounds 2-4 arms are width-matched to round 1's matrix arm (~105k params).
"""

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.func import stack_module_state, functional_call, vmap, grad
from torchvision.datasets import MNIST

from vector_mlp import (VectorMLP, PlainMLP, count_params, matched_width,
                        matched_mlp_width)

DIM = 16
HIDDEN = [64, 64]
SIZES = [500, 2000, 10000, 60000]
SEEDS = 5
STEPS, BATCH, LR = 4000, 256, 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
P = 784

# Arm spec: 'width' is an int (fixed) or 'matched' (widest uniform width
# whose params fit round 1's matrix arm). 'mlp_match' arms are plain MLPs
# width-matched to the named vector arm of the same round.
ROUNDS = {
    '1': {
        'matrix':     dict(dim=16, channel_mix='matrix', width=64),
        'ring':       dict(dim=16, channel_mix='ring', kernel_size=5, width=64),
        'shared':     dict(dim=16, channel_mix='shared', width=64),
        'mlp@matrix': dict(mlp_match='matrix'),
        'mlp@ring':   dict(mlp_match='ring'),
    },
    '2': {
        'ring3':     dict(dim=16, channel_mix='ring', kernel_size=3, width='matched'),
        'ring16':    dict(dim=16, channel_mix='ring', kernel_size=16, width='matched'),
        'matrix-d8': dict(dim=8, channel_mix='matrix', width='matched'),
        'matrix-d4': dict(dim=4, channel_mix='matrix', width='matched'),
    },
    '3': {
        'lowrank-d16-r4': dict(dim=16, channel_mix='lowrank', rank=4, width='matched'),
        'matrix-d32':     dict(dim=32, channel_mix='matrix', width='matched'),
        'lowrank-d32-r4': dict(dim=32, channel_mix='lowrank', rank=4, width='matched'),
        'lowrank-d64-r4': dict(dim=64, channel_mix='lowrank', rank=4, width='matched'),
    },
    '4': {
        'purelr-d16-r4': dict(dim=16, channel_mix='lowrank_pure', rank=4, width='matched'),
        'purelr-d16-r8': dict(dim=16, channel_mix='lowrank_pure', rank=8, width='matched'),
    },
}


def load_mnist():
    tr = MNIST(ROOT / 'data', train=True, download=True)
    te = MNIST(ROOT / 'data', train=False, download=True)
    return (tr.data.float().reshape(-1, P) / 255.0, tr.targets,
            te.data.float().reshape(-1, P) / 255.0, te.targets)


def rotated(images, max_deg=45.0, seed=123):
    """Rotate each 28x28 image by a random angle in [-max_deg, max_deg]."""
    g = torch.Generator().manual_seed(seed)
    n = images.shape[0]
    ang = torch.deg2rad((torch.rand(n, generator=g) * 2 - 1) * max_deg)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
    theta[:, 1, 0], theta[:, 1, 1] = sin, cos
    imgs = images.reshape(n, 1, 28, 28)
    grid = F.affine_grid(theta, imgs.shape, align_corners=False)
    return F.grid_sample(imgs, grid, align_corners=False).reshape(n, P)


def balanced_subset(ty, n, gen):
    """n class-balanced indices."""
    per = n // 10
    idx = []
    for c in range(10):
        pool = (ty == c).nonzero(as_tuple=True)[0]
        idx.append(pool[torch.randperm(len(pool), generator=gen)[:per]])
    return torch.cat(idx)


def make_models(factory):
    """One model per (size, seed) cell, deterministic init per cell."""
    models = []
    for si in range(len(SIZES)):
        for seed in range(SEEDS):
            torch.manual_seed(1000 + 97 * si + seed)
            models.append(factory())
    return models


def train_stack(models, subsets, tx, ty, compile=False):
    """Train len(models) models in lockstep; subsets[i] indexes tx for model i."""
    models = [m.to(DEVICE) for m in models]
    params, buffers = stack_module_state(models)
    base = copy.deepcopy(models[0]).to('meta')
    for p in params.values():
        p.requires_grad_(True)
    opt = torch.optim.Adam(params.values(), lr=LR)

    def loss_fn(p, b, x, y):
        return F.cross_entropy(functional_call(base, (p, b), (x,)), y)

    grad_fn = vmap(grad(loss_fn), in_dims=(0, 0, 0, 0))
    if compile:
        grad_fn = torch.compile(grad_fn)
    tx, ty = tx.to(DEVICE), ty.to(DEVICE)
    subsets = [s.to(DEVICE) for s in subsets]
    gen = torch.Generator(device='cpu').manual_seed(7)
    for _ in range(STEPS):
        rows = torch.stack([s[torch.randint(0, len(s), (BATCH,), generator=gen)]
                            for s in subsets])                     # [M, B]
        grads = grad_fn(params, buffers, tx[rows], ty[rows])
        opt.zero_grad()
        for k, p in params.items():
            p.grad = grads[k]
        opt.step()
    return params, buffers, base


@torch.no_grad()
def eval_stack(params, buffers, base, ex, ey, chunk=1000, compile=False):
    fwd = vmap(lambda p, b, x: functional_call(base, (p, b), (x,)),
               in_dims=(0, 0, None))
    if compile:
        fwd = torch.compile(fwd)
    ex, ey = ex.to(DEVICE), ey.to(DEVICE)
    correct = None
    for i in range(0, ex.shape[0], chunk):
        pred = fwd(params, buffers, ex[i:i + chunk]).argmax(-1)     # [M, c]
        hit = (pred == ey[i:i + chunk]).sum(-1)
        correct = hit if correct is None else correct + hit
    return (correct.float() / ex.shape[0]).cpu()                    # [M]


def arm_factories(round_specs):
    target = count_params(VectorMLP(P, HIDDEN, 10, DIM, channel_mix='matrix'))
    arms = {}
    for name, spec in round_specs.items():
        spec = dict(spec)
        if 'mlp_match' in spec:
            ref = count_params(arms[spec['mlp_match']]())
            w, got = matched_mlp_width(ref, P, 10, len(HIDDEN))
            arms[name] = (lambda w=w: PlainMLP(P, [w] * len(HIDDEN), 10))
        else:
            dim, width = spec.pop('dim'), spec.pop('width')
            if width == 'matched':
                width, got = matched_width(
                    target,
                    lambda w: VectorMLP(P, [w] * len(HIDDEN), 10, dim, **spec))
            arms[name] = (lambda w=width, dim=dim, kw=spec:
                          VectorMLP(P, [w] * len(HIDDEN), 10, dim, **kw))
        print(f'{name}: {count_params(arms[name]()):,} params', flush=True)
    return arms


def main(round_id):
    tx, ty, ex, ey = load_mnist()
    ex_rot = rotated(ex)
    gen = torch.Generator().manual_seed(42)   # identical subsets every round
    subsets = [[balanced_subset(ty, n, gen) for _ in range(SEEDS)]
               for n in SIZES]
    flat_subsets = [s for group in subsets for s in group]

    results = {}
    for name, factory in arm_factories(ROUNDS[round_id]).items():
        models = make_models(factory)
        print(f'--- {name}: {count_params(models[0])} params x {len(models)} models',
              flush=True)
        params, buffers, base = train_stack(models, flat_subsets, tx, ty)
        acc = eval_stack(params, buffers, base, ex, ey).reshape(len(SIZES), SEEDS)
        acc_r = eval_stack(params, buffers, base, ex_rot, ey).reshape(len(SIZES), SEEDS)
        results[name] = {'params': count_params(models[0]),
                         'clean': acc.tolist(), 'rot45': acc_r.tolist()}
        for si, n in enumerate(SIZES):
            print(f'  n={n:<6} clean {acc[si].mean():.4f}+-{acc[si].std():.4f}'
                  f'   rot45 {acc_r[si].mean():.4f}+-{acc_r[si].std():.4f}',
                  flush=True)

    suffix = '' if round_id == '1' else round_id
    out = ROOT / 'results' / f'mnist_results{suffix}.json'
    with open(out, 'w') as f:
        json.dump({'sizes': SIZES, 'seeds': SEEDS, 'steps': STEPS,
                   'results': results}, f, indent=1)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ROUNDS, \
        f'usage: python mnist_grid.py [{"|".join(ROUNDS)}]'
    main(sys.argv[1])
