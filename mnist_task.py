"""MNIST sample-efficiency test (spec: channel_mixing_neuron_spec.md, exp 2-ish).

Fully-learned drop-in comparison: VectorMLP (matrix / ring / shared) vs
param-matched plain MLPs, across training-set sizes. Primary metric: clean
test accuracy vs train size. Secondary: accuracy on +-45deg rotated test set.

All (size x seed) models of one arm share parameter shapes, so each arm is
trained as ONE stacked computation via torch.func.vmap - 20 models advance in
lockstep on the GPU instead of 100 sequential runs.

Run inside Toastenv (CUDA torch + torchvision).
"""

import copy
import json
import math

import torch
import torch.nn.functional as F
from torch.func import stack_module_state, functional_call, vmap, grad
from torchvision.datasets import MNIST

from vector_mlp import VectorMLP, PlainMLP, count_params, matched_mlp_width

DIM = 16
HIDDEN = [64, 64]
SIZES = [500, 2000, 10000, 60000]
SEEDS = 5
STEPS, BATCH, LR = 4000, 256, 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = r'C:\Users\JmgLi\documents\Python Projects\VectorMLP\data'
OUT_JSON = r'C:\Users\JmgLi\documents\Python Projects\VectorMLP\mnist_results.json'


def load_mnist():
    tr = MNIST(DATA_DIR, train=True, download=True)
    te = MNIST(DATA_DIR, train=False, download=True)
    tx = tr.data.float().reshape(-1, 784) / 255.0
    ty = tr.targets
    ex = te.data.float().reshape(-1, 784) / 255.0
    ey = te.targets
    return tx, ty, ex, ey


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
    return F.grid_sample(imgs, grid, align_corners=False).reshape(n, 784)


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


def train_stack(models, subsets, tx, ty):
    """Train len(models) models in lockstep; subsets[i] indexes tx for model i."""
    models = [m.to(DEVICE) for m in models]
    params, buffers = stack_module_state(models)
    base = copy.deepcopy(models[0]).to('meta')
    M = len(models)
    for p in params.values():
        p.requires_grad_(True)
    opt = torch.optim.Adam(params.values(), lr=LR)

    def loss_fn(p, b, x, y):
        return F.cross_entropy(functional_call(base, (p, b), (x,)), y)

    grad_fn = vmap(grad(loss_fn), in_dims=(0, 0, 0, 0))
    tx, ty = tx.to(DEVICE), ty.to(DEVICE)
    subsets = [s.to(DEVICE) for s in subsets]
    gen = torch.Generator(device='cpu').manual_seed(7)
    for _ in range(STEPS):
        rows = torch.stack([s[torch.randint(0, len(s), (BATCH,), generator=gen)]
                            for s in subsets])                     # [M, B]
        xb, yb = tx[rows], ty[rows]                                # [M, B, 784]
        grads = grad_fn(params, buffers, xb, yb)
        opt.zero_grad()
        for k, p in params.items():
            p.grad = grads[k]
        opt.step()
    return params, buffers, base


@torch.no_grad()
def eval_stack(params, buffers, base, ex, ey, chunk=1000):
    fwd = vmap(lambda p, b, x: functional_call(base, (p, b), (x,)),
               in_dims=(0, 0, None))
    ex, ey = ex.to(DEVICE), ey.to(DEVICE)
    correct = None
    for i in range(0, ex.shape[0], chunk):
        pred = fwd(params, buffers, ex[i:i + chunk]).argmax(-1)     # [M, c]
        hit = (pred == ey[i:i + chunk]).sum(-1)
        correct = hit if correct is None else correct + hit
    return (correct.float() / ex.shape[0]).cpu()                    # [M]


def main():
    tx, ty, ex, ey = load_mnist()
    ex_rot = rotated(ex)
    gen = torch.Generator().manual_seed(42)
    # subsets[si][seed], shared across all arms (paired comparison)
    subsets = [[balanced_subset(ty, n, gen) for _ in range(SEEDS)]
               for n in SIZES]
    flat_subsets = [s for group in subsets for s in group]

    P = 784
    arms = {
        'matrix': lambda: VectorMLP(P, HIDDEN, 10, DIM, channel_mix='matrix'),
        'ring':   lambda: VectorMLP(P, HIDDEN, 10, DIM, channel_mix='ring',
                                    kernel_size=5),
        'shared': lambda: VectorMLP(P, HIDDEN, 10, DIM, channel_mix='shared'),
    }
    for ref in ('matrix', 'ring'):
        w, _ = matched_mlp_width(count_params(arms[ref]()), P, 10, len(HIDDEN))
        arms[f'mlp@{ref}'] = (lambda w=w: PlainMLP(P, [w] * len(HIDDEN), 10))

    results = {}
    for name, factory in arms.items():
        models = make_models(factory)
        n_par = count_params(models[0])
        print(f'--- {name}: {n_par} params x {len(models)} models', flush=True)
        params, buffers, base = train_stack(models, flat_subsets, tx, ty)
        acc = eval_stack(params, buffers, base, ex, ey).reshape(len(SIZES), SEEDS)
        acc_r = eval_stack(params, buffers, base, ex_rot, ey).reshape(len(SIZES), SEEDS)
        results[name] = {'params': n_par,
                         'clean': acc.tolist(), 'rot45': acc_r.tolist()}
        for si, n in enumerate(SIZES):
            print(f'  n={n:<6} clean {acc[si].mean():.4f}+-{acc[si].std():.4f}'
                  f'   rot45 {acc_r[si].mean():.4f}+-{acc_r[si].std():.4f}',
                  flush=True)

    with open(OUT_JSON, 'w') as f:
        json.dump({'sizes': SIZES, 'seeds': SEEDS, 'steps': STEPS,
                   'dim': DIM, 'hidden': HIDDEN, 'results': results}, f, indent=1)
    print(f'\nsaved -> {OUT_JSON}')


if __name__ == '__main__':
    main()
