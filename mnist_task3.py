"""MNIST grid, round 3: D-sweep upward (32, 64) + low-rank factorized mixing.

Low-rank mixer: y = g + U(V g), U [D,r], V [r,D] — 2*D*r params/neuron vs D*D.
At the 105k budget, full-matrix D=64 is degenerate (matched width = 1 neuron:
on-ramp 50k + readout matrices 41k), so low-rank is what makes D>=32 viable.

Arms (width-matched to round 1's matrix arm, same seeds/subsets -> paired):
    lowrank-d16-r4   does rank-4 mixing keep matrix's edge at the flagship D?
    matrix-d32       full DxD at D=32 (width collapses to 23)
    lowrank-d32-r4
    lowrank-d64-r4
"""

import json

import torch

from vector_mlp import VectorMLP, count_params, matched_width
from mnist_task import (DIM, HIDDEN, SIZES, SEEDS, load_mnist, rotated,
                        balanced_subset, make_models, train_stack, eval_stack)

OUT_JSON = r'C:\Users\JmgLi\documents\Python Projects\VectorMLP\mnist_results3.json'
P = 784

SPECS = {
    'lowrank-d16-r4': dict(dim=16, channel_mix='lowrank', rank=4),
    'matrix-d32':     dict(dim=32, channel_mix='matrix'),
    'lowrank-d32-r4': dict(dim=32, channel_mix='lowrank', rank=4),
    'lowrank-d64-r4': dict(dim=64, channel_mix='lowrank', rank=4),
}


def arm_factories():
    target = count_params(VectorMLP(P, HIDDEN, 10, DIM, channel_mix='matrix'))
    arms = {}
    for name, kw in SPECS.items():
        kw = dict(kw)
        dim = kw.pop('dim')
        w, got = matched_width(
            target, lambda w: VectorMLP(P, [w] * len(HIDDEN), 10, dim, **kw))
        print(f'{name}: width {w}, {got} params (target {target})', flush=True)
        arms[name] = (lambda w=w, dim=dim, kw=kw:
                      VectorMLP(P, [w] * len(HIDDEN), 10, dim, **kw))
    return arms, target


def main():
    tx, ty, ex, ey = load_mnist()
    ex_rot = rotated(ex)
    gen = torch.Generator().manual_seed(42)   # identical subsets to rounds 1-2
    subsets = [[balanced_subset(ty, n, gen) for _ in range(SEEDS)]
               for n in SIZES]
    flat_subsets = [s for group in subsets for s in group]

    arms, target = arm_factories()
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
        json.dump({'sizes': SIZES, 'seeds': SEEDS, 'target_params': target,
                   'results': results}, f, indent=1)
    print(f'\nsaved -> {OUT_JSON}')


if __name__ == '__main__':
    main()
