"""MNIST grid, round 2: ring-kernel ladder + matrix D-sweep (see
experiment_results.md, Queue).

Arms, all width-matched to round 1's matrix arm (~105k params) so the
existing matrix / mlp@matrix numbers remain the paired baselines:
    ring3       D=16, K=3 circular kernel  (maximally local mixing)
    ring16      D=16, K=16 full circulant  (ceiling of the ring family)
    matrix-d8   D=8 per-neuron matrices
    matrix-d4   D=4 per-neuron matrices

Same seeds, same subset generator (seed 42, same construction order) as
round 1 -> per-seed paired with mnist_results.json.
"""

import json

import torch

from vector_mlp import VectorMLP, count_params, matched_width
from mnist_task import (DIM, HIDDEN, SIZES, SEEDS, load_mnist, rotated,
                        balanced_subset, make_models, train_stack, eval_stack)

OUT_JSON = r'C:\Users\JmgLi\documents\Python Projects\VectorMLP\mnist_results2.json'
P = 784


def arm_factories():
    target = count_params(VectorMLP(P, HIDDEN, 10, DIM, channel_mix='matrix'))
    specs = {
        'ring3':     dict(dim=16, channel_mix='ring', kernel_size=3),
        'ring16':    dict(dim=16, channel_mix='ring', kernel_size=16),
        'matrix-d8': dict(dim=8, channel_mix='matrix'),
        'matrix-d4': dict(dim=4, channel_mix='matrix'),
    }
    arms = {}
    for name, kw in specs.items():
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
    gen = torch.Generator().manual_seed(42)   # identical subsets to round 1
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
