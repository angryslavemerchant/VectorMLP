"""MNIST grid, round 4: pure low-rank factorization ablation.

Round 3's lowrank was residual (y = g + UVg): full-rank map, rank-r *learnable
mixing*. This round tests the pure factorization (y = UVg): the whole mixer is
rank-r, so each neuron's channel state is compressed to r numbers between
layers. Prediction from rounds 2-3: keeps the clean edge, loses more of the
rot45 edge than residual rank-4 did.

Same target/seeds/subsets as rounds 1-3 -> paired.
"""

import json

import torch

from vector_mlp import VectorMLP, count_params, matched_width
from mnist_task import (DIM, HIDDEN, SIZES, SEEDS, load_mnist, rotated,
                        balanced_subset, make_models, train_stack, eval_stack)

OUT_JSON = r'C:\Users\JmgLi\documents\Python Projects\VectorMLP\mnist_results4.json'
P = 784

SPECS = {
    'purelr-d16-r4': dict(dim=16, channel_mix='lowrank_pure', rank=4),
    'purelr-d16-r8': dict(dim=16, channel_mix='lowrank_pure', rank=8),
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
    gen = torch.Generator().manual_seed(42)   # identical subsets to rounds 1-3
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
