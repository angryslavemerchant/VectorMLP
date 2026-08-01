"""CIFAR-10 heads on frozen DINO features: the guest-in-a-scalar-world test.

The backbone's 384-dim features are reinterpreted as 24 vector neurons x 16
channels by pure reshape — no on-ramp, no learned interface. Arms:

    vec       VectorMLP head, contiguous grouping, pure rank-4 mixing
    vec-perm  same, but features first shuffled by a fixed random permutation
              (control: DINO features can't co-adapt, so contiguous vs
              permuted measures whether the feature vector has local
              channel structure our grouping accidentally exploits)
    mlp       param-matched plain MLP head on the raw 384 features

Round 2 (`python experiments/cifar_head_task.py 2`) swaps in the new neuron
architectures (magnitude-direction coupled / decoupled-tag, see spec), all
param-matched to the same round-1 vec-head target:

    proj-d2, proj-d4   Variant A ProjNet: signals are single D-vectors,
                       modReLU magnitude gate, per-neuron DxD projection
    tagw-d4            Variant B TagNet 'weighted': scalar path modulated by
                       per-neuron agreement over reused connection weights
    tagq-d4            Variant B TagNet 'query': shared layer field, per-
                       neuron query direction picks the agreement axis

Round 3 (`... 3`) asks what part of ProjNet is doing the work: proj-d1
(geometry control — collapses to a scalar net with a |z|-style activation),
projI-d4 (P frozen to identity), mlp-flop (MLP matched to proj-d2's FLOPs
instead of its params).

Round 4 (`... 4`) tests the multi-gate neuron (MGN, see mgn.py) against a
param-matched plain MLP:

    mgn    MGNNet head: per-neuron learned softmax mix of SUM/AND/OR
           reductions over the same weighted inputs (v1: per-synapse
           sigmoid(w*x) truth values, [B, n_out, n_in] expansion)
    mgnv4  MGNv4Net head: project-then-reduce — each neuron projects the
           input down to k=dim learned features via a plain matmul, then
           reduces AND/OR over those k instead of over n_in, so
           discrimination doesn't decay with fan-in the way v1-v3 do
    mlp    param-matched plain MLP head (same target as round 1)

(mgnv2/mgnv3 — matmul-native intermediate attempts — were dropped from
this round once v4 superseded them; still defined in mgn.py.)

Sample-efficiency sweep across train sizes, 5 seeds, vmap-stacked like the
MNIST grids. Run experiments/cifar_features.py once first.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from vector_mlp import (VectorMLP, PlainMLP, ProjNet, TagNet, count_params,
                        matched_mlp_width, matched_width, proj_flops,
                        matched_mlp_flops)
from mgn import MGNNet, MGNv4Net
from experiments.mnist_grid import (balanced_subset, make_models, train_stack,
                                    eval_stack)

ROUND = int(sys.argv[1]) if len(sys.argv) > 1 else 1
CACHE = ROOT / 'data' / 'cifar_dino_vits16.pt'
OUT_JSON = ROOT / 'results' / (
    'cifar_head_results.json' if ROUND == 1 else f'cifar_head_results{ROUND}.json')
FEAT, DIM = 384, 16
N_VEC = FEAT // DIM            # 24 vector neurons
HIDDEN = [64, 64]
SIZES = [500, 2000, 10000, 50000]
SEEDS = 5


def main():
    cache = torch.load(CACHE)
    tx, ty = cache['train_x'].float(), cache['train_y']
    ex, ey = cache['test_x'].float(), cache['test_y']
    ex_rot = cache['test_rot_x'].float()
    # z-score per feature dim (train stats, same transform for every arm) so
    # the unit-scale init/gate-bias recipe carries over from the MNIST grids
    mu, sd = tx.mean(0), tx.std(0) + 1e-6
    tx, ex, ex_rot = (tx - mu) / sd, (ex - mu) / sd, (ex_rot - mu) / sd

    gen = torch.Generator().manual_seed(42)
    subsets = [[balanced_subset(ty, n, gen) for _ in range(SEEDS)]
               for n in SIZES]
    flat_subsets = [s for group in subsets for s in group]
    perm = torch.randperm(FEAT, generator=torch.Generator().manual_seed(9))

    def vec_head():
        return VectorMLP(N_VEC, HIDDEN, 10, DIM, channel_mix='lowrank_pure',
                         rank=4, vector_in=True)

    target = count_params(vec_head())

    def new_arm(name, build):
        """Width-match `build(w)` to the round-1 vec-head param target."""
        w, got = matched_width(target, build)
        print(f'{name}: width {w}, {got:,} params', flush=True)
        return lambda: build(w)

    if ROUND == 1:
        mlp_w, mlp_par = matched_mlp_width(target, FEAT, 10, len(HIDDEN))
        # learned on-ramp diagnostic: per-feature 16-d directions (384 input
        # vectors, full per-feature input wiring) instead of the free reshape.
        # Separates "reshape interface failed" from "neuron fails on features".
        ramp_w, ramp_par = matched_width(
            target, lambda w: VectorMLP(FEAT, [w] * len(HIDDEN), 10, DIM,
                                        channel_mix='lowrank_pure', rank=4))
        print(f'vec head: {target:,} params | mlp head: width {mlp_w}, '
              f'{mlp_par:,} params | vec-onramp: width {ramp_w}, '
              f'{ramp_par:,} params', flush=True)

        # mixer ablations on the reshape base: 'none' = channels never interact
        # (16 disjoint thin MLPs, one per within-group position, meeting only at
        # the readout); 'ring' = K=5 circular conv over the 16 channel positions
        # (assumes neighboring features within a group are related — arbitrary
        # for DINO, so this measures structured-local vs none vs full mixing).
        def abl_head(mix, **kw):
            return new_arm(f'vec-{mix}', lambda w: VectorMLP(
                N_VEC, [w] * len(HIDDEN), 10, DIM,
                channel_mix=mix, vector_in=True, **kw))

        arms = {
            'vec':        (vec_head, tx, ex, ex_rot),
            'vec-perm':   (vec_head, tx[:, perm], ex[:, perm], ex_rot[:, perm]),
            'mlp':        (lambda: PlainMLP(FEAT, [mlp_w] * len(HIDDEN), 10),
                           tx, ex, ex_rot),
            'vec-onramp': (lambda: VectorMLP(FEAT, [ramp_w] * len(HIDDEN), 10, DIM,
                                             channel_mix='lowrank_pure', rank=4),
                           tx, ex, ex_rot),
            'vec-none':   (abl_head('none'), tx, ex, ex_rot),
            'vec-ring':   (abl_head('ring', kernel_size=5), tx, ex, ex_rot),
        }
    elif ROUND == 2:
        print(f'round 2: matching new-neuron arms to {target:,} params', flush=True)
        arms = {
            name: (new_arm(name, build), tx, ex, ex_rot)
            for name, build in {
                'proj-d2': lambda w: ProjNet(FEAT, [w] * len(HIDDEN), 10, 2),
                'proj-d4': lambda w: ProjNet(FEAT, [w] * len(HIDDEN), 10, 4),
                'tagw-d4': lambda w: TagNet(FEAT, [w] * len(HIDDEN), 10, 4,
                                            mode='weighted'),
                'tagq-d4': lambda w: TagNet(FEAT, [w] * len(HIDDEN), 10, 4,
                                            mode='query'),
            }.items()}
    elif ROUND == 3:
        # round 3 — is ProjNet real, and what part of it is doing the work?
        #   proj-d1   geometry control: at D=1 the net collapses to a scalar
        #             MLP with activation ReLU(|z|+b)*sign(z). If this
        #             matches proj-d2/d4, the win is the activation, not the
        #             vector geometry.
        #   projI-d4  P frozen to identity: gate = ||sum w_i x_i|| raw.
        #             Does the learned projection matter, or is interference
        #             over on-ramp directions alone enough?
        #   mlp-flop  FLOP-matched MLP (compute of round-2 proj-d2, params
        #             unconstrained). Kills the "it's just 2x compute"
        #             objection — the control owed since MNIST round 1.
        w2, _ = matched_width(target,
                              lambda w: ProjNet(FEAT, [w] * len(HIDDEN), 10, 2))
        flop_target = proj_flops(FEAT, [w2] * len(HIDDEN), 10, 2)
        fw, ff = matched_mlp_flops(flop_target, FEAT, 10, len(HIDDEN))
        fpar = count_params(PlainMLP(FEAT, [fw] * len(HIDDEN), 10))
        print(f'round 3: param target {target:,} | flop target {flop_target:,} '
              f'(proj-d2 w{w2}) -> mlp-flop width {fw}, {ff:,} flops, '
              f'{fpar:,} params', flush=True)
        arms = {
            'proj-d1':  (new_arm('proj-d1', lambda w: ProjNet(
                             FEAT, [w] * len(HIDDEN), 10, 1)), tx, ex, ex_rot),
            'projI-d4': (new_arm('projI-d4', lambda w: ProjNet(
                             FEAT, [w] * len(HIDDEN), 10, 4,
                             learn_proj=False)), tx, ex, ex_rot),
            'mlp-flop': (lambda: PlainMLP(FEAT, [fw] * len(HIDDEN), 10),
                         tx, ex, ex_rot),
        }
    else:
        # round 4 — multi-gate neuron (MGN v1 and v4) vs param-matched plain MLP.
        mlp_w, mlp_par = matched_mlp_width(target, FEAT, 10, len(HIDDEN))
        print(f'round 4: mgn target {target:,} | mlp width {mlp_w}, '
              f'{mlp_par:,} params', flush=True)
        arms = {
            'mgn': (new_arm('mgn', lambda w: MGNNet(
                        FEAT, [w] * len(HIDDEN), 10)), tx, ex, ex_rot),
            'mgnv4': (new_arm('mgnv4', lambda w: MGNv4Net(
                        FEAT, [w] * len(HIDDEN), 10, DIM)), tx, ex, ex_rot),
            'mlp': (lambda: PlainMLP(FEAT, [mlp_w] * len(HIDDEN), 10),
                    tx, ex, ex_rot),
        }

    results = {}
    for name, (factory, atx, aex, aex_rot) in arms.items():
        models = make_models(factory)
        print(f'--- {name}: {count_params(models[0])} params x {len(models)} models',
              flush=True)
        params, buffers, base = train_stack(models, flat_subsets, atx, ty)
        acc = eval_stack(params, buffers, base, aex, ey).reshape(len(SIZES), SEEDS)
        acc_r = eval_stack(params, buffers, base, aex_rot, ey).reshape(len(SIZES), SEEDS)
        results[name] = {'params': count_params(models[0]),
                         'clean': acc.tolist(), 'rot45': acc_r.tolist()}
        for si, n in enumerate(SIZES):
            print(f'  n={n:<6} clean {acc[si].mean():.4f}+-{acc[si].std():.4f}'
                  f'   rot45 {acc_r[si].mean():.4f}+-{acc_r[si].std():.4f}',
                  flush=True)

    with open(OUT_JSON, 'w') as f:
        json.dump({'sizes': SIZES, 'seeds': SEEDS, 'dim': DIM,
                   'hidden': HIDDEN, 'results': results}, f, indent=1)
    print(f'\nsaved -> {OUT_JSON}')


if __name__ == '__main__':
    main()
