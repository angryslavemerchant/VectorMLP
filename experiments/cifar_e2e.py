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

Round 2 (`python experiments/cifar_e2e.py 2`) swaps in the new neuron
architectures as heads on the same trainable backbone, head-param-matched to
the round-1 vec head:

    cnn-proj-d4   ProjNet head (Variant A: coupled magnitude/direction)
    cnn-tagw-d4   TagNet 'weighted' head (Variant B, agreement over weights)
    cnn-tagq-d4   TagNet 'query' head (Variant B, per-neuron query direction)

Round 3 (`... 3`): ProjNet controls — cnn-proj-d2, cnn-proj-d1 (geometry
control), cnn-projI-d4 (P frozen to identity), cnn-mlp-flop (head matched
to cnn-proj-d2's head FLOPs, params unconstrained).

Round 4 (`... 4`): multi-gate neuron (MGN, see mgn.py) head vs a
param-matched plain MLP head, same trainable backbone:

    cnn-mgn    MGNNet head (per-neuron learned SUM/AND/OR softmax mix,
               v1: per-synapse expansion)
    cnn-mgnv4  MGNv4Net head: project-then-reduce (k=2 learned features
               per neuron, AND/OR reduce over k instead of over n_in;
               k=2 not dim=16 — at k=dim the param-matched width
               collapses to 3-4 hidden units)
    cnn-mlp    param-matched plain MLP head (same target as round 1)

(cnn-mgnv2/cnn-mgnv3 — matmul-native intermediate attempts — were
dropped from this round once v4 superseded them; still defined in
mgn.py.)

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

from vector_mlp import (VectorMLP, PlainMLP, ProjNet, TagNet, count_params,
                        matched_mlp_width, matched_width, proj_flops,
                        matched_mlp_flops)
from mgn import MGNNet, MGNv4Net
from dendritic_linear import DendriticLinear
from staged_linear import StagedMLP
from swiglu import SwiGLU
from experiments.mnist_grid import balanced_subset, train_stack, eval_stack
from experiments.cifar_features import rotated  # also patches the HF mirror

ROUND = int(sys.argv[1]) if len(sys.argv) > 1 else 1
OUT_JSON = ROOT / 'results' / (
    'cifar_e2e_results.json' if ROUND == 1 else f'cifar_e2e_results{ROUND}.json')
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

    if ROUND == 1:
        mlp_w, mlp_par = matched_mlp_width(head_target, 2048, 10, len(HEAD_HIDDEN))
        print(f'vec head {head_target:,} | mlp head width {mlp_w}, {mlp_par:,} | '
              f'backbone {count_params(SmallCNN()):,}', flush=True)
        arms = {
            'cnn-vec': lambda: E2E(vec_head()),
            'cnn-mlp': lambda: E2E(PlainMLP(2048, [mlp_w] * len(HEAD_HIDDEN), 10)),
        }
    elif ROUND == 2:
        print(f'round 2: matching new-neuron heads to {head_target:,} params',
              flush=True)

        def new_arm(name, build):
            w, got = matched_width(head_target, build)
            print(f'{name}: head width {w}, {got:,} params', flush=True)
            return lambda: E2E(build(w))

        arms = {
            f'cnn-{name}': new_arm(name, build)
            for name, build in {
                'proj-d4': lambda w: ProjNet(2048, [w] * len(HEAD_HIDDEN), 10, 4),
                'tagw-d4': lambda w: TagNet(2048, [w] * len(HEAD_HIDDEN), 10, 4,
                                            mode='weighted'),
                'tagq-d4': lambda w: TagNet(2048, [w] * len(HEAD_HIDDEN), 10, 4,
                                            mode='query'),
            }.items()}
    elif ROUND == 3:
        # round 3 — ProjNet controls (see cifar_head_task.py round 3).
        # cnn-proj-d2 wasn't in round 2, so it runs here both as the D
        # ablation and as the FLOP reference for cnn-mlp-flop. The FLOP
        # match is head-only (backbones are identical).
        def new_arm(name, build):
            w, got = matched_width(head_target, build)
            print(f'{name}: head width {w}, {got:,} params', flush=True)
            return lambda: E2E(build(w))

        w2, _ = matched_width(head_target, lambda w: ProjNet(
            2048, [w] * len(HEAD_HIDDEN), 10, 2))
        flop_target = proj_flops(2048, [w2] * len(HEAD_HIDDEN), 10, 2)
        fw, ff = matched_mlp_flops(flop_target, 2048, 10, len(HEAD_HIDDEN))
        fpar = count_params(PlainMLP(2048, [fw] * len(HEAD_HIDDEN), 10))
        print(f'round 3: head flop target {flop_target:,} (proj-d2 w{w2}) -> '
              f'mlp-flop head width {fw}, {ff:,} flops, {fpar:,} params',
              flush=True)
        arms = {
            'cnn-proj-d2':  new_arm('proj-d2', lambda w: ProjNet(
                                2048, [w] * len(HEAD_HIDDEN), 10, 2)),
            'cnn-proj-d1':  new_arm('proj-d1', lambda w: ProjNet(
                                2048, [w] * len(HEAD_HIDDEN), 10, 1)),
            'cnn-projI-d4': new_arm('projI-d4', lambda w: ProjNet(
                                2048, [w] * len(HEAD_HIDDEN), 10, 4,
                                learn_proj=False)),
            'cnn-mlp-flop': lambda: E2E(
                PlainMLP(2048, [fw] * len(HEAD_HIDDEN), 10)),
        }
    elif ROUND == 4:
        # round 4 — multi-gate neuron (MGN v1 and v4) heads vs param-matched
        # plain MLP.
        mlp_w, mlp_par = matched_mlp_width(head_target, 2048, 10, len(HEAD_HIDDEN))
        print(f'round 4: mgn head target {head_target:,} | mlp head width '
              f'{mlp_w}, {mlp_par:,} params', flush=True)

        def new_arm(name, build):
            w, got = matched_width(head_target, build)
            print(f'{name}: head width {w}, {got:,} params', flush=True)
            return lambda: E2E(build(w))

        arms = {
            'cnn-mgn': new_arm('mgn', lambda w: MGNNet(
                           2048, [w] * len(HEAD_HIDDEN), 10)),
            'cnn-mgnv4': new_arm('mgnv4', lambda w: MGNv4Net(
                           2048, [w] * len(HEAD_HIDDEN), 10, 2)),
            'cnn-mlp': lambda: E2E(
                           PlainMLP(2048, [mlp_w] * len(HEAD_HIDDEN), 10)),
        }
    elif ROUND == 5:
        # round 5 — a single DendriticLinear as the whole head (its internal
        # dendrite stage already plays the role of a hidden layer), fed into
        # a plain Linear classifier from the dendritic module's output size
        # to the class count. DEND_OUT is separate from the dendritic
        # module's own construction so its width can change independently
        # (fan_in K=16, coverage=2 -> each soma's dendrites collectively
        # tile the 2048 inputs twice) without touching the classifier.
        # Compared against the shape-matched dense MLP the layer's own
        # docstring describes: Linear(2048, M) -> ReLU -> Linear(M, DEND_OUT)
        # -> Linear(DEND_OUT, 10), M = out_features * D dendrites.
        DEND_OUT = 10

        def build_dend():
            dend = DendriticLinear(2048, DEND_OUT, fan_in=16, coverage=2)
            return nn.Sequential(dend, nn.Linear(DEND_OUT, 10))

        dend = DendriticLinear(2048, DEND_OUT, fan_in=16, coverage=2)
        mlp = PlainMLP(2048, [dend.M], DEND_OUT)
        print(f'round 5: dendritic head {count_params(dend):,} params '
              f'(K={dend.K}, D={dend.D}, M={dend.M}) | '
              f'shape-matched mlp head {count_params(mlp):,} params', flush=True)
        arms = {
            'cnn-dendritic': lambda: E2E(build_dend()),
            'cnn-mlp-matched': lambda: E2E(nn.Sequential(
                               PlainMLP(2048, [dend.M], DEND_OUT),
                               nn.Linear(DEND_OUT, 10))),
        }
    else:
        # round 6 — StagedLinear head (extra_stages=1: each neuron gets one
        # extra learned scale-shift-leaky_relu bend on top of the base
        # Linear+leaky_relu) vs param-matched plain MLP. 2 hidden layers
        # (same depth as HEAD_HIDDEN), width found via matched_width since
        # extra_stages adds a tiny per-neuron param cost on top of the same
        # matmul. Final classifier is a bare nn.Linear — StagedLinear always
        # leaky_relus internally, which would clip logits if used last.
        mlp_w, mlp_par = matched_mlp_width(head_target, 2048, 10, len(HEAD_HIDDEN))
        print(f'round 6: staged-linear head target {head_target:,} | '
              f'mlp head width {mlp_w}, {mlp_par:,} params', flush=True)

        def new_arm(name, build):
            w, got = matched_width(head_target, build)
            print(f'{name}: head width {w}, {got:,} params', flush=True)
            return lambda: E2E(build(w))

        arms = {
            'cnn-staged1': new_arm('staged1', lambda w: StagedMLP(
                           2048, [w] * len(HEAD_HIDDEN), 10, extra_stages=1)),
        }
        # SwiGLU gated FFN block, single param-matched stand-in for the
        # whole head (same convention as DendriticLinear in round 5), not
        # stacked to HEAD_HIDDEN's depth.
        arms['cnn-swiglu'] = new_arm('swiglu', lambda w: SwiGLU(2048, w, 10))
        arms['cnn-mlp'] = lambda: E2E(
                               PlainMLP(2048, [mlp_w] * len(HEAD_HIDDEN), 10))

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
