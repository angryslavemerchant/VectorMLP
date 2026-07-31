"""Sanity checks for vector_mlp.py: shapes, gradients, param matching,
ring shift-equivariance, and a tiny overfit run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from vector_mlp import (VectorLinear, VectorMLP, PlainMLP, count_params,
                        build_matched_pair)

torch.manual_seed(0)
B, N_IN, N_OUT, D = 4, 12, 8, 16


def check(name, ok, detail=''):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    assert ok, name


# --- shapes + gradients, all variants ---
x = torch.randn(B, N_IN, D, requires_grad=True)
for mix in ('matrix', 'ring', 'shared', 'none'):
    for order in ('gate_then_mix', 'mix_then_gate'):
        layer = VectorLinear(N_IN, N_OUT, D, channel_mix=mix, gate_order=order)
        y = layer(x)
        check(f'shape {mix}/{order}', y.shape == (B, N_OUT, D))
        y.sum().backward()
        grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in layer.parameters())
        check(f'grads {mix}/{order}', grads_ok)
        x.grad = None

# --- gate actually gates: negative bias should zero some channels ---
layer = VectorLinear(N_IN, N_OUT, D, channel_mix='none', bias_init=-0.5)
g = layer(torch.randn(256, N_IN, D))
frac_zero = (g == 0).float().mean().item()
check('gate sparsity', 0.05 < frac_zero < 0.99, f'zeros={frac_zero:.2f}')

# --- ring shift-equivariance: shift input channels -> output shifts equally ---
layer = VectorLinear(N_IN, N_OUT, D, channel_mix='ring', kernel_size=5)
xa = torch.randn(B, N_IN, D)
s = 3
ya_shift = torch.roll(layer(xa), s, dims=-1)
y_shifted = layer(torch.roll(xa, s, dims=-1))
err = (ya_shift - y_shifted).abs().max().item()
check('ring shift equivariance', err < 1e-5, f'max err={err:.2e}')

# --- and confirm per-channel bias breaks it (expected) ---
# (at init the bias is constant across channels, so perturb it the way
# training would before checking)
layer = VectorLinear(N_IN, N_OUT, D, channel_mix='ring', per_channel_bias=True)
with torch.no_grad():
    layer.bias.add_(0.3 * torch.randn_like(layer.bias))
err = (torch.roll(layer(xa), s, -1) - layer(torch.roll(xa, s, -1))).abs().max().item()
check('per-channel bias breaks equivariance (expected)', err > 1e-3,
      f'max err={err:.2e}')

# --- param matcher ---
vnet, mlp, info = build_matched_pair(in_features=64, hidden=[32, 32],
                                     num_classes=10, dim=16)
rel_gap = info['gap'] / info['vector_params']
check('param match', 0 <= rel_gap < 0.02,
      f"vec={info['vector_params']} mlp={info['mlp_params']} "
      f"width={info['mlp_width']} gap={rel_gap:.3%}")

# --- tiny overfit: both nets memorize 64 random samples ---
def overfit(model, steps=400):
    xs = torch.randn(64, 64)
    ys = torch.randint(0, 10, (64,))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(xs), ys)
        loss.backward()
        opt.step()
    return (model(xs).argmax(-1) == ys).float().mean().item()

acc_v = overfit(vnet)
acc_m = overfit(mlp)
check('vector net overfits', acc_v > 0.95, f'acc={acc_v:.2f}')
check('plain mlp overfits', acc_m > 0.95, f'acc={acc_m:.2f}')

# ring version end-to-end too
vring = VectorMLP(64, [32, 32], 10, 16, channel_mix='ring')
acc_r = overfit(vring)
check('ring net overfits', acc_r > 0.95, f'acc={acc_r:.2f}')

print('\nall checks passed')
print(f"vector(matrix) params: {count_params(vnet)}")
print(f"vector(ring)   params: {count_params(vring)}")
print(f"plain mlp      params: {count_params(mlp)} (width {info['mlp_width']})")
