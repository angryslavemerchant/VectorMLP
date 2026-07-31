"""Channel-mixing vector neuron layers (spec: channel_mixing_neuron_spec.md).

Each neuron carries a D-dim vector. A layer does:
    mix across neurons (scalar weights) -> per-channel ReLU gate -> mix across
    channels (per-neuron DxD matrix, or K-tap circular conv in the ring variant).

Also provides a plain-MLP baseline and a parameter matcher so both models can
be built at the same total parameter count.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorLinear(nn.Module):
    """One vector-neuron layer: [B, N_in, D] -> [B, N_out, D].

    channel_mix:
        'matrix' - per-neuron DxD matrix M, y = M @ g
        'ring'   - per-neuron K-tap circular conv over the channel axis
        'shared' - single DxD matrix shared by all neurons in the layer
        'none'   - no channel mixing (elementwise variant, ablation arm)
    gate_order:
        'gate_then_mix' (default) or 'mix_then_gate'
    per_channel_bias:
        If True, b is [N_out, D]. If False, b is [N_out] broadcast over
        channels. Default False for 'ring' (a per-channel bias breaks shift
        equivariance), True otherwise.
    tie_groups:
        Optional LongTensor [n_in] assigning each input to a weight-sharing
        group; inputs in the same group share one scalar weight per output
        neuron. Used to make a first layer's spatial weights rotation-invariant
        (tie pixels at equal radius) so image rotation stays a pure channel
        shift.
    """

    def __init__(self, n_in, n_out, dim, channel_mix='matrix', kernel_size=5,
                 gate_order='gate_then_mix', per_channel_bias=None,
                 bias_init=-0.1, tie_groups=None, rank=4):
        super().__init__()
        assert channel_mix in ('matrix', 'lowrank', 'ring', 'shared', 'none')
        assert gate_order in ('gate_then_mix', 'mix_then_gate')
        if per_channel_bias is None:
            per_channel_bias = channel_mix != 'ring'
        self.n_in, self.n_out, self.dim = n_in, n_out, dim
        self.channel_mix = channel_mix
        self.gate_order = gate_order

        # He-style: preserves per-channel variance through the ReLU gate.
        # (nn.Linear's default init is far too small here and lets the
        # negative bias shut every gate within two layers.)
        if tie_groups is not None:
            assert tie_groups.shape == (n_in,)
            self.register_buffer('tie_index', tie_groups.long())
            n_groups = int(tie_groups.max().item()) + 1
            self.weight = nn.Parameter(
                torch.randn(n_groups, n_out) * math.sqrt(2.0 / n_in))
        else:
            self.tie_index = None
            self.weight = nn.Parameter(
                torch.randn(n_in, n_out) * math.sqrt(2.0 / n_in))

        bias_shape = (n_out, dim) if per_channel_bias else (n_out, 1)
        self.bias = nn.Parameter(torch.full(bias_shape, float(bias_init)))

        if channel_mix == 'matrix':
            # identity + noise: start near "no channel mixing"
            eye = torch.eye(dim).expand(n_out, dim, dim)
            self.mix = nn.Parameter(eye + 0.01 * torch.randn(n_out, dim, dim))
        elif channel_mix == 'lowrank':
            # residual low-rank: y = g + U (V g). Starts near identity like
            # 'matrix'; learnable mixing is rank-`rank`. 2*D*rank params/neuron
            # instead of D*D.
            self.rank = rank
            self.mix_u = nn.Parameter(0.01 * torch.randn(n_out, dim, rank))
            self.mix_v = nn.Parameter(0.01 * torch.randn(n_out, rank, dim))
            self.mix = None
        elif channel_mix == 'shared':
            self.mix = nn.Parameter(torch.eye(dim) + 0.01 * torch.randn(dim, dim))
        elif channel_mix == 'ring':
            # even kernels allowed (e.g. K=dim, the full circulant); their tap
            # window is off-center by half a step, which circular conv doesn't
            # mind — shift equivariance holds for any tap layout
            assert kernel_size <= dim
            self.kernel_size = kernel_size
            # delta kernel + noise: starts near identity
            k = 0.01 * torch.randn(n_out, 1, kernel_size)
            k[:, 0, kernel_size // 2] += 1.0
            self.mix = nn.Parameter(k)
        else:
            self.mix = None

    def _mix_channels(self, g):
        if self.channel_mix == 'matrix':
            return torch.einsum('mde,bme->bmd', self.mix, g)
        if self.channel_mix == 'lowrank':
            low = torch.einsum('mre,bme->bmr', self.mix_v, g)
            return g + torch.einsum('mdr,bmr->bmd', self.mix_u, low)
        if self.channel_mix == 'shared':
            return torch.einsum('de,bme->bmd', self.mix, g)
        if self.channel_mix == 'ring':
            p = self.kernel_size // 2
            g = F.pad(g, (p, self.kernel_size - 1 - p), mode='circular')
            return F.conv1d(g, self.mix, groups=self.n_out)
        return g

    def forward(self, x):
        w = self.weight if self.tie_index is None else self.weight[self.tie_index]
        r = torch.einsum('bnd,nm->bmd', x, w)
        if self.gate_order == 'gate_then_mix':
            return self._mix_channels(F.relu(r + self.bias))
        return F.relu(self._mix_channels(r) + self.bias)


class VectorMLP(nn.Module):
    """Scalar inputs -> vector hidden layers -> class logits.

    On-ramp:  per-input direction, v_i = x_i * p_i. Learned by default; pass
              proj_init (e.g. a hand-wired circular encoding) to set it, and
              freeze_proj=True to keep it fixed.
    Off-ramp: final VectorLinear maps to num_classes neurons, then
              readout='dirs'   - logit_k = class vector k . learned direction c_k
              readout='pooled' - logit_k = mean over channels of class vector k
                                 (shift-invariant: preserves ring equivariance)
    """

    def __init__(self, in_features, hidden, num_classes, dim,
                 channel_mix='matrix', proj_init=None, freeze_proj=False,
                 readout='dirs', tie_first=None, **layer_kw):
        super().__init__()
        assert readout in ('dirs', 'pooled')
        # unit-std per channel so hidden activations start at O(1) scale
        proj = torch.randn(in_features, dim) if proj_init is None else proj_init.clone()
        self.proj_in = nn.Parameter(proj, requires_grad=not freeze_proj)
        widths = [in_features] + list(hidden)
        self.layers = nn.ModuleList(
            VectorLinear(a, b, dim, channel_mix=channel_mix,
                         tie_groups=tie_first if i == 0 else None, **layer_kw)
            for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])))
        self.readout = VectorLinear(widths[-1], num_classes, dim,
                                    channel_mix=channel_mix, **layer_kw)
        self.readout_mode = readout
        if readout == 'dirs':
            self.class_dirs = nn.Parameter(torch.randn(num_classes, dim) / math.sqrt(dim))

    def hidden(self, x):
        """Last hidden layer's vectors [B, N, D] (for probes)."""
        v = x.unsqueeze(-1) * self.proj_in
        for layer in self.layers:
            v = layer(v)
        return v

    def forward(self, x):
        v = self.readout(self.hidden(x))              # [B, C, D]
        if self.readout_mode == 'pooled':
            return v.mean(-1)                         # [B, C]
        return (v * self.class_dirs).sum(-1)          # [B, C]


class PlainMLP(nn.Module):
    """Param-matched baseline: ReLU MLP over the raw scalar inputs."""

    def __init__(self, in_features, hidden, num_classes):
        super().__init__()
        widths = [in_features] + list(hidden)
        self.body = nn.Sequential(*[
            m for a, b in zip(widths[:-1], widths[1:])
            for m in (nn.Linear(a, b), nn.ReLU())])
        self.head = nn.Linear(widths[-1], num_classes)

    def forward(self, x):
        return self.head(self.body(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_mlp_width(target_params, in_features, num_classes, depth):
    """Widest uniform-width ReLU MLP (`depth` hidden layers) whose parameter
    count does not exceed target_params. Returns (width, param_count)."""
    lo, hi = 1, 1
    def n_params(w):
        return count_params(PlainMLP(in_features, [w] * depth, num_classes))
    while n_params(hi) <= target_params:
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if n_params(mid) <= target_params:
            lo = mid
        else:
            hi = mid
    return lo, n_params(lo)


def matched_width(target_params, build):
    """Widest uniform hidden width w such that build(w) has at most
    target_params parameters. build: int -> nn.Module. Returns (w, params)."""
    lo, hi = 1, 1
    def n_params(w):
        return count_params(build(w))
    while n_params(hi) <= target_params:
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if n_params(mid) <= target_params:
            lo = mid
        else:
            hi = mid
    return lo, n_params(lo)


def build_matched_pair(in_features, hidden, num_classes, dim, depth=None,
                      channel_mix='matrix', **layer_kw):
    """Build a VectorMLP and a plain MLP of (at most) equal parameter count.

    The MLP's uniform hidden width is solved to match the vector net's total.
    """
    vnet = VectorMLP(in_features, hidden, num_classes, dim,
                     channel_mix=channel_mix, **layer_kw)
    target = count_params(vnet)
    depth = depth if depth is not None else len(hidden)
    width, got = matched_mlp_width(target, in_features, num_classes, depth)
    mlp = PlainMLP(in_features, [width] * depth, num_classes)
    return vnet, mlp, {'vector_params': target, 'mlp_params': got,
                       'mlp_width': width, 'gap': target - got}
