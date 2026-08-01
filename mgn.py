"""Multi-gate neuron (MGN) layers (spec: multi-gate-neuron-spec.md).

Each neuron computes three parallel reductions over the same weighted
inputs z = w * x and mixes them with a learned per-neuron softmax gate:

    s_sum = sum(z) + b                      accumulation (standard neuron)
    s_and = exp(mean(logsigmoid(z)))        geometric mean of soft truth
                                            values: any low input drags the
                                            output toward 0 (conjunction)
    s_or  = logsumexp(tau * z) / tau        smooth max: one strong input
                                            suffices (disjunction)

    gate  = softmax([alpha, beta, gamma])   per neuron
    out   = gate . [s_sum, s_and, s_or]

s_and lives in (0, 1) and s_or tracks max(z), while s_sum is unbounded, so
by default each non-SUM path gets a learned per-neuron affine (a * s + c,
init identity) to lift it onto a scale that competes fairly with SUM — and
a negative learned scale gives NAND/NOR for free (path_affine=True).

Follows new_neuron_guide.md: MGNLinear is one hidden layer, MGNNet is the
full model. The signal between layers is plain scalar activations [B, N]
(the degenerate case of the guide's "signal shape" — no on-ramp needed,
off-ramp is a plain linear head).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MGNLinear(nn.Module):
    """One multi-gate layer: [B, n_in] -> [B, n_out] (pre-activation).

    dim is accepted for harness-interface uniformity and ignored — MGN is a
    scalar neuron family, there is no channel axis.

    tau_init:      initial OR-path temperature (learnable, stored as log).
    sum_bias_init: initial SUM-gate logit. 0.0 gives an equal [1/3,1/3,1/3]
                   mixture; positive starts the layer near a plain linear
                   layer, discovering AND/OR only if the loss rewards it.
    path_affine:   learned per-neuron scale+shift on the AND and OR paths
                   (+4 scalars/neuron), fixing the SUM/AND/OR scale
                   mismatch and allowing negative (NAND/NOR) responses.

    Memory note: the AND/OR paths expand to [B, n_out, n_in]. Fine at MLP
    scale; chunk over n_out if it ever isn't.
    """

    def __init__(self, n_in, n_out, dim=None, tau_init=5.0,
                 sum_bias_init=0.0, path_affine=True):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out, bias=True)

        mix = torch.zeros(n_out, 3)
        mix[:, 0] = sum_bias_init
        self.mix_logits = nn.Parameter(mix)                 # [n_out, 3]

        self.log_tau = nn.Parameter(
            torch.full((n_out,), math.log(tau_init)))

        self.path_affine = path_affine
        if path_affine:
            self.and_scale = nn.Parameter(torch.ones(n_out))
            self.and_shift = nn.Parameter(torch.zeros(n_out))
            self.or_scale = nn.Parameter(torch.ones(n_out))
            self.or_shift = nn.Parameter(torch.zeros(n_out))

    def forward(self, x):
        w = self.linear.weight                              # [n_out, n_in]

        # SUM path: plain matmul, no expansion.
        s_sum = F.linear(x, w, self.linear.bias)            # [B, n_out]

        # Shared weighted inputs for the AND/OR reductions.
        z = x.unsqueeze(-2) * w                             # [B, n_out, n_in]

        # AND: geometric mean of sigmoid(z), computed in log space.
        s_and = torch.exp(F.logsigmoid(z).mean(-1))         # [B, n_out]

        # OR: temperature-scaled logsumexp ~ smooth max over inputs.
        tau = self.log_tau.exp()                            # [n_out]
        s_or = torch.logsumexp(tau.unsqueeze(-1) * z, -1) / tau

        if self.path_affine:
            s_and = self.and_scale * s_and + self.and_shift
            s_or = self.or_scale * s_or + self.or_shift

        gate = F.softmax(self.mix_logits, dim=-1)           # [n_out, 3]
        return gate[:, 0] * s_sum + gate[:, 1] * s_and + gate[:, 2] * s_or

    def gate_distribution(self):
        """Per-neuron softmax gate, detached: [n_out, 3] = (SUM, AND, OR)."""
        return F.softmax(self.mix_logits, dim=-1).detach()


class MGNNet(nn.Module):
    """Full model: scalar features -> MGN hidden layers -> class logits.

    Signal between layers is plain scalars [B, N]; each hidden layer is
    MGNLinear followed by ReLU (the spec's post-mixture activation).
    Off-ramp is a plain linear head on the last hidden activations.

    dim is accepted per the harness interface and ignored (scalar family).
    Layer kwargs (tau_init, sum_bias_init, path_affine) pass through **kw.
    """

    def __init__(self, in_features, hidden, num_classes, dim=None, **kw):
        super().__init__()
        widths = [in_features] + list(hidden)
        self.layers = nn.ModuleList(
            MGNLinear(a, b, dim, **kw)
            for a, b in zip(widths[:-1], widths[1:]))
        self.head = nn.Linear(widths[-1], num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.head(x)

    def gate_distributions(self):
        """List of per-layer gate tensors [n_out, 3], for histograms."""
        return [layer.gate_distribution() for layer in self.layers]


class MGNv2Linear(nn.Module):
    """Matmul-native multi-gate layer: [B, n_in] -> [B, n_out].

    Same three-path/gated-mixture idea as MGNLinear, but the squash is
    applied to the INPUT before weighting, so every path is one plain
    matmul with the shared weight matrix — no [B, n_out, n_in] expansion,
    no custom kernels, tensor-core friendly (~3x a plain linear layer).

    With p = sigmoid(in_scale * x + in_shift) (per-input learned squash,
    shared across output neurons) and u its logit:

        s_sum = W @ x + b                       unchanged
        s_and = exp((W @ log p) / n)            weighted geometric mean:
                                                prod p_i^(w_i/n)
        s_or  = 1 - exp((W @ log(1 - p)) / n)   normalized noisy-OR:
                                                1 - prod (1-p_i)^(w_i/n)

    Weights act as log-space exponents ("how much this input's truth value
    counts"), and a NEGATIVE weight is soft negation: p^(-|w|) fires when
    the input is off. Both /n normalizations play the same role as the
    spec's geometric mean: without them the products vanish/saturate as
    fan-in grows. What v1 has and v2 gives up: per-synapse truth
    thresholds (v1's sigmoid(w*x) lets each connection place its own
    threshold; here the per-input affine is shared across output neurons).

    The exponent W @ log p / n can go large-positive when negative weights
    meet p ~ 0, so it is clamped before exp for stability.
    """

    EXP_CLAMP = 20.0

    def __init__(self, n_in, n_out, dim=None, sum_bias_init=0.0,
                 path_affine=True):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out, bias=True)

        self.in_scale = nn.Parameter(torch.ones(n_in))
        self.in_shift = nn.Parameter(torch.zeros(n_in))

        mix = torch.zeros(n_out, 3)
        mix[:, 0] = sum_bias_init
        self.mix_logits = nn.Parameter(mix)                 # [n_out, 3]

        self.path_affine = path_affine
        if path_affine:
            self.and_scale = nn.Parameter(torch.ones(n_out))
            self.and_shift = nn.Parameter(torch.zeros(n_out))
            self.or_scale = nn.Parameter(torch.ones(n_out))
            self.or_shift = nn.Parameter(torch.zeros(n_out))

    def forward(self, x):
        n = x.shape[-1]
        u = self.in_scale * x + self.in_shift               # [B, n_in]
        log_p = F.logsigmoid(u) / n                         # log p, pre-scaled
        log_q = F.logsigmoid(-u) / n                        # log(1 - p)

        # One matmul for all three paths (shared W, stacked along dim 0).
        stacked = torch.stack([x, log_p, log_q], 0)         # [3, B, n_in]
        h = F.linear(stacked, self.linear.weight)           # [3, B, n_out]

        s_sum = h[0] + self.linear.bias
        s_and = torch.exp(h[1].clamp(max=self.EXP_CLAMP))
        s_or = 1 - torch.exp(h[2].clamp(max=self.EXP_CLAMP))

        if self.path_affine:
            s_and = self.and_scale * s_and + self.and_shift
            s_or = self.or_scale * s_or + self.or_shift

        gate = F.softmax(self.mix_logits, dim=-1)           # [n_out, 3]
        return gate[:, 0] * s_sum + gate[:, 1] * s_and + gate[:, 2] * s_or

    def gate_distribution(self):
        """Per-neuron softmax gate, detached: [n_out, 3] = (SUM, AND, OR)."""
        return F.softmax(self.mix_logits, dim=-1).detach()


class MGNv2Net(nn.Module):
    """Full model over MGNv2Linear layers; interface identical to MGNNet."""

    def __init__(self, in_features, hidden, num_classes, dim=None, **kw):
        super().__init__()
        widths = [in_features] + list(hidden)
        self.layers = nn.ModuleList(
            MGNv2Linear(a, b, dim, **kw)
            for a, b in zip(widths[:-1], widths[1:]))
        self.head = nn.Linear(widths[-1], num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.head(x)

    def gate_distributions(self):
        """List of per-layer gate tensors [n_out, 3], for histograms."""
        return [layer.gate_distribution() for layer in self.layers]
