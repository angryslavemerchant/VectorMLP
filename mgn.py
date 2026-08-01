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
        s_and = exp((W @ log p) / d)            weighted geometric mean:
                                                prod p_i^(w_i/d)
        s_or  = 1 - exp((W @ log(1 - p)) / d)   normalized noisy-OR:
                                                1 - prod (1-p_i)^(w_i/d)

    Weights act as log-space exponents ("how much this input's truth value
    counts"), and a NEGATIVE weight is soft negation: p^(-|w|) fires when
    the input is off. What v1 has and v2 gives up: per-synapse truth
    thresholds (v1's sigmoid(w*x) lets each connection place its own
    threshold; here the per-input affine is shared across output neurons).

    norm -- the divisor d, which keeps the log-space products from
    vanishing/saturating as fan-in grows:
        'l1' (default) d = sum_j |w_j| per neuron. Makes the exponent a
             true weighted average of log p, so the path has O(1) dynamic
             range and is invariant to overall weight magnitude.
        'l2' d = ||w||_2 per neuron. Also magnitude-invariant, and matches
             the SUM path's batch variance by construction. Exponents no
             longer sum to 1, so this is a SHARPENED AND/OR (geometric
             mean raised to ~sqrt(n)) rather than a literal weighted
             average -- which discriminates better at large fan-in, where
             a true geometric mean barely moves when one of n inputs drops.
        'n'  d = fan-in. The original formulation -- normalizes twice
             (W's random signs already cancel the sum to ~1/sqrt(n)), so
             AND/OR come out near-constant (~500x less signal than SUM at
             init) and path_affine has to learn a huge scale/shift pair to
             compensate. Kept for A/B against earlier runs.

    balance_init -- initialize the path affine so AND/OR start with the
    same batch variance as SUM. Without it AND/OR are quieter than SUM by
    a factor ~sqrt(n) under 'l1' (11x at n=128, 31x at n=1024), starving
    the gate of the gradient signal it needs to tell the paths apart.
    Sets scale = 1.67 * d (1.67 ~ 1/std(logsigmoid(x)) for unit-normal x)
    and shift to minus each path's analytic init mean (E[s_and] ~ 1,
    E[s_or] ~ 0) so both paths start centered. Requires path_affine.

    The exponent can go large-positive when negative weights meet p ~ 0,
    so it is clamped before exp for stability.
    """

    EXP_CLAMP = 20.0

    # 1 / std(logsigmoid(x)) for x ~ N(0, 1); sets the balanced init scale.
    BALANCE_C = 1.67

    def __init__(self, n_in, n_out, dim=None, sum_bias_init=0.0,
                 path_affine=True, norm='l1', balance_init=True):
        super().__init__()
        assert norm in ('l1', 'l2', 'n')
        self.norm = norm
        self.linear = nn.Linear(n_in, n_out, bias=True)

        self.in_scale = nn.Parameter(torch.ones(n_in))
        self.in_shift = nn.Parameter(torch.zeros(n_in))

        mix = torch.zeros(n_out, 3)
        mix[:, 0] = sum_bias_init
        self.mix_logits = nn.Parameter(mix)                 # [n_out, 3]

        self.path_affine = path_affine
        if path_affine:
            if balance_init:
                with torch.no_grad():
                    d = torch.as_tensor(self._divisor(), dtype=torch.float)
                scale = (self.BALANCE_C * d).expand(n_out).clone()
            else:
                scale = torch.ones(n_out)
            self.and_scale = nn.Parameter(scale.clone())
            self.or_scale = nn.Parameter(scale.clone())
            # centre each path on its analytic init mean
            self.and_shift = nn.Parameter(-scale.clone() if balance_init
                                          else torch.zeros(n_out))
            self.or_shift = nn.Parameter(torch.zeros(n_out))

    def _divisor(self):
        """Per-neuron log-space normalizer d: [n_out] tensor, or a float."""
        w = self.linear.weight
        if self.norm == 'l1':
            return w.abs().sum(-1) + 1e-6
        if self.norm == 'l2':
            return w.pow(2).sum(-1).sqrt() + 1e-6
        return float(w.shape[-1])

    def forward(self, x):
        w = self.linear.weight
        u = self.in_scale * x + self.in_shift               # [B, n_in]

        # One matmul for all three paths (shared W, stacked along dim 0).
        stacked = torch.stack([x, F.logsigmoid(u), F.logsigmoid(-u)], 0)
        h = F.linear(stacked, w)                            # [3, B, n_out]

        d = self._divisor()

        s_sum = h[0] + self.linear.bias
        s_and = torch.exp((h[1] / d).clamp(max=self.EXP_CLAMP))
        s_or = 1 - torch.exp((h[2] / d).clamp(max=self.EXP_CLAMP))

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


class MGNv3Linear(nn.Module):
    """Matmul-native layer with fan-in robust AND/OR: [B, n_in] -> [B, n_out].

    v2's log-probability paths stop working as fan-in grows: an "off" input
    contributes q ~ 1 to a noisy-OR, and enough of those drown the one input
    that is actually on. v3 replaces them with soft-max / soft-min, where an
    off input contributes exp(-large) ~ 0 and is harmless at any width.

    The trick that keeps it matmul-native is moving the weights OUTSIDE the
    exponential -- exp(tau*u) depends only on the input, so it is computed
    once and then it is just a matmul:

        u     = in_scale * x + in_shift          per-input learned squash
        w^    = |W| / sum|W|                     row-normalized, non-negative
        s_sum = W @ x + b
        s_or  =  log(w^ @ exp( tau*u)) / tau     soft max over inputs  (OR)
        s_and = -log(w^ @ exp(-tau*u)) / tau     soft min over inputs  (AND)

    Both are computed with the standard subtract-the-max stabilization, so
    exp() cannot overflow. tau is a single learnable scalar per LAYER (it
    sits inside the exponential, so a per-neuron tau would reintroduce the
    [B, n_out, n_in] expansion). Weights enter as non-negative row-normalized
    mixing coefficients, which is what makes each path a genuine weighted
    max/min -- so unlike v1/v2 there is no free NOT on these paths.

    Cost: two matmuls (one [B, n_in] for SUM, one [2B, n_in] for AND/OR) --
    the same ~3x-a-plain-linear budget as v2, with 1x memory.
    """

    EPS = 1e-12

    def __init__(self, n_in, n_out, dim=None, tau_init=5.0, sum_bias_init=0.0,
                 path_affine=True):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out, bias=True)

        self.in_scale = nn.Parameter(torch.ones(n_in))
        self.in_shift = nn.Parameter(torch.zeros(n_in))

        # one tau per layer: it lives inside exp(), so it must be shared
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))

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
        w = self.linear.weight
        u = self.in_scale * x + self.in_shift               # [B, n_in]
        tau = self.log_tau.exp()

        # non-negative, row-normalized mixing coefficients
        wa = w.abs()
        wn = wa / (wa.sum(-1, keepdim=True) + self.EPS)     # [n_out, n_in]

        a = tau * u
        m_hi = a.max(-1, keepdim=True).values               # [B, 1]
        m_lo = (-a).max(-1, keepdim=True).values
        # one matmul for both log-space paths
        e = torch.stack([(a - m_hi).exp(), (-a - m_lo).exp()], 0)
        h = F.linear(e, wn)                                 # [2, B, n_out]

        s_sum = F.linear(x, w, self.linear.bias)
        s_or = (m_hi + torch.log(h[0] + self.EPS)) / tau
        s_and = -(m_lo + torch.log(h[1] + self.EPS)) / tau

        if self.path_affine:
            s_and = self.and_scale * s_and + self.and_shift
            s_or = self.or_scale * s_or + self.or_shift

        gate = F.softmax(self.mix_logits, dim=-1)           # [n_out, 3]
        return gate[:, 0] * s_sum + gate[:, 1] * s_and + gate[:, 2] * s_or

    def gate_distribution(self):
        """Per-neuron softmax gate, detached: [n_out, 3] = (SUM, AND, OR)."""
        return F.softmax(self.mix_logits, dim=-1).detach()


class MGNv3Net(nn.Module):
    """Full model over MGNv3Linear layers; interface identical to MGNNet."""

    def __init__(self, in_features, hidden, num_classes, dim=None, **kw):
        super().__init__()
        widths = [in_features] + list(hidden)
        self.layers = nn.ModuleList(
            MGNv3Linear(a, b, dim, **kw)
            for a, b in zip(widths[:-1], widths[1:]))
        self.head = nn.Linear(widths[-1], num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.head(x)

    def gate_distributions(self):
        """List of per-layer gate tensors [n_out, 3], for histograms."""
        return [layer.gate_distribution() for layer in self.layers]


class MGNv4Linear(nn.Module):
    """Project-then-reduce multi-gate layer: [B, n_in] -> [B, n_out].

    v1-v3 reduce over the n_in inputs, which forces a choice between a
    [B, n_out, n_in] expansion (v1, exact but expensive) and keeping the
    weights outside the nonlinearity (v2/v3, matmul-native but weaker), and
    makes the AND/OR reductions degrade as fan-in grows.

    v4 sidesteps both: each neuron first projects the input down to k
    learned features with a plain matmul, then reduces over those k.

        s_sum = W_sum @ x + b                       [B, n_out]
        z     = W_proj @ x + c  -> [B, n_out, k]    k features per neuron
        s_and = exp(mean_k logsigmoid(z))           geometric mean over k
        s_or  = logsumexp_k(tau * z) / tau          smooth max over k

    Because the nonlinearity is applied AFTER the matmul, nothing expands:
    W_proj is one [n_out*k, n_in] matmul. And because the reduction runs
    over k (small) rather than n_in (large), AND/OR keep their
    discrimination at any layer width -- the fan-in decay that breaks v1's
    AND and v2's OR simply does not apply.

    The semantics shift from "all/any of my inputs" to "all/any of my k
    learned conditions", which is strictly more expressive per unit of
    compute. The OR path alone is essentially maxout; the gated AND/OR
    mixture is the new part. tau can be per-neuron here (it multiplies z
    after the matmul, so it costs nothing).

    k defaults to `dim` (the harness's shared width hyperparameter) so arms
    compare apples-to-apples, or 2 if dim is None.

    Cost: 1 + k matmul units vs a plain linear layer's 1, and (1+k)x the
    parameters -- so in a param-matched comparison the layer is narrower.
    """

    def __init__(self, n_in, n_out, dim=None, k=None, tau_init=5.0,
                 sum_bias_init=0.0, path_affine=True):
        super().__init__()
        self.k = k if k is not None else (dim if dim else 2)
        self.n_out = n_out

        self.linear = nn.Linear(n_in, n_out, bias=True)          # SUM path
        # shared projection bank read by BOTH the AND and OR reductions
        self.proj = nn.Linear(n_in, n_out * self.k, bias=True)

        self.log_tau = nn.Parameter(
            torch.full((n_out,), math.log(tau_init)))

        mix = torch.zeros(n_out, 3)
        mix[:, 0] = sum_bias_init
        self.mix_logits = nn.Parameter(mix)                      # [n_out, 3]

        self.path_affine = path_affine
        if path_affine:
            self.and_scale = nn.Parameter(torch.ones(n_out))
            self.and_shift = nn.Parameter(torch.zeros(n_out))
            self.or_scale = nn.Parameter(torch.ones(n_out))
            self.or_shift = nn.Parameter(torch.zeros(n_out))

    def forward(self, x):
        s_sum = self.linear(x)                                   # [B, n_out]
        z = self.proj(x).unflatten(-1, (self.n_out, self.k))     # [B, out, k]

        s_and = torch.exp(F.logsigmoid(z).mean(-1))              # [B, n_out]
        tau = self.log_tau.exp().unsqueeze(-1)                   # [n_out, 1]
        s_or = torch.logsumexp(tau * z, -1) / tau.squeeze(-1)

        if self.path_affine:
            s_and = self.and_scale * s_and + self.and_shift
            s_or = self.or_scale * s_or + self.or_shift

        gate = F.softmax(self.mix_logits, dim=-1)                # [n_out, 3]
        return gate[:, 0] * s_sum + gate[:, 1] * s_and + gate[:, 2] * s_or

    def gate_distribution(self):
        """Per-neuron softmax gate, detached: [n_out, 3] = (SUM, AND, OR)."""
        return F.softmax(self.mix_logits, dim=-1).detach()


class MGNv4Net(nn.Module):
    """Full model over MGNv4Linear layers; interface identical to MGNNet.

    `dim` is used here (unlike v1-v3): it sets k, the number of learned
    features each neuron reduces over.
    """

    def __init__(self, in_features, hidden, num_classes, dim=None, **kw):
        super().__init__()
        widths = [in_features] + list(hidden)
        self.layers = nn.ModuleList(
            MGNv4Linear(a, b, dim, **kw)
            for a, b in zip(widths[:-1], widths[1:]))
        self.head = nn.Linear(widths[-1], num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.head(x)

    def gate_distributions(self):
        """List of per-layer gate tensors [n_out, 3], for histograms."""
        return [layer.gate_distribution() for layer in self.layers]
