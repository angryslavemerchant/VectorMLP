"""FFN whose hidden units are quadratic forms rather than linear projections.

A quadratic neuron computes x^T A x + w^T x + b instead of w^T x + b. Storing A
outright is d(d+1)/2 parameters per neuron -- 73,920 at d=384 -- so this uses
the eigen form, which is the same function written as a sum of squared
projections:

    y_j = sum_r lambda_jr * (v_jr . x + c_jr)^2

r = d spans every symmetric A, and the squared affine term expands to a
quadratic plus a linear plus a constant, so nothing is lost against the full
inhomogeneous form. Smaller r gives a low-rank quadratic.

One caveat on parameter efficiency: with V unconstrained this uses h*r*d
parameters, and a symmetric matrix only has d(d+1)/2 degrees of freedom, so at
full rank it is 2x over-parameterised. That matters only at the extreme end --
at matched budget r=d buys ~7 neurons rather than the ~16 a triangular
parameterisation would. Low-to-mid r, which is the interesting region, is
unaffected.

Computationally this is one GEMM (d -> h*r), an elementwise square, and a
grouped sum, so it stays compute-bound with high arithmetic intensity. That is
the opposite of the dendritic and conv attempts, which both saved parameters or
FLOPs and then lost it all to activation memory traffic.

rank is the width-versus-order knob. At the FFN budget with d=384:

    r=1    h=1534    squared activation at full width (cf. Primer's ReLU^2)
    r=2    h=1021
    r=4    h=612
    r=8    h=340
    r=16   h=180
    r=32   h=92
    r=64   h=47      genuinely quadratic neurons, few of them

r=1 is the cheap end and has prior support; high r is the bet that a handful of
second-order neurons beat many first-order ones. SwiGLU already won this trade
once at effectively rank 1, which is the reason to look at the rest of it.
"""

import math

import torch
import torch.nn as nn


class QuadraticFFN(nn.Module):
    """Args:
        d_model: Residual width (in and out).
        hidden:  Number of quadratic neurons.
        rank:    Squared projections summed per neuron. rank=d_model spans the
                 full quadratic form.
        bias:    Bias inside the squared term, which is what supplies the
                 linear and constant parts of the quadratic.
    """

    def __init__(self, d_model: int, hidden: int, rank: int = 4,
                 bias: bool = True):
        super().__init__()
        if rank < 1:
            raise ValueError('rank must be >= 1')
        self.d_model = d_model
        self.hidden = hidden
        self.rank = rank

        self.proj = nn.Linear(d_model, hidden * rank, bias=bias)
        # summing r squares would grow the scale with r, so shrink lambda to
        # keep the neuron's output roughly rank-independent
        self.lam = nn.Parameter(torch.randn(hidden, rank) / math.sqrt(rank))
        self.out = nn.Linear(hidden, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model) -> (..., d_model)"""
        z = self.proj(x)                                  # (..., h*r)
        z = z * z                                         # the quadratic part
        z = z.unflatten(-1, (self.hidden, self.rank))     # (..., h, r)
        return self.out((z * self.lam).sum(-1))           # (..., h) -> (..., d)

    def macs_per_token(self) -> int:
        return self.d_model * self.hidden * self.rank + self.hidden * self.d_model

    def extra_repr(self) -> str:
        p = sum(q.numel() for q in self.parameters())
        return (f'd_model={self.d_model}, hidden={self.hidden}, rank={self.rank}, '
                f'params={p:,}')
