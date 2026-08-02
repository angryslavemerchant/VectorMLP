"""Per-neuron learnable activation, decoupled from any matmul.

BranchedLinear bundles a Linear with its branched activation, which makes it
impossible to drop the mechanism somewhere else — into SwiGLU's gate, say, or
onto the product SwiGLU feeds to its down projection. This module is just the
activation, so it goes anywhere.

The TinyStories FFN run is what motivated pulling it out. Branched was the only
per-neuron variant that helped (-0.029 nats against its leaky_relu control,
while staged and neighbor were both slightly worse than doing nothing), but the
plain GELU block still beat it by 0.018, and SwiGLU beat everything. Two
questions follow, and both need the activation separated from the layer:

  * does branched still help on a SMOOTH base, or was it only compensating for
    leaky_relu being a poor activation?   -> base='gelu'
  * does it add anything on top of GATING, which is what actually won?
    -> drop it into SwiGLU's gate or onto its product
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def activation(name, negative_slope=0.1):
    """name -> callable. 'identity' is for cases where the baseline applies no
    activation at all, so branch weights of zero reproduce it exactly."""
    if name == 'identity':
        return lambda z: z
    if name == 'gelu':
        return F.gelu
    if name == 'silu':
        return F.silu
    if name == 'relu':
        return F.relu
    if name == 'lrelu':
        return lambda z: F.leaky_relu(z, negative_slope)
    raise ValueError(f'unknown activation {name!r}')


class BranchedActivation(nn.Module):
    """Learnable per-neuron activation: a fixed base plus parallel branches.

        y = base(z) + sum_i w_i * phi(-z + c_i)

    c_i places branch i's bend, w_i sets how sharply the curve turns there,
    both one scalar per feature. Branches are summed, never composed — their
    slopes ADD, which is what lets k branches span every piecewise-linear
    function with k bends. (Composing stages multiplies slopes instead and
    reaches only a subset; that is the staged variant, which did not help.)

    w is initialised near zero, so the module starts at `base` and moves away
    only if the loss rewards it. At w = 0 it IS base exactly — the arm nests
    its own control, so a null result means the mechanism did not help rather
    than that the block was changed for the worse.

    base='identity' is the setting for slots where the baseline applies no
    activation, e.g. the product inside SwiGLU that feeds its down projection.
    Branches still use a real nonlinearity there, otherwise the whole thing
    collapses to an affine map.

    Args:
        features:       Number of neurons (last dim of the input).
        extra_branches: Branches beyond the base. 0 adds no parameters at all.
        base:           'lrelu' | 'gelu' | 'silu' | 'relu' | 'identity'.
        branch_act:     Nonlinearity used inside the branches. Defaults to the
                        base, or 'lrelu' when the base is identity.
        bp_scale:       Breakpoints init as U(-bp_scale, bp_scale). Raise it
                        where the input has a wide range — SwiGLU's product is
                        heavier-tailed than an ordinary pre-activation, and
                        bends initialised too close to zero can sit in a region
                        the data never reaches.
        negative_slope: For leaky_relu.
    """

    def __init__(
        self,
        features: int,
        extra_branches: int = 1,
        base: str = 'lrelu',
        branch_act: str | None = None,
        bp_scale: float = 1.0,
        negative_slope: float = 0.1,
        random_init: bool = True,
    ):
        super().__init__()
        if extra_branches < 0:
            raise ValueError('extra_branches must be >= 0')
        self.features = features
        self.extra_branches = extra_branches
        self.base_name = base
        self.branch_name = branch_act or ('lrelu' if base == 'identity' else base)
        self.bp_scale = bp_scale

        self.base = activation(base, negative_slope)
        self.branch = activation(self.branch_name, negative_slope)

        if extra_branches:
            self.breakpoint = nn.Parameter(torch.empty(extra_branches, features))
            self.weight = nn.Parameter(torch.empty(extra_branches, features))
            if random_init:
                nn.init.uniform_(self.breakpoint, -bp_scale, bp_scale)
                nn.init.uniform_(self.weight, -0.1, 0.1)
            else:
                nn.init.zeros_(self.breakpoint)
                nn.init.zeros_(self.weight)
            self.register_buffer('weight_init', self.weight.detach().clone())
        else:
            self.register_parameter('breakpoint', None)
            self.register_parameter('weight', None)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = self.base(z)
        for i in range(self.extra_branches):
            y = y + self.weight[i] * self.branch(self.breakpoint[i] - z)
        return y

    # ------------------------------------------------------------------
    @torch.no_grad()
    def drift(self) -> dict:
        """How far the branch weights moved from their init.

        A branch that never leaves init is dead — either the mechanism is
        useless here or its bends were initialised somewhere the data never
        goes. Without this a null result is ambiguous between the two.
        """
        if not self.extra_branches:
            return {}
        w = self.weight.detach()
        return {
            'w_abs_mean': float(w.abs().mean()),
            'w_abs_init': float(self.weight_init.abs().mean()),
            'w_drift': float((w - self.weight_init).abs().mean()),
            'bp_abs_mean': float(self.breakpoint.detach().abs().mean()),
        }

    def extra_repr(self) -> str:
        return (f'features={self.features}, extra_branches={self.extra_branches}, '
                f'base={self.base_name}, branch={self.branch_name}, '
                f'bp_scale={self.bp_scale}')


class SwiGLUBranched(nn.Module):
    """SwiGLU with a learnable per-neuron activation spliced in.

        gate/up projections -> h = act_gate(W_gate x) * (W_up x)
                            -> y = W_down(act_post(h))

    where='gate'   replace SwiGLU's fixed silu on the gate with a branched
                   activation on a silu base. At weight 0 this is stock SwiGLU.
    where='post'   leave the gate alone and add branches to the PRODUCT, on an
                   identity base — SwiGLU applies nothing there, so identity is
                   what makes weight 0 reproduce it exactly. A silu base would
                   change the block even before any branch is used, and would
                   measure two things at once.

    Either way the cost is 2 parameters per hidden unit against three full
    projections, so the hidden width barely moves and nothing is confounded by
    size.
    """

    def __init__(
        self,
        in_features: int,
        hidden: int,
        out_features: int,
        extra_branches: int = 1,
        where: str = 'gate',
        bp_scale: float | None = None,
        bias: bool = True,
    ):
        super().__init__()
        if where not in ('gate', 'post'):
            raise ValueError("where must be 'gate' or 'post'")
        self.in_features = in_features
        self.hidden = hidden
        self.out_features = out_features
        self.where = where

        self.gate = nn.Linear(in_features, hidden, bias=bias)
        self.up = nn.Linear(in_features, hidden, bias=bias)
        self.down = nn.Linear(hidden, out_features, bias=bias)

        if bp_scale is None:
            # the product is heavier-tailed than a pre-activation, so give the
            # 'post' bends a wider spread to start in
            bp_scale = 1.0 if where == 'gate' else 3.0
        self.act = BranchedActivation(
            hidden, extra_branches,
            base='silu' if where == 'gate' else 'identity',
            bp_scale=bp_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.gate(x), self.up(x)
        if self.where == 'gate':
            return self.down(self.act(g) * u)
        return self.down(self.act(F.silu(g) * u))

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, hidden={self.hidden}, '
                f'out_features={self.out_features}, where={self.where}')
