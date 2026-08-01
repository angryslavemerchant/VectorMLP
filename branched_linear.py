import torch
import torch.nn as nn
import torch.nn.functional as F


class BranchedLinear(nn.Module):
    """A dense layer whose neurons learn their activation shape in PARALLEL.

        z = W x + b
        y = phi(z) + sum_i w_i * phi(-z + c_i)      for i in 1..extra_branches

    Every branch reads the same pre-activation z and the results are summed —
    no branch feeds another. c_i places branch i's breakpoint, w_i sets how
    sharply the curve bends there, both one scalar per neuron.

    Versus StagedLinear (which composes stages sequentially): parallel branches
    ADD their slopes, so k branches span every piecewise-linear function with k
    breakpoints. Sequential composition MULTIPLIES slopes and only reaches a
    constrained subset of those functions. Parallel is also friendlier to the
    GPU — the branches are independent, so there is no serial dependency chain
    and they fuse into one pass.

    This is the Adaptive Piecewise Linear unit. Note there are only TWO
    parameters per branch, not three: an inner scale a_i on phi(a_i*z + c_i)
    would be redundant, because leaky_relu is positively homogeneous and
    scaling (a_i, c_i) is indistinguishable from scaling w_i.

    extra_branches=0 IS a plain Linear + leaky_relu with no extra parameters,
    so it is an exact control. Unlike the sequential version no slope
    correction is needed: branches do not compose, so the negative slope is
    not decayed by depth.

    Args:
        in_features:    Input dimension.
        out_features:   Output dimension.
        extra_branches: Parallel branches beyond the base phi(z). 0 is a plain
                        layer; each branch adds one bend to the activation.
        negative_slope: leaky_relu negative slope.
        random_init:    Spread breakpoints over U(-1, 1) and draw branch
                        weights from U(-0.1, 0.1), so neurons start with
                        different curves but close to a plain activation.
                        False starts every branch at weight 0 (exactly plain).
        bias:           Bias on the matmul.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        extra_branches: int = 1,
        negative_slope: float = 0.1,
        random_init: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        if extra_branches < 0:
            raise ValueError("extra_branches must be >= 0")
        self.in_features = in_features
        self.out_features = out_features
        self.extra_branches = extra_branches
        self.negative_slope = negative_slope

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if extra_branches:
            self.breakpoint = nn.Parameter(torch.empty(extra_branches, out_features))
            self.weight = nn.Parameter(torch.empty(extra_branches, out_features))
            if random_init:
                nn.init.uniform_(self.breakpoint, -1.0, 1.0)
                nn.init.uniform_(self.weight, -0.1, 0.1)
            else:
                nn.init.zeros_(self.breakpoint)
                nn.init.zeros_(self.weight)
        else:
            self.register_parameter("breakpoint", None)
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features) -> (..., out_features)"""
        z = self.linear(x)
        y = F.leaky_relu(z, self.negative_slope)
        for i in range(self.extra_branches):
            y = y + self.weight[i] * F.leaky_relu(
                self.breakpoint[i] - z, self.negative_slope)
        return y

    # ------------------------------------------------------------------
    def bends(self) -> int:
        return self.extra_branches + 1

    def cost(self) -> dict:
        matmul = self.in_features * self.out_features
        branches = 2 * self.extra_branches * self.out_features
        params = sum(p.numel() for p in self.parameters())
        return {
            "params": params,
            "branch_params": 2 * self.extra_branches * self.out_features,
            "matmul_macs": matmul,
            "branch_macs": branches,
            "macs": matmul + branches,
            "branch_fraction": branches / (matmul + branches),
            "bends": self.bends(),
        }

    def extra_repr(self) -> str:
        c = self.cost()
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"extra_branches={self.extra_branches}, bends={self.bends()}, "
            f"branch_params={c['branch_params']} of {c['params']}"
        )


class BranchedMLP(nn.Module):
    """Same shape/role as vector_mlp.PlainMLP, but hidden layers are
    BranchedLinear instead of Linear+ReLU pairs — BranchedLinear already
    bakes its own activation in, so no external activation is added after
    it. The final classifier stays a bare nn.Linear: BranchedLinear always
    leaky_relus internally (even at extra_branches=0), which would clip the
    logits if used as the last layer."""

    def __init__(self, in_features, hidden, num_classes, extra_branches=1, **kw):
        super().__init__()
        widths = [in_features] + list(hidden)
        self.body = nn.Sequential(*[
            BranchedLinear(a, b, extra_branches=extra_branches, **kw)
            for a, b in zip(widths[:-1], widths[1:])])
        self.head = nn.Linear(widths[-1], num_classes)

    def forward(self, x):
        return self.head(self.body(x))
