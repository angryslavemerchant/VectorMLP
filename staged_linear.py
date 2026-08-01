import torch
import torch.nn as nn
import torch.nn.functional as F


class StagedLinear(nn.Module):
    """A dense layer whose neurons learn their own activation shape.

        z  = phi(W x + b)                      the ordinary dense layer
        z <- phi(a_i * z + c_i)                for i in 1..extra_stages

    a_i and c_i are one scalar per neuron, not matrices — neuron j in an extra
    stage reads only neuron j below it. All mixing between neurons happens in
    the single matmul; the extra stages only reshape each neuron's response
    curve.

    extra_stages=0 IS an ordinary Linear + leaky_relu, with no extra parameters
    at all, so it is an exact control rather than an approximate one.
    extra_stages=1 is the smallest interesting setting: one per-neuron weight,
    bias and nonlinearity stacked on the layer.

    Each extra stage adds one bend to the neuron's piecewise-linear activation
    (extra_stages=0 bends once, at zero; extra_stages=1 bends twice; and so
    on), with the layer learning where each bend sits and how steep the
    segments are. This is the Adaptive-Piecewise-Linear-unit idea folded into
    the layer.

    The extra weight and bias only buy anything BECAUSE a nonlinearity sits in
    front of them: without one, a*(Wx+b)+c is still affine and folds straight
    back into W and b for exactly the same function.

    Cost is dominated entirely by the matmul. Each extra stage adds 2 params
    per neuron and one elementwise pass, against a matmul of in_features *
    out_features — at 2048 -> 512 that is 1024 params on top of 1.05M, ~0.1%.
    Run it under torch.compile: the stages fuse into the matmul epilogue, and
    eager mode pays a real per-stage cost instead.

    Args:
        in_features:     Input dimension.
        out_features:    Output dimension.
        extra_stages:    Per-neuron scale/shift/nonlinearity stages stacked on
                         the dense layer. 0 is a plain layer (the control).
        negative_slope:  Target negative slope of the COMPOSITE activation.
        correct_slope:   Give each nonlinearity a slope of
                         negative_slope**(1/(extra_stages+1)) so the
                         composition lands on negative_slope regardless of
                         depth. Without it, stacking leaky_relu shrinks the
                         negative slope geometrically (0.1 -> 0.01 -> 0.001),
                         so different depths would start from different
                         functions and confound any comparison.
        random_init:     Draw stage scales from U(0.5, 1.5) and shifts from
                         U(-0.1, 0.1) so neurons start with different curves.
                         False starts every stage at scale 1, shift 0.
        bias:            Bias on the matmul.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        extra_stages: int = 1,
        negative_slope: float = 0.1,
        correct_slope: bool = True,
        random_init: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        if extra_stages < 0:
            raise ValueError("extra_stages must be >= 0")
        self.in_features = in_features
        self.out_features = out_features
        self.extra_stages = extra_stages
        self.negative_slope = negative_slope

        # one nonlinearity for the dense layer itself, plus one per extra stage
        self.nonlinearities = extra_stages + 1
        self.slope = (negative_slope ** (1.0 / self.nonlinearities)
                      if correct_slope else negative_slope)

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if extra_stages:
            self.scale = nn.Parameter(torch.empty(extra_stages, out_features))
            self.shift = nn.Parameter(torch.empty(extra_stages, out_features))
            if random_init:
                nn.init.uniform_(self.scale, 0.5, 1.5)
                nn.init.uniform_(self.shift, -0.1, 0.1)
            else:
                nn.init.ones_(self.scale)
                nn.init.zeros_(self.shift)
        else:
            self.register_parameter("scale", None)
            self.register_parameter("shift", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features) -> (..., out_features)"""
        z = F.leaky_relu(self.linear(x), self.slope)
        for i in range(self.extra_stages):
            z = F.leaky_relu(self.scale[i] * z + self.shift[i], self.slope)
        return z

    # ------------------------------------------------------------------
    def bends(self) -> int:
        """Bends in each neuron's piecewise-linear activation."""
        return self.nonlinearities

    def cost(self) -> dict:
        """Multiply-adds per input vector, split between matmul and stages."""
        matmul = self.in_features * self.out_features
        stages = self.extra_stages * self.out_features
        params = sum(p.numel() for p in self.parameters())
        return {
            "params": params,
            "stage_params": 2 * self.extra_stages * self.out_features,
            "matmul_macs": matmul,
            "stage_macs": stages,
            "macs": matmul + stages,
            "stage_fraction": stages / (matmul + stages),
            "bends": self.bends(),
        }

    def extra_repr(self) -> str:
        c = self.cost()
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"extra_stages={self.extra_stages}, bends={self.bends()}, "
            f"slope_each={self.slope:.4f}, "
            f"stage_params={c['stage_params']} of {c['params']}"
        )


class StagedMLP(nn.Module):
    """Same shape/role as vector_mlp.PlainMLP, but hidden layers are
    StagedLinear instead of Linear+ReLU pairs — StagedLinear already bakes
    its own activation in, so no external activation is added after it.
    The final classifier stays a bare nn.Linear: StagedLinear always applies
    leaky_relu internally (even at extra_stages=0), which would clip the
    logits if used as the last layer."""

    def __init__(self, in_features, hidden, num_classes, extra_stages=1, **stage_kw):
        super().__init__()
        widths = [in_features] + list(hidden)
        self.body = nn.Sequential(*[
            StagedLinear(a, b, extra_stages=extra_stages, **stage_kw)
            for a, b in zip(widths[:-1], widths[1:])])
        self.head = nn.Linear(widths[-1], num_classes)

    def forward(self, x):
        return self.head(self.body(x))
