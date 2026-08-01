import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """Gated FFN block (Shazeer, "GLU Variants Improve Transformer", 2020):

        h = silu(W_gate x) * (W_up x)
        y = W_down h

    Used here as a single param-matched stand-in for the whole head, the
    same convention as DendriticLinear in round 5 — not stacked per hidden
    layer. `hidden` is the gate/up projection width; W_gate and W_up both
    map in_features -> hidden, so params are dominated by
    2 * in_features * hidden against a plain Linear(in_features, hidden)'s
    single in_features * hidden.

    Args:
        in_features:  Input dimension.
        hidden:       Gate/up projection width.
        out_features: Output dimension.
        bias:         Bias on all three projections.
    """

    def __init__(self, in_features, hidden, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.hidden = hidden
        self.out_features = out_features

        self.gate = nn.Linear(in_features, hidden, bias=bias)
        self.up = nn.Linear(in_features, hidden, bias=bias)
        self.down = nn.Linear(hidden, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features) -> (..., out_features)"""
        return self.down(F.silu(self.gate(x)) * self.up(x))

    def extra_repr(self) -> str:
        p = sum(p.numel() for p in self.parameters())
        return (
            f"in_features={self.in_features}, hidden={self.hidden}, "
            f"out_features={self.out_features}, total_params={p}"
        )
