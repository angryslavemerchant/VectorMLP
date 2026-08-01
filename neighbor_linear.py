import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighborLinear(nn.Module):
    """A dense layer with a cheap sparse second stage that MIXES neurons.

        h = phi(W x + b)
        y = phi( sum_t a_t * h[j + offset_t]  +  c )

    where the offsets are a small centred window wrapped around the layer, so
    neuron j reads `neighbors` values instead of just its own. a_t and c are
    one scalar per neuron.

    This is a different axis from StagedLinear and BranchedLinear. Those make a
    neuron's own response curve more elaborate but bring in no new information;
    this brings in other neurons' values. neighbors=1 reduces exactly to one
    sequential stage (the diagonal case), so the neighbour version strictly
    contains it.

    A caveat worth stating: after a dense first layer the neuron ordering is
    arbitrary, so "neighbours" are just some other neurons — there is nothing
    spatially meaningful about j-1 and j+1. What the window buys is a fixed,
    contiguous access pattern that compiles into shifted multiply-adds rather
    than a gather.

    neighbors=0 is a plain Linear + leaky_relu with no extra parameters, and
    neighbors=1 is the diagonal (equivalent to StagedLinear extra_stages=1).

    Cost is neighbors * out_features weights plus out_features biases, against
    a matmul of in_features * out_features. At 2048 -> 512 with 3 neighbours
    that is 2048 parameters on top of 1.05M.

    Args:
        in_features:    Input dimension.
        out_features:   Output dimension.
        neighbors:      How many values each neuron reads in the second stage.
                        0 = plain layer, 1 = diagonal, 3 = itself plus one on
                        each side. Offsets are centred and ring-wrapped.
        negative_slope: Target negative slope of the COMPOSITE activation; each
                        of the two nonlinearities gets sqrt(negative_slope) so
                        the composition lands on it (see StagedLinear).
        correct_slope:  Apply that correction.
        random_init:    Draw mixing weights from U(-0.5, 0.5) with the centre
                        tap at 1.0, so it starts near the identity mix.
        bias:           Bias on the matmul.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        neighbors: int = 3,
        negative_slope: float = 0.1,
        correct_slope: bool = True,
        random_init: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        if neighbors < 0:
            raise ValueError("neighbors must be >= 0")
        if neighbors > out_features:
            raise ValueError("neighbors cannot exceed out_features")
        self.in_features = in_features
        self.out_features = out_features
        self.neighbors = neighbors
        self.negative_slope = negative_slope

        self.nonlinearities = 2 if neighbors else 1
        self.slope = (negative_slope ** (1.0 / self.nonlinearities)
                      if correct_slope else negative_slope)

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if neighbors:
            # centred window: neighbors=3 -> (-1, 0, 1); neighbors=2 -> (0, 1)
            # Plain Python ints, NOT a buffer: reading an int out of a tensor
            # inside forward graph-breaks dynamo on every iteration, which
            # silently disables compilation for the whole module.
            self.offsets = tuple(range(-(neighbors // 2),
                                       neighbors - (neighbors // 2)))
            self.centre = self.offsets.index(0)

            self.mix = nn.Parameter(torch.empty(neighbors, out_features))
            self.shift = nn.Parameter(torch.zeros(out_features))
            if random_init:
                nn.init.uniform_(self.mix, -0.5, 0.5)
                with torch.no_grad():
                    self.mix[self.centre].fill_(1.0)
            else:
                nn.init.zeros_(self.mix)
                with torch.no_grad():
                    self.mix[self.centre].fill_(1.0)
        else:
            self.register_parameter("mix", None)
            self.register_parameter("shift", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features) -> (..., out_features)"""
        h = F.leaky_relu(self.linear(x), self.slope)
        if not self.neighbors:
            return h

        # shifted multiply-adds: roll(h, -off) puts neuron j+off at position j,
        # so no gather is needed and the whole loop fuses under compile
        acc = None
        for t, off in enumerate(self.offsets):
            hs = h if off == 0 else h.roll(-off, dims=-1)
            term = self.mix[t] * hs
            acc = term if acc is None else acc + term
        return F.leaky_relu(acc + self.shift, self.slope)

    # ------------------------------------------------------------------
    def cost(self) -> dict:
        matmul = self.in_features * self.out_features
        mix = self.neighbors * self.out_features
        params = sum(p.numel() for p in self.parameters())
        return {
            "params": params,
            "mix_params": mix + (self.out_features if self.neighbors else 0),
            "matmul_macs": matmul,
            "mix_macs": mix,
            "macs": matmul + mix,
            "mix_fraction": mix / (matmul + mix),
        }

    def extra_repr(self) -> str:
        c = self.cost()
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"neighbors={self.neighbors}, slope_each={self.slope:.4f}, "
            f"mix_params={c['mix_params']} of {c['params']}"
        )


class NeighborMLP(nn.Module):
    """Same shape/role as vector_mlp.PlainMLP, but hidden layers are
    NeighborLinear instead of Linear+ReLU pairs — NeighborLinear already
    bakes its own activation(s) in, so no external activation is added
    after it. The final classifier stays a bare nn.Linear.

    NeighborLinear requires out_features >= neighbors, so any width search
    over this class needs a floor at `neighbors` (see matched_width's
    min_w)."""

    def __init__(self, in_features, hidden, num_classes, neighbors=3, **kw):
        super().__init__()
        widths = [in_features] + list(hidden)
        self.body = nn.Sequential(*[
            NeighborLinear(a, b, neighbors=neighbors, **kw)
            for a, b in zip(widths[:-1], widths[1:])])
        self.head = nn.Linear(widths[-1], num_classes)

    def forward(self, x):
        return self.head(self.body(x))
