import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DendriticLinear(nn.Module):
    """A sparse two-stage layer:

        N inputs  ->  M = S*D dendrites  ->  leaky_relu  ->  S soma

    Each dendrite reads K inputs and applies a nonlinearity. Each soma takes a
    weighted sum of its OWN D dendrites — dendrites are private to their soma.

    The shape-matched dense equivalent is Linear(N, M) -> leaky_relu ->
    Linear(M, S): same units in every layer, same nonlinearity, but everything
    fully connected. This layer does M*K + M multiply-adds per input vector
    against the dense N*M + M*S.

    Input coverage
    --------------
    A soma reads D*K inputs, which in the sparse regime is far fewer than N
    (64 of 2048 at the defaults). If every soma read the SAME D*K inputs the
    layer would ignore the rest of its input entirely, so soma are split into
    G groups and each group reads a different window of the input:

        group g covers inputs [g*D*K, (g+1)*D*K)   (mod N)

    with G = ceil(N / (D*K)) so the groups tile the whole input. Collectively
    the soma see everything; each one individually still sees only its own D*K.

    G does not have to divide S: the soma count is padded internally and the
    surplus sliced off the output, so coverage is never traded away for a tidy
    reshape. Every group is guaranteed at least one real soma, the padding is
    smaller than one group, and `sparsity()["padded_soma"]` reports it. Full
    coverage is impossible only when S*D*K < N — genuinely too few soma to
    reach every input.

    When G * D * K == N the windows line up with a plain reshape, so no gather
    happens at all and the dendrite stage is a single batched matmul.

    Args:
        in_features:        Input dimension (N).
        out_features:       Number of soma (S) — the output dimension.
        fan_in:             Inputs per dendrite (K). int -> absolute count,
                            float -> fraction of in_features.
        dendrites_per_soma: D. Total dendrites M = out_features * D.
        coverage:           Legacy alternative to D: D = round(coverage*N/K),
                            i.e. how many times each soma tiles the input.
                            coverage=1.0 gives D*K == N, so every soma reads
                            the whole input and G collapses to 1. Ignored when
                            dendrites_per_soma is given.
        bias:               Bias terms on dendrites and soma.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fan_in: int | float = 16,
        dendrites_per_soma: int | None = None,
        coverage: float | None = None,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if isinstance(fan_in, int):
            self.K = max(1, min(in_features, fan_in))
        else:
            self.K = max(1, min(in_features, round(fan_in * in_features)))

        if dendrites_per_soma is not None:
            self.D = max(1, dendrites_per_soma)
        elif coverage is not None:
            self.D = max(1, round(coverage * in_features / self.K))
        else:
            self.D = 4

        S, D, K = out_features, self.D, self.K
        self.window = D * K          # inputs a single soma sees

        # Groups of soma, each reading a different window, chosen so the windows
        # tile the input. G need not divide S: the soma count is padded to a
        # multiple of the group size and the surplus is sliced off the output,
        # so the grouping stays a clean reshape without costing coverage.
        #
        # Sg is derived FIRST and G from it, not the other way around. Taking
        # G = ceil(N/window) and Sg = ceil(S/G) can push a whole group into the
        # padded tail (S=6, G=4 -> Sg=2 -> group 3 is soma 6..7, both padding),
        # silently dropping that group's window. Deriving G = ceil(S/Sg)
        # guarantees (G-1)*Sg < S, so every group keeps at least one soma.
        groups_needed = math.ceil(in_features / self.window)
        self.Sg = max(1, S // groups_needed)
        self.G = math.ceil(S / self.Sg)
        self.S_pad = self.G * self.Sg
        self.M = S * D               # dendrites that actually reach the output
        self.M_pad = self.S_pad * D

        # taps[g] = the window of input indices group g reads.
        taps = (torch.arange(self.G).unsqueeze(1) * self.window
                + torch.arange(self.window)) % in_features
        self.register_buffer("taps", taps)                      # (G, window)
        # When the windows tile the input exactly they are just a reshape.
        self.exact_tiling = self.G * self.window == in_features

        # Sized for the padded soma count; the tail is dropped in forward.
        self.synaptic = nn.Parameter(torch.empty(self.M_pad, K))  # dendrite weights
        self.cable = nn.Parameter(torch.empty(self.S_pad, D))     # dendrite -> soma

        if bias:
            self.dendrite_bias = nn.Parameter(torch.zeros(self.M_pad))
            self.soma_bias = nn.Parameter(torch.zeros(S))
        else:
            self.register_parameter("dendrite_bias", None)
            self.register_parameter("soma_bias", None)

        nn.init.kaiming_uniform_(self.synaptic, a=0.1, mode="fan_in")
        nn.init.kaiming_uniform_(self.cable, a=0.1, mode="fan_in")

    # ------------------------------------------------------------------
    def _windows(self, xf: torch.Tensor) -> torch.Tensor:
        """(B, N) -> (B, G, D, K): each group's window, without a copy when
        the windows tile the input exactly."""
        if self.exact_tiling:
            return xf.view(-1, self.G, self.D, self.K)
        return xf[:, self.taps].view(-1, self.G, self.D, self.K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features) -> (..., out_features)"""
        G, Sg, D, K = self.G, self.Sg, self.D, self.K
        lead = x.shape[:-1]

        xw = self._windows(x.reshape(-1, self.in_features))
        xw = xw.permute(1, 2, 0, 3)                             # (G, D, B, K)

        # Dendrites: one batched matmul over the (G, D) grid. Explicit matmul
        # rather than einsum so the backward is cuBLAS batched GEMM too.
        w = self.synaptic.view(G, Sg, D, K).permute(0, 2, 3, 1)  # (G, D, K, Sg)
        h = torch.matmul(xw, w)                                  # (G, D, B, Sg)

        if self.dendrite_bias is not None:
            h = h + self.dendrite_bias.view(G, Sg, D).permute(0, 2, 1).unsqueeze(2)
        h = F.leaky_relu(h, 0.1)

        # Soma: weighted sum over each soma's own D dendrites.
        cable = self.cable.view(G, Sg, D).permute(0, 2, 1).unsqueeze(2)
        s = (h * cable).sum(1)                                   # (G, B, Sg)
        s = s.permute(1, 0, 2).reshape(-1, self.S_pad)            # (B, S_pad)
        s = s[:, : self.out_features].reshape(*lead, self.out_features)

        if self.soma_bias is not None:
            s = s + self.soma_bias
        return s

    # ------------------------------------------------------------------
    def soma_indices(self) -> torch.Tensor:
        """(S, D, K) — which input each soma's dendrites read. Reference /
        introspection only; the forward never builds this."""
        per_group = self.taps.view(self.G, self.D, self.K)
        idx = per_group.repeat_interleave(self.Sg, dim=0)         # (S_pad, D, K)
        return idx[: self.out_features]

    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Straightforward gather implementation. Slow — it materializes a
        (..., S, D, K) tensor — and used only to check `forward`."""
        S, D, K = self.out_features, self.D, self.K
        idx = self.soma_indices()                                # (S, D, K)
        g = x[..., idx]                                          # (..., S, D, K)
        d = (g * self.synaptic[: S * D].view(S, D, K)).sum(-1)
        if self.dendrite_bias is not None:
            d = d + self.dendrite_bias[: S * D].view(S, D)
        d = F.leaky_relu(d, 0.1)

        s = (d * self.cable[:S]).sum(-1)
        if self.soma_bias is not None:
            s = s + self.soma_bias
        return s

    # ------------------------------------------------------------------
    def sparsity(self) -> dict:
        """Multiply-adds per input vector vs the shape-matched dense MLP
        (Linear(N, M) -> leaky_relu -> Linear(M, S))."""
        # MACs count the padded dendrites, since that is the work actually
        # executed; self.M is how many of them reach the output.
        ours = self.M_pad * self.K + self.M_pad
        dense = self.in_features * self.M + self.M * self.out_features
        return {
            "dendrites": self.M,
            "dendrites_computed": self.M_pad,
            "groups": self.G,
            "padded_soma": self.S_pad - self.out_features,
            "inputs_seen_per_soma": min(self.window, self.in_features),
            "inputs_covered": min(self.G * self.window, self.in_features),
            "macs": ours,
            "dense_macs": dense,
            "fraction_of_dense": ours / dense,
        }

    def extra_repr(self) -> str:
        p = sum(p.numel() for p in self.parameters())
        cov = self.sparsity()
        pad = f", padded_soma={cov['padded_soma']}" if cov["padded_soma"] else ""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"K={self.K}, D={self.D}, M={self.M}, groups={self.G}, "
            f"covers={cov['inputs_covered']}/{self.in_features}{pad}, "
            f"total_params={p}"
        )


class DendriticMLP(nn.Module):
    """DendriticLinear followed by a dense readout.

        N inputs -> M dendrites -> leaky_relu -> S soma -> leaky_relu
                 -> out_features   (fully connected)

    The readout provides mixing. Each soma sees only D*K of the N inputs and
    soma never see each other, so nothing inside DendriticLinear combines
    information across soma; one dense layer on the soma fixes that.

    Cost note: the readout is S * out_features weights, which at S=256 and
    out_features=256 is 65.8K — more than the 18.7K in the dendritic stage.
    It is the expensive part of this model, so keep S modest.

    Args:
        in_features:        Input dimension (N).
        soma:               Number of soma (S) — the dendritic layer's width.
        out_features:       Final output dimension. Defaults to `soma`.
        fan_in:             Inputs per dendrite (K).
        dendrites_per_soma: D. Total dendrites M = soma * D.
        bias:               Bias terms throughout.
    """

    def __init__(
        self,
        in_features: int,
        soma: int,
        out_features: int | None = None,
        fan_in: int | float = 16,
        dendrites_per_soma: int = 4,
        bias: bool = True,
    ):
        super().__init__()
        out_features = soma if out_features is None else out_features
        self.dendritic = DendriticLinear(
            in_features, soma, fan_in=fan_in,
            dendrites_per_soma=dendrites_per_soma, bias=bias,
        )
        self.readout = nn.Linear(soma, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(F.leaky_relu(self.dendritic(x), 0.1))

    def sparsity(self) -> dict:
        """MACs per input vector, split across the two stages. The dense
        comparison is the same three layers fully connected."""
        d = self.dendritic
        inner = d.sparsity()
        readout = d.out_features * self.out_features
        ours = inner["macs"] + readout
        dense = (d.in_features * d.M + d.M * d.out_features
                 + d.out_features * self.out_features)
        return {
            **{k: inner[k] for k in
               ("dendrites", "dendrites_computed", "groups", "padded_soma",
                "inputs_seen_per_soma", "inputs_covered")},
            "dendritic_macs": inner["macs"],
            "readout_macs": readout,
            "macs": ours,
            "dense_macs": dense,
            "fraction_of_dense": ours / dense,
        }
