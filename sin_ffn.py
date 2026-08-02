"""FFN with a fraction of periodic (grid-cell-like) hidden neurons.

    z = W1 x + b
    y = concat[ sin(omega * z_sin + phi),  gelu(z_rest) ]
    out = W2 y

Every activation tested so far — relu, gelu, silu, leaky_relu, and the staged/
branched piecewise-linear variants — is monotone or nearly so, meaning a neuron
encodes "how much of this feature is present". A sine neuron is periodic and
non-monotone, so it encodes "this feature is NEAR a particular value, modulo a
period". That is a different representational scheme rather than a different
curve shape, and it is what a grid cell does: fire periodically as a function of
a linear projection of the input.

Placement is in the hidden layer, not before or after, for two reasons. After W2
the output enters the residual stream, which must stay unbounded — sin would
clamp it to [-1, 1]. Before W1 is the Fourier-features placement, which targets
low-dimensional coordinate inputs and would double W1's input width. In the
hidden layer, sin(omega * (w . x) + phi) is also the faithful grid-cell
analogue.

INIT IS THE WHOLE GAME. SIREN uses omega_0 = 30, but that assumes SIREN's own
weight init. Here the FFN input is LayerNorm'd (unit scale) and W1 inits at
std 0.02 with fan_in d_model, so the pre-activation lands at roughly
N(0, (0.02 * sqrt(d))^2) — about N(0, 0.39^2) at d=384. For a neuron to see a
meaningful arc of a period, omega * std(z) wants to be ~1-4, so omega ~ 2.5-10.
Porting omega_0 = 30 unchanged would wrap several periods per unit of input and
look like noise rather than a slightly worse model.

omega is therefore drawn LOG-UNIFORMLY over a range spanning that estimate, so
different neurons land at different scales without anyone having to guess the
single right frequency. Multi-scale modules are a grid-cell property too.

sin_frac=0 adds no parameters and is exactly Linear -> GELU -> Linear, so the
arm nests its own control.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinFFN(nn.Module):
    """Args:
        d_model:    Residual width (in and out).
        hidden:     Hidden width.
        sin_frac:   Fraction of hidden neurons that are periodic. 0 is a plain
                    GELU FFN with no extra parameters.
        omega_lo/hi: Log-uniform range for the initial frequencies. The default
                    1..16 brackets omega*std(z) ~ 0.4..6 at d_model=384.
        learn_freq: Train omega and phi. False pins the grid and trains only the
                    matmuls, which is the "tune the wiring, not the grid"
                    variant.
    """

    def __init__(self, d_model: int, hidden: int, sin_frac: float = 0.25,
                 omega_lo: float = 1.0, omega_hi: float = 16.0,
                 learn_freq: bool = True, bias: bool = True):
        super().__init__()
        if not 0.0 <= sin_frac <= 1.0:
            raise ValueError('sin_frac must be in [0, 1]')
        self.d_model = d_model
        self.hidden = hidden
        self.sin_frac = sin_frac
        self.n_sin = int(round(sin_frac * hidden))

        self.fc1 = nn.Linear(d_model, hidden, bias=bias)
        self.fc2 = nn.Linear(hidden, d_model, bias=bias)

        if self.n_sin:
            # log-uniform frequencies: neurons spread over scales
            u = torch.rand(self.n_sin)
            omega = torch.exp(u * (math.log(omega_hi) - math.log(omega_lo))
                              + math.log(omega_lo))
            phi = torch.rand(self.n_sin) * (2 * math.pi) - math.pi
            if learn_freq:
                self.omega = nn.Parameter(omega)
                self.phi = nn.Parameter(phi)
            else:
                self.register_buffer('omega', omega)
                self.register_buffer('phi', phi)
            self.register_buffer('omega_init', omega.clone())
        else:
            self.register_parameter('omega', None)
            self.register_parameter('phi', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.fc1(x)
        if self.n_sin == 0:
            return self.fc2(F.gelu(z))
        n = self.n_sin
        periodic = torch.sin(self.omega * z[..., :n] + self.phi)
        if n == self.hidden:
            return self.fc2(periodic)
        return self.fc2(torch.cat([periodic, F.gelu(z[..., n:])], dim=-1))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def drift(self) -> dict:
        """Did the frequencies move? If omega sits on its init the grid was
        never tuned, which is a different failure from the grid not helping."""
        if not self.n_sin:
            return {}
        w = self.omega.detach()
        return {
            'omega_mean': float(w.mean()),
            'omega_init_mean': float(self.omega_init.mean()),
            'omega_drift': float((w - self.omega_init).abs().mean()),
            'omega_min': float(w.min()),
            'omega_max': float(w.max()),
        }

    def extra_repr(self) -> str:
        return (f'd_model={self.d_model}, hidden={self.hidden}, '
                f'sin_frac={self.sin_frac} ({self.n_sin} of {self.hidden} '
                f'periodic)')
