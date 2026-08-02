"""Convolutional FFN: reshape the residual vector into a grid and U-Net it.

    384 -> 4x4x24 -> up -> up -> process at peak -> down -> down -> 4x4x24 -> 384

The residual stream has no spatial structure, so the grid is arbitrary — this
is not exploiting locality in the way a vision model does. What it exploits is
WEIGHT SHARING: the same 3x3 kernel is applied at every position, so the block
does far more arithmetic per parameter than a matmul does.

That is the whole point, and the whole risk. At the default channel counts the
block is parameter-matched to a standard FFN almost exactly (1,178,104 against
1,181,568) while doing ~170x the arithmetic — 201M MACs per token versus 1.18M.
Every other arm in this study was both param- and FLOP-matched; this one cannot
be, because weight sharing decouples the two by construction.

Whether that trade is good is an open question worth asking: a transformer FFN
is parameter-bound and runs well below the hardware's arithmetic ceiling, so
spending compute that would otherwise sit idle is not obviously wasteful. But
it is 170x, not free, and the wall clock reflects it.

Defaults follow the 384-dim spec:

    stage        grid    channels   dims
    input        4x4     24         384
    after up1    8x8     96         6,144
    after up2    16x16   272        69,632   <- peak
    after down1  8x8     96         6,144
    output       4x4     24         384
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFFN(nn.Module):
    """Drop-in FFN replacement operating on a reshaped grid.

    Args:
        d_model:  Residual width. Must be divisible by side*side.
        c1:       Channels at the first up stage (grid 2*side).
        c2:       Channels at the peak (grid 4*side). Defaults to round(17*c1/6),
                  the ratio in the 96/272 spec.
        side:     Base grid side. 4 gives 4x4 with d_model//16 channels.
        peak:     'wide' runs the process conv at the 4*side grid as specified;
                  'mid' runs it at 2*side, which cuts the arithmetic ~4x and is
                  worth trying first if the full version is too slow.
    """

    def __init__(self, d_model: int, c1: int = 96, c2: int | None = None,
                 side: int = 4, peak: str = 'wide'):
        super().__init__()
        if d_model % (side * side):
            raise ValueError(f'd_model {d_model} not divisible by {side*side}')
        if peak not in ('wide', 'mid'):
            raise ValueError("peak must be 'wide' or 'mid'")

        self.d_model = d_model
        self.side = side
        self.base_ch = d_model // (side * side)
        self.c1 = c1
        self.c2 = c2 if c2 is not None else max(1, round(17 * c1 / 6))
        self.peak = peak

        k = dict(kernel_size=3, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(self.base_ch, self.c1, output_padding=1, **k)
        self.up2 = nn.ConvTranspose2d(self.c1, self.c2, output_padding=1, **k)
        # the expensive one: same grid in and out, so it runs at every position
        self.mid = nn.Conv2d(self.c2, self.c2, kernel_size=3, padding=1)
        self.down1 = nn.Conv2d(self.c2, self.c1, **k)
        self.down2 = nn.Conv2d(self.c1, self.base_ch, **k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model) -> (..., d_model)"""
        lead = x.shape[:-1]
        g = x.reshape(-1, self.base_ch, self.side, self.side)

        g = F.gelu(self.up1(g))                       # side -> 2*side
        if self.peak == 'wide':
            g = F.gelu(self.up2(g))                   # -> 4*side
            g = F.gelu(self.mid(g))                   # process at peak
            g = F.gelu(self.down1(g))                 # -> 2*side
        else:
            # process at the middle grid instead; up2/down1 still change the
            # channel count so the parameter count is unchanged
            g = F.gelu(F.conv2d(g, self.up2.weight.transpose(0, 1),
                                self.up2.bias, padding=1))
            g = F.gelu(self.mid(g))
            g = F.gelu(F.conv2d(g, self.down1.weight, self.down1.bias,
                                stride=1, padding=1))
        g = self.down2(g)                             # -> side
        return g.reshape(*lead, self.d_model)

    # ------------------------------------------------------------------
    def macs_per_token(self) -> int:
        """Multiply-adds for one token, against a standard FFN's 2*d*hidden."""
        s, b = self.side, self.base_ch
        if self.peak == 'wide':
            return (s * s * 9 * b * self.c1                       # up1
                    + (2 * s) ** 2 * 9 * self.c1 * self.c2        # up2
                    + (4 * s) ** 2 * 9 * self.c2 * self.c2        # mid
                    + (2 * s) ** 2 * 9 * self.c2 * self.c1        # down1
                    + s * s * 9 * self.c1 * b)                    # down2
        return (s * s * 9 * b * self.c1
                + (2 * s) ** 2 * 9 * self.c1 * self.c2
                + (2 * s) ** 2 * 9 * self.c2 * self.c2
                + (2 * s) ** 2 * 9 * self.c2 * self.c1
                + s * s * 9 * self.c1 * b)

    def extra_repr(self) -> str:
        p = sum(q.numel() for q in self.parameters())
        return (f'd_model={self.d_model}, grid={self.side}x{self.side}x'
                f'{self.base_ch}, c1={self.c1}, c2={self.c2}, peak={self.peak}, '
                f'params={p:,}, macs/token={self.macs_per_token():,}')
