"""Synthetic angle kill test (spec: channel_mixing_neuron_spec.md, experiment 1).

16x16 images of one of three oriented shapes (bar+blob, cross+blob, T) drawn at
angle phi with pixel noise. Train on phi in [0, 120) degrees, test on the
held-out arc [120, 360). Primary metric: shape classification accuracy on
held-out angles. Probe: a fixed population decoder reading phi off the last
hidden layer's channel pattern.

Arms:
  mlp            param-matched plain MLP (matched to the ring net)
  matrix         VectorMLP, per-neuron DxD matrix, learned on-ramp
  ring-learned   VectorMLP ring, learned on-ramp          (arm A)
  ring-circular  VectorMLP ring, hand-wired circular on-ramp, frozen (arm B)

Run inside the Toastenv conda env (CUDA torch).
"""

import math

import torch
import torch.nn.functional as F

from vector_mlp import VectorMLP, PlainMLP, count_params, matched_mlp_width

SIZE = 16
DIM = 16
HIDDEN = [64, 64]
TRAIN_DEG = 120.0          # train arc [0, TRAIN_DEG)
N_TRAIN, N_TEST = 20000, 4000
STEPS, BATCH, LR = 3000, 256, 1e-3
NOISE = 0.1
SEED = 0

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# each shape: list of segments ((x0,y0),(x1,y1),width) + optional blob (x,y,r).
# Canonical frame: "forward" is +x. Units are pixels; grid spans [-7.5, 7.5].
SHAPES = [
    {'segs': [((-5, 0), (5, 0), 1.1)],                       'blob': (5, 0, 1.6)},
    {'segs': [((-4, 0), (4, 0), 1.1),
              ((0, -4), (0, 4), 1.1)],                       'blob': (4, 0, 1.6)},
    {'segs': [((-4, 0), (4, 0), 1.1),
              ((0, 0), (0, 5), 1.1)],                        'blob': None},
]


def _seg_dist(px, py, a, b):
    """Distance from points [B,P] to segment a-b (canonical frame)."""
    ax, ay = a
    bx, by = b
    abx, aby = bx - ax, by - ay
    t = ((px - ax) * abx + (py - ay) * aby) / (abx * abx + aby * aby)
    t = t.clamp(0, 1)
    return ((px - (ax + t * abx)) ** 2 + (py - (ay + t * aby)) ** 2).sqrt()


def render(shape_ids, phis):
    """shape_ids [B] long, phis [B] radians -> images [B, SIZE*SIZE]."""
    B = shape_ids.shape[0]
    c = torch.arange(SIZE, dtype=torch.float32) - (SIZE - 1) / 2
    gy, gx = torch.meshgrid(c, c, indexing='ij')
    gx, gy = gx.reshape(-1), gy.reshape(-1)                  # [P]
    cos, sin = torch.cos(phis)[:, None], torch.sin(phis)[:, None]
    # rotate grid by -phi = express pixels in the shape's canonical frame
    px = cos * gx + sin * gy                                 # [B, P]
    py = -sin * gx + cos * gy
    img = torch.zeros(B, SIZE * SIZE)
    for s, spec in enumerate(SHAPES):
        m = shape_ids == s
        if not m.any():
            continue
        acc = torch.zeros(int(m.sum()), SIZE * SIZE)
        for a, b, w in spec['segs']:
            acc = torch.maximum(acc, torch.exp(-(_seg_dist(px[m], py[m], a, b) / w) ** 2))
        if spec['blob'] is not None:
            bx, by, r = spec['blob']
            d = ((px[m] - bx) ** 2 + (py[m] - by) ** 2).sqrt()
            acc = torch.maximum(acc, torch.exp(-(d / r) ** 2))
        img[m] = acc
    return img + NOISE * torch.randn_like(img)


def make_split(n, lo_deg, hi_deg, gen):
    shapes = torch.randint(0, len(SHAPES), (n,), generator=gen)
    phis = torch.deg2rad(lo_deg + (hi_deg - lo_deg) * torch.rand(n, generator=gen))
    return render(shapes, phis), shapes, phis


def circular_onramp(size, dim, sigma_bins=1.5):
    """Fixed on-ramp: pixel at geometric angle theta from center projects to a
    wrapped gaussian bump on the ring centered at channel theta*D/2pi, scaled
    by radius (center pixels are uninformative about angle)."""
    c = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    gy, gx = torch.meshgrid(c, c, indexing='ij')
    gx, gy = gx.reshape(-1), gy.reshape(-1)
    theta = torch.atan2(gy, gx)                              # [P]
    r = (gx ** 2 + gy ** 2).sqrt()
    bins = torch.arange(dim) * (2 * math.pi / dim)           # [D]
    d = (theta[:, None] - bins[None, :] + math.pi) % (2 * math.pi) - math.pi
    sigma = sigma_bins * 2 * math.pi / dim
    p = torch.exp(-(d / sigma) ** 2)                         # [P, D]
    p = p * (r[:, None] / r.max())
    # scale so per-entry RMS ~ 1, matching the learned-init activation scale
    return p / p.pow(2).mean().sqrt()


def probe_angle(hidden_vecs):
    """Population-decode an angle from channel activity. hidden [B, N, D]."""
    m = hidden_vecs.mean(1)                                  # [B, D]
    bins = torch.arange(DIM, device=m.device) * (2 * math.pi / DIM)
    return torch.atan2((m * bins.sin()).sum(-1), (m * bins.cos()).sum(-1))


def circ_err_deg(pred, true):
    d = torch.rad2deg(torch.atan2(torch.sin(pred - true), torch.cos(pred - true)))
    return d.abs().mean().item()


def train(model, data, steps=STEPS):
    model = model.to(DEVICE)
    x, y = data
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(SEED + 1)
    for step in range(steps):
        idx = torch.randint(0, x.shape[0], (BATCH,), generator=gen)
        opt.zero_grad()
        loss = F.cross_entropy(model(x[idx].to(DEVICE)), y[idx].to(DEVICE))
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def evaluate(model, x, y, phi=None):
    model.eval()
    x, y = x.to(DEVICE), y.to(DEVICE)
    acc = (model(x).argmax(-1) == y).float().mean().item()
    err = None
    if phi is not None and hasattr(model, 'hidden'):
        err = circ_err_deg(probe_angle(model.hidden(x)), phi.to(DEVICE))
    model.train()
    return acc, err


def main():
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    train_x, train_y, _ = make_split(N_TRAIN, 0, TRAIN_DEG, gen)
    id_x, id_y, id_phi = make_split(N_TEST, 0, TRAIN_DEG, gen)          # in-dist
    ood_x, ood_y, ood_phi = make_split(N_TEST, TRAIN_DEG, 360, gen)     # held-out

    P = SIZE * SIZE
    onramp = circular_onramp(SIZE, DIM)
    ring_kw = dict(channel_mix='ring', kernel_size=5, readout='pooled')

    # radius groups for weight tying: pixels at equal distance from center
    # share first-layer weights, making the spatial mix rotation-invariant
    c = torch.arange(SIZE, dtype=torch.float32) - (SIZE - 1) / 2
    gy, gx = torch.meshgrid(c, c, indexing='ij')
    radius = (gx ** 2 + gy ** 2).sqrt().reshape(-1)
    rad_groups = torch.unique(radius.round(), sorted=True, return_inverse=True)[1]

    arms = {
        'matrix':        VectorMLP(P, HIDDEN, 3, DIM, channel_mix='matrix',
                                   readout='pooled'),
        'ring-learned':  VectorMLP(P, HIDDEN, 3, DIM, **ring_kw),
        'ring-circular': VectorMLP(P, HIDDEN, 3, DIM, proj_init=onramp,
                                   freeze_proj=True, **ring_kw),
        'ring-radial':   VectorMLP(P, HIDDEN, 3, DIM, proj_init=onramp,
                                   freeze_proj=True, tie_first=rad_groups,
                                   **ring_kw),
    }

    # ring-radial should be *exactly* invariant to 90-degree image rotation
    # (grid permutation = 4-bin ring shift); run 1's arms silently weren't.
    with torch.no_grad():
        img = render(torch.zeros(2, dtype=torch.long), torch.tensor([0.3, 1.1]))
        rot = torch.rot90(img.reshape(2, SIZE, SIZE), 1, (1, 2)).reshape(2, -1)
        gap = (arms['ring-radial'](img) - arms['ring-radial'](rot)).abs().max()
        assert gap < 1e-4, f'90deg invariance broken: {gap:.2e}'
        print(f'90deg-rotation invariance check (ring-radial): max gap {gap:.1e}')
    width, _ = matched_mlp_width(count_params(arms['ring-learned']), P, 3,
                                 len(HIDDEN))
    arms = {'mlp': PlainMLP(P, [width] * len(HIDDEN), 3), **arms}

    print(f'device={DEVICE}  train arc [0,{TRAIN_DEG:.0f})deg  '
          f'chance=33.3%  mlp width={width}')
    print(f"{'arm':<14} {'params':>8} {'in-dist':>8} {'held-out':>9} "
          f"{'probe-id':>9} {'probe-ood':>10}")
    for name, model in arms.items():
        model = train(model, (train_x, train_y))
        acc_id, err_id = evaluate(model, id_x, id_y, id_phi)
        acc_ood, err_ood = evaluate(model, ood_x, ood_y, ood_phi)
        fmt = lambda e: f'{e:9.1f}' if e is not None else '        -'
        print(f'{name:<14} {count_params(model):>8} {acc_id:>8.3f} '
              f'{acc_ood:>9.3f} {fmt(err_id)} {fmt(err_ood)}')


if __name__ == '__main__':
    main()
