"""One-off: cache DINO ViT-S/16 features for CIFAR-10.

Extracts 384-dim CLS features for the train set, clean test set, and a
+-45deg-rotated test set (rotation applied to pixels BEFORE the backbone,
so the features genuinely see rotated objects). Saved float16 to
data/cifar_dino_vits16.pt (~70 MB). Downloads DINO weights (~80 MB) and
CIFAR-10 (~170 MB) on first run.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

# HF CDN mirror of the exact official tarball (same size + checksum; the
# Toronto origin server throttles to ~80 kB/s). torchvision still verifies
# the md5 after download.
CIFAR10.url = ('https://huggingface.co/datasets/liangnanying/cifar-10-python'
               '/resolve/main/cifar-10-python.tar.gz')

OUT = ROOT / 'data' / 'cifar_dino_vits16.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH = 128
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def rotated(images, max_deg=45.0, seed=123):
    """images [N, 3, 32, 32] in [0,1] -> each rotated by U(-max_deg, max_deg)."""
    g = torch.Generator().manual_seed(seed)
    n = images.shape[0]
    ang = torch.deg2rad((torch.rand(n, generator=g) * 2 - 1) * max_deg)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
    theta[:, 1, 0], theta[:, 1, 1] = sin, cos
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(images, grid, align_corners=False)


@torch.no_grad()
def extract(model, images):
    """images [N, 3, 32, 32] in [0,1] -> [N, 384] float16 CLS features."""
    feats = []
    for i in range(0, images.shape[0], BATCH):
        x = images[i:i + BATCH].to(DEVICE)
        x = F.interpolate(x, size=224, mode='bicubic', align_corners=False)
        x = (x - IMAGENET_MEAN.to(DEVICE)) / IMAGENET_STD.to(DEVICE)
        feats.append(model(x).half().cpu())
        if (i // BATCH) % 50 == 0:
            print(f'  {i}/{images.shape[0]}', flush=True)
    return torch.cat(feats)


def main():
    tr = CIFAR10(ROOT / 'data', train=True, download=True)
    te = CIFAR10(ROOT / 'data', train=False, download=True)
    to_t = lambda d: torch.from_numpy(d).permute(0, 3, 1, 2).float() / 255.0
    tr_x, te_x = to_t(tr.data), to_t(te.data)

    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    model = model.to(DEVICE).eval()

    print('train features...', flush=True)
    train_f = extract(model, tr_x)
    print('test features...', flush=True)
    test_f = extract(model, te_x)
    print('rotated test features...', flush=True)
    test_rot_f = extract(model, rotated(te_x))

    torch.save({'train_x': train_f, 'train_y': torch.tensor(tr.targets),
                'test_x': test_f, 'test_y': torch.tensor(te.targets),
                'test_rot_x': test_rot_f}, OUT)
    print(f'saved -> {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()
