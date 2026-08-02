"""Do the per-neuron FFN variants help inside an actual transformer?

The CIFAR rounds put these layers in a small head on top of a CNN that already
solved the task, so every arm landed within ~1pp and the differences were hard
to separate from noise. Here the layer under test is the FFN of a small GPT,
which is ~60% of the model's parameters and is repeated at every layer, so an
effect compounds with depth instead of being diluted. Validation loss is also
far smoother than classification accuracy, so 2-3 seeds resolve what needed 20+
on CIFAR.

Arms (FFN param-matched to the standard block, by shrinking the hidden width):

    mlp        Linear -> GELU -> Linear            the standard block
    lrelu      Linear -> leaky_relu -> Linear      exact control for the
                                                   variants, which are all
                                                   leaky_relu internally
    swiglu     gated FFN (3 projections)
    staged{k}  StagedLinear  -> Linear   k sequential per-neuron bends
    branch{k}  BranchedLinear-> Linear   k parallel per-neuron bends
    nbr{k}     NeighborLinear-> Linear   second stage reads k neurons

Prepare the corpus first (downloads TinyStories, trains a 4k BPE, writes
uint16 bins — all cached):

    python experiments/tinystories_data.py

Then screen cheaply and confirm expensively:

    python experiments/tinystories_ffn.py --tokens 100_000_000 --seeds 3
    python experiments/tinystories_ffn.py --tokens 500_000_000 --seeds 2 \
        --arms mlp,swiglu,staged1,branch1

The first run said: swiglu 1.6512 < mlp 1.6679 < branch4 1.6860 < lrelu 1.7146,
with staged and nbr slightly WORSE than lrelu. So gating wins, branched is the
only per-neuron mechanism that helps, and the single largest effect is just
using GELU instead of leaky_relu. The follow-up arms ask whether branched
survives contact with either of those:

    python experiments/tinystories_ffn.py --tokens 100_000_000 --seeds 3 \
        --arms mlp,swiglu,branch1,branchG1,branchG2,sw-gate1,sw-gate2,sw-post1,sw-post2

    branchG{k}   branched on a GELU base   -> mechanism, or leaky_relu crutch?
    sw-gate{k}   branched on SwiGLU's gate -> complementary to gating?
    sw-post{k}   branched on SwiGLU's product (identity base, so weight 0 is
                 exactly stock SwiGLU)

Each arm reports how far its branch weights moved from init, so a null result
can be told apart from branches that never engaged.

NOTE: this harness has not been executed. Expect to fix something on the first
run; --tokens 2_000_000 --seeds 1 --no-compile is a ~1 minute smoke test.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from branched_linear import BranchedLinear
from conv_ffn import ConvFFN
from neighbor_linear import NeighborLinear
from per_neuron import BranchedActivation, SwiGLUBranched
from quadratic_ffn import QuadraticFFN
from sin_ffn import SinFFN
from staged_linear import StagedLinear
from swiglu import SwiGLU
from vector_mlp import count_params, matched_width
from experiments.tinystories_data import DATA, memmap, prepare

# Every arm compiles the same code objects with different modules, and the
# default limit of 8 is per code object, so without this the later arms
# silently fall back to eager (see BENCHMARKING.md in ComputeNeuron).
torch._dynamo.config.recompile_limit = 64

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_JSON = ROOT / 'results' / 'tinystories_ffn.json'


# --------------------------------------------------------------------------
# FFN variants. Each maps d_model -> d_model with an internal hidden width.
# --------------------------------------------------------------------------

def ffn_mlp(d, h, act='gelu'):
    a = nn.GELU() if act == 'gelu' else nn.LeakyReLU(0.1)
    return nn.Sequential(nn.Linear(d, h), a, nn.Linear(h, d))


def ffn_swiglu(d, h):
    return SwiGLU(d, h, d)


def ffn_macs(module):
    """Multiply-adds per token. Convs report their own, since their cost
    depends on the grid they run at, not just their weight count."""
    if hasattr(module, 'macs_per_token'):
        return module.macs_per_token()
    return sum(m.in_features * m.out_features
               for m in module.modules() if isinstance(m, nn.Linear))


def ffn_staged(d, h, k):
    # StagedLinear already applies its own leaky_relu, so no activation is
    # added between it and the down projection.
    return nn.Sequential(StagedLinear(d, h, extra_stages=k), nn.Linear(h, d))


def ffn_branched(d, h, k):
    return nn.Sequential(BranchedLinear(d, h, extra_branches=k), nn.Linear(h, d))


def ffn_neighbor(d, h, k):
    return nn.Sequential(NeighborLinear(d, h, neighbors=k), nn.Linear(h, d))


def ffn_branch_base(d, h, k, base):
    """Branched activation on an arbitrary base — 'lrelu' reproduces the
    branch{k} arm, 'gelu' is the ablation that asks whether the mechanism
    survives a smooth base or was only rescuing leaky_relu."""
    return nn.Sequential(nn.Linear(d, h),
                         BranchedActivation(h, k, base=base),
                         nn.Linear(h, d))


def ffn_swiglu_branched(d, h, k, where):
    return SwiGLUBranched(d, h, d, extra_branches=k, where=where)


def ffn_builders(d):
    """name -> (build(width) -> Module, min_width, fixed_width).

    fixed_width=None means search for the width that hits the FFN parameter
    budget. A number pins it instead, for arms that are deliberately NOT
    param-matched — the small conv block and the tiny MLP that controls for it.

    For the conv arms the search knob is the first-stage channel count rather
    than a hidden width; peak channels follow at the 17/6 ratio of the spec.
    """
    b = {
        'mlp':    (lambda h: ffn_mlp(d, h, 'gelu'), 1, None),
        'lrelu':  (lambda h: ffn_mlp(d, h, 'lrelu'), 1, None),
        'swiglu': (lambda h: ffn_swiglu(d, h), 1, None),
        'conv':     (lambda c: ConvFFN(d, c1=c, peak='wide'), 4, None),
        'conv-mid': (lambda c: ConvFFN(d, c1=c, peak='mid'), 4, None),
        # Deliberately tiny: 24 -> 10 -> 16 -> 16 -> 10 -> 24, ~9.6K params and
        # 0.71x the standard FFN's arithmetic. Cheaper on BOTH axes, so it is
        # not comparable to the param-matched arms...
        'conv-small': (lambda c: ConvFFN(d, c1=10, c2=16, peak='wide'), 1, 10),
        # ...which is what mlp-tiny is for: hidden 12 gives 9,612 params,
        # within 0.3% of conv-small, so the two can be compared honestly.
        'mlp-tiny': (lambda h: ffn_mlp(d, h, 'gelu'), 1, 12),
    }
    # Grid-cell-ish: a fraction of hidden neurons get periodic tuning curves.
    # sin0 is exactly the mlp arm (no extra params), so the sweep nests its
    # own control; sin100 is all-periodic, i.e. SIREN-style.
    for pct in (0, 12, 25, 50, 100):
        b[f'sin{pct}'] = (lambda h, p=pct: SinFFN(d, h, sin_frac=p / 100), 1, None)
    # frequencies pinned at init: tune the wiring, not the grid
    b['sin25-fixed'] = (lambda h: SinFFN(d, h, sin_frac=0.25,
                                         learn_freq=False), 1, None)
    # Quadratic neurons: rank sweeps width-versus-order at fixed params.
    # rank=1 is a squared activation at full width; high rank is a handful of
    # genuinely second-order neurons.
    for r in (1, 2, 4, 8, 16, 32, 64):
        b[f'quad{r}'] = (lambda h, r=r: QuadraticFFN(d, h, rank=r), 1, None)
    for k in (1, 2, 4):
        b[f'staged{k}'] = (lambda h, k=k: ffn_staged(d, h, k), 1, None)
        b[f'branch{k}'] = (lambda h, k=k: ffn_branched(d, h, k), 1, None)
        b[f'nbr{k}'] = (lambda h, k=k: ffn_neighbor(d, h, k), k, None)
        # does branched survive a smooth base?
        b[f'branchG{k}'] = (lambda h, k=k: ffn_branch_base(d, h, k, 'gelu'), 1, None)
        # does it add anything on top of gating, which is what actually won?
        b[f'sw-gate{k}'] = (lambda h, k=k: ffn_swiglu_branched(d, h, k, 'gate'), 1, None)
        b[f'sw-post{k}'] = (lambda h, k=k: ffn_swiglu_branched(d, h, k, 'post'), 1, None)
    return b


# --------------------------------------------------------------------------
# A small GPT, nanoGPT-shaped, with the FFN factored out.
# --------------------------------------------------------------------------

def last_linear(module):
    """The final nn.Linear inside a block, whatever its structure.

    The residual-scaled init has to land on the same place in every arm. Its
    index differs by variant (Sequential[2] for the plain MLP, Sequential[1]
    for the staged/branched/neighbor pairs, .down for SwiGLU), so matching on
    a name would quietly give the arms different initialisations — a confound
    that would look like an architecture effect.
    """
    found = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            found = m
    return found


class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        assert d % n_head == 0
        self.n_head, self.d = n_head, d
        self.attn = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.attn(x).split(C, dim=2)
        shape = (B, T, self.n_head, C // self.n_head)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, d, n_head, make_ffn):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_head)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = make_ffn()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, d, n_layer, n_head, block, make_ffn):
        super().__init__()
        self.block_size = block
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(block, d)
        self.blocks = nn.ModuleList(
            [Block(d, n_head, make_ffn) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.wte.weight          # tied

        self.apply(self._init)
        # residual-scaled init on whatever each block writes back into the
        # stream: the attention projection, and the FFN's final Linear
        # wherever it happens to live in that variant
        scale = 0.02 / math.sqrt(2 * n_layer)
        for b in self.blocks:
            nn.init.normal_(b.attn.proj.weight, std=scale)
            out = last_linear(b.ffn)
            if out is not None:
                nn.init.normal_(out.weight, std=scale)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.reshape(-1))
        return logits, loss


# --------------------------------------------------------------------------

def batches(data, batch, block, device, generator=None):
    """Random windows into the flat token array, nanoGPT style."""
    while True:
        # .tolist() so numpy gets plain ints rather than torch scalars
        ix = torch.randint(len(data) - block - 1, (batch,),
                           generator=generator).tolist()
        x = torch.stack([torch.from_numpy(
            data[i:i + block].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(
            data[i + 1:i + 1 + block].astype(np.int64)) for i in ix])
        yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, data, batch, block, device, iters=50):
    model.eval()
    gen = torch.Generator().manual_seed(1234)       # same windows every time
    it = batches(data, batch, block, device, gen)
    total = 0.0
    for _ in range(iters):
        x, y = next(it)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / iters


def lr_at(step, total, lr, warmup):
    """Linear warmup then cosine decay to 0.1 * lr."""
    if step < warmup:
        return lr * (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.1 * lr + 0.45 * lr * (1 + math.cos(math.pi * t))


def train_one(make_ffn, vocab, args, seed, train_data, val_data):
    torch.manual_seed(seed)
    model = GPT(vocab, args.d_model, args.n_layer, args.n_head,
                args.block, make_ffn).to(DEVICE)
    n_params = count_params(model)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{'params': decay, 'weight_decay': 0.1},
         {'params': nodecay, 'weight_decay': 0.0}],
        lr=args.lr, betas=(0.9, 0.95))

    net = model
    if not args.no_compile:
        torch._dynamo.reset()
        net = torch.compile(model)

    steps = args.tokens // (args.batch * args.block)
    gen = torch.Generator().manual_seed(seed)
    it = batches(train_data, args.batch, args.block, DEVICE, gen)

    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g['lr'] = lr_at(step, steps, args.lr, args.warmup)
        x, y = next(it)
        with torch.autocast('cuda', dtype=torch.bfloat16,
                            enabled=DEVICE == 'cuda'):
            _, loss = net(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if args.log_every and step % args.log_every == 0:
            print(f'    step {step}/{steps}  loss {loss.item():.4f}  '
                  f'{(step + 1) * args.batch * args.block / (time.time() - t0):,.0f} tok/s',
                  flush=True)

    val = evaluate(net, val_data, args.batch, args.block, DEVICE, args.eval_iters)
    if not args.no_compile:
        torch._dynamo.reset()

    # Did the branches actually engage? Without this a null result is ambiguous
    # between "the mechanism does not help" and "the bends were initialised
    # somewhere the data never reaches", which are very different conclusions.
    drifts = [d for d in (m.drift() for m in model.modules()
                          if hasattr(m, 'drift')) if d]
    drift = {}
    if drifts:
        drift = {k: float(np.mean([d[k] for d in drifts])) for k in drifts[0]}
    return val, n_params, time.time() - t0, drift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', type=int, default=100_000_000)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--arms', default=None,
                    help='comma-separated subset, e.g. mlp,swiglu,staged1')
    ap.add_argument('--vocab', type=int, default=4096)
    ap.add_argument('--d-model', type=int, default=384)
    ap.add_argument('--n-layer', type=int, default=6)
    ap.add_argument('--n-head', type=int, default=6)
    ap.add_argument('--block', type=int, default=256)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--eval-iters', type=int, default=50)
    ap.add_argument('--log-every', type=int, default=500)
    ap.add_argument('--no-compile', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    prepare(args.vocab)                       # cached after the first run
    train_data = memmap(DATA / f'train{args.vocab}.bin')
    val_data = memmap(DATA / f'val{args.vocab}.bin')
    print(f'corpus: {len(train_data):,} train tokens, {len(val_data):,} val',
          flush=True)

    d = args.d_model
    builders = ffn_builders(d)
    if args.arms:
        want = [a.strip() for a in args.arms.split(',')]
        missing = [a for a in want if a not in builders]
        if missing:
            ap.error(f'unknown arms {missing}; have {sorted(builders)}')
        builders = {a: builders[a] for a in want}

    # FFN parameter budget = the standard block at 4x expansion. Every variant
    # gets its hidden width shrunk until it fits, so arms differ in FFN
    # structure and not in size.
    target = count_params(ffn_mlp(d, 4 * d))
    print(f'\nmodel {args.n_layer}L/{d}d/{args.n_head}H block {args.block} | '
          f'FFN budget {target:,} params (mlp 4x = {4 * d} hidden)\n', flush=True)

    base_macs = ffn_macs(ffn_mlp(d, 4 * d))
    plans = {}
    for name, (build, min_h, fixed) in builders.items():
        if fixed is None:
            h, got = matched_width(target, build, min_w=min_h)
            tag = ''
        else:
            h = fixed
            got = count_params(build(h))
            tag = '  NOT param-matched'
        plans[name] = (build, h)
        macs = ffn_macs(build(h))
        # arms are param-matched unless tagged; conv arms are never
        # flop-matched, since weight sharing decouples params from compute
        print(f'  {name:<11} width {h:<5} {got:>9,} params '
              f'({got/target:>5.1%} of budget)  {macs/1e6:>7.2f}M MACs/token '
              f'({macs/base_macs:>5.2f}x){tag}', flush=True)

    steps = args.tokens // (args.batch * args.block)
    print(f'\n{args.tokens:,} tokens = {steps:,} steps of '
          f'{args.batch}x{args.block}\n', flush=True)

    results = {}
    for name, (build, h) in plans.items():
        losses, drift = [], {}
        for seed in range(args.seeds):
            val, n_params, secs, drift = train_one(
                lambda: build(h), args.vocab, args, seed, train_data, val_data)
            losses.append(val)
            if 'omega_mean' in drift:
                extra = (f'  omega {drift["omega_mean"]:.2f} '
                         f'(init {drift["omega_init_mean"]:.2f}, '
                         f'moved {drift["omega_drift"]:.2f}, '
                         f'range {drift["omega_min"]:.2f}-{drift["omega_max"]:.2f})')
            elif drift:
                extra = (f'  |w| {drift["w_abs_mean"]:.3f} '
                         f'(init {drift["w_abs_init"]:.3f}, '
                         f'moved {drift["w_drift"]:.3f})')
            else:
                extra = ''
            print(f'--- {name} seed {seed}: val loss {val:.4f}  '
                  f'({n_params:,} params, {secs/60:.1f} min){extra}', flush=True)
        arr = np.array(losses)
        results[name] = {'hidden': h, 'params': n_params,
                         'val_loss': losses, 'drift': drift,
                         'mean': float(arr.mean()), 'std': float(arr.std())}
        print(f'=== {name}: {arr.mean():.4f} +- {arr.std():.4f} '
              f'(ppl {math.exp(arr.mean()):.2f})\n', flush=True)

    out = Path(args.out) if args.out else OUT_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'args': vars(args), 'results': results}, f, indent=1)

    print('\nsummary (lower is better):')
    for name, r in sorted(results.items(), key=lambda kv: kv[1]['mean']):
        print(f'  {name:<11} {r["mean"]:.4f} +- {r["std"]:.4f}  '
              f'ppl {math.exp(r["mean"]):.2f}  width {r["hidden"]}')
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
