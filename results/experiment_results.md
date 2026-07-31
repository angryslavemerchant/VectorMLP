# Channel-Mixing Vector Neuron — Experiment Results

Companion to `channel_mixing_neuron_spec.md`. Code: `vector_mlp.py` (layers,
models, param matcher), `angle_task.py` (exp 1), `mnist_task.py` (exp 2, raw
numbers in `mnist_results.json`). Env: conda `Toastenv` (CUDA torch);
`KMP_DUPLICATE_LIB_OK=TRUE` needed on this machine.

## Implementation notes (learned the hard way)

- **Init matters:** nn.Linear-default init collapses the net — activation
  variance decays ~3x/layer and the negative gate bias shuts every gate by
  layer 2 (exactly-zero gradients). Fix: He-scaled weights N(0, 2/n_in) +
  unit-std on-ramp. Bias init must be read relative to pre-gate std (~0.8).
- **Per-channel bias breaks ring shift-equivariance** (bias pattern doesn't
  rotate with the input). Ring variant defaults to scalar bias per neuron.
- Training many small models: stack seeds/sizes into one vmapped computation
  (`torch.func`); 100 runs ran in ~25 min on the RTX 2060 at 84% util.
  Sequential looping would have idled the GPU (~1% util). Renting bigger GPUs
  doesn't fix launch overhead; batching does.

## Experiment 1 — synthetic angle kill test (`angle_task.py`)

16x16 images, 3 shapes (bar+blob / cross+blob / T) at angle phi, train
phi in [0,120) deg, test held-out [120,360). Chance 33%. Invariant (pooled)
readout; per-angle class bins are impossible OOD for any model (untrained
class weights), hence shape classification + zero-param population probe.

| arm | params | in-dist | held-out |
|---|---|---|---|
| MLP (param-matched) | 25.2k | 1.000 | 0.62 (stable across 8 seeds) |
| matrix, learned on-ramp | 60.4k | 1.000 | 0.61 |
| ring, learned on-ramp (arm A) | 25.6k | 1.000 | 0.61 |
| ring, circular on-ramp (arm B as specced) | 21.5k | 1.000 | 0.58 |
| ring + circular on-ramp + radial tying | 5.8k | 1.000 | **1.000** (6/8 seeds; 2/8 at 0.665) |

Findings:

1. **Spec's equivariance story was incomplete.** Circular channel topology is
   not sufficient: the first layer's per-pixel spatial weights re-break
   equivariance (rotation also permutes which pixels are bright). Fix: tie
   first-layer weights across pixels at equal radius (`tie_groups` in
   VectorLinear). With tying, 90-deg rotation invariance is exact (gap 2e-8)
   and held-out generalization is perfect at 4x fewer params. A pipeline is
   only as equivariant as its least equivariant stage.
2. **Structure is unlearnable from the loss** — all learned arms hit zero
   training loss without it, so no gradient favors the equivariant solution.
   We landed in the "only hand-wired generalizes" cell of the 2x2.
3. **Failure mode of the 2/8 stuck seeds:** cross class merges into bar
   (cross logit never wins; gates otherwise healthy). Optimization, not
   architecture; likely fixable with init/warmup.
4. Identity x attribute factorization worked as designed: shape in *which
   neuron*, angle on *the channel ring* (population probe reads angle at ~44
   deg error, identical in-dist and OOD). Structure survives depth 2.
5. Honest framing: the winning arm is essentially a hand-built polar CNN.
   The neuron is a **container, not a generator** of structure. Open ladder:
   hand-wired-init-unfrozen, soft radial tying, data regimes where structure
   pays in-distribution.

## Experiment 2 — MNIST sample efficiency (`mnist_task.py`)

Fully learned, no hand-wiring. D=16, hidden [64,64], 4k steps Adam 1e-3,
5 seeds, subsets shared across arms (paired). Sizes 500/2k/10k/60k. Metrics:
clean test acc + rotated (+-45 deg) test acc.

Clean acc (means):

| arm | params | 500 | 2k | 10k | 60k |
|---|---|---|---|---|---|
| matrix | 105.2k | **87.4** | **92.2** | 95.7 | 97.5 |
| mlp@matrix | 104.8k | 86.1 | 91.2 | 95.5 | **97.7** |
| ring | 68.4k | 86.4 | 91.8 | 95.3 | 97.3 |
| mlp@ring | 68.2k | 86.0 | 91.1 | 95.2 | 97.5 |
| shared | 70.6k | 86.2 | 91.6 | 95.1 | 97.3 |

Paired per-seed gaps (matrix minus mlp@matrix, percentage points):

- clean: +1.29+-0.52 (n=500, 5/5 seeds), +0.98 (2k, 5/5), +0.19 (10k),
  -0.15 (60k) — the predicted sample-efficiency crossover.
- rot45: +1.72 (500), +1.84 (2k), +1.16 (10k, all 5/5), +0.39 (60k) —
  **rotation-robustness edge bigger and longer-lived than the clean edge**,
  with nothing enforcing equivariance. Most interesting unexplained result.
  Follow-ups: gap vs rotation angle; linear-probe hidden vectors for pose.

Findings:

1. Per-neuron DxD matrices are where the value is; ring and shared show
   marginal-to-no edge over matched MLPs on i.i.d. data. Topology only pays
   when symmetry is imposed end-to-end (exp 1).
2. Caveats: one dataset, one config, no per-arm tuning, and matched-params
   means ~3x FLOPs for vector arms (weight sharing: compute scales with N*D,
   params don't). Next control: FLOP-matched (wider) MLP — expect it to eat
   part of the 1.3-pt gap.

## Experiment 2b — ring-kernel ladder + matrix D-sweep (`mnist_task2.py`)

All arms width-matched to round 1's matrix arm (105.2k params; widths
102/99/94/108), same seeds and identical data subsets -> per-seed paired with
round 1. Raw numbers: `mnist_results2.json`. Ring now supports even kernels
(asymmetric circular pad); shift-equivariance re-verified for K=3/5/16.

Paired gaps vs mlp@matrix, clean / rot45, percentage points at n=500 -> 60k:

| arm | clean 500 | clean 2k | rot45 500 | rot45 2k | rot45 10k |
|---|---|---|---|---|---|
| matrix D=16 (round 1) | +1.29 (5/5) | +0.98 | +1.72 (5/5) | +1.84 | +1.16 |
| ring3 D=16 | +1.18 (5/5) | +0.73 | +1.02 | +0.77 | +0.46 |
| ring16 D=16 | +0.90 (5/5) | +0.85 | +0.66 | +0.96 | +0.38 |
| matrix D=8 | +0.58 | +0.95 | +0.97 | +1.20 | +1.03 |
| matrix D=4 | -2.71* | +0.67 | -2.17* | +0.77 | +1.05 |

Findings:

1. **The clean sample-efficiency edge only needs *some* per-neuron channel
   mixing.** At matched params, even K=3 ring recovers ~90% of matrix's clean
   edge at n=500. Round 1's "per-neuron DxD is where the value is" was partly
   a width artifact (round-1 ring ran at 68k).
2. **Reach on the ring is worthless: ring16 (full circulant, the most
   expressive shift-equivariant mixer) is no better than ring3** and both
   trail matrix on rot45 by a consistent 0.7-1.1 pts (0-1/5 seeds). So the
   rotation-robustness edge specifically requires *breaking* the ring
   symmetry — channels with individual identities, not interchangeable
   positions. Deepens the rot45 mystery rather than resolving it.
3. **D-sweep at fixed params:** D=8 (94 neurons) ~ D=16 (64 neurons) — the
   factorization is insensitive over that range. D=4 is fragile: one n=500
   seed collapsed (*std 5x the others; recovers by n=2k), and it's the worst
   arm small-data. Sweet spot D in [8, 16].
4. Vs matrix head-to-head, every new arm is equal-or-worse everywhere;
   matrix D=16 remains the best configuration tested.

## Experiment 2c — D-sweep up (32, 64) + low-rank mixing (`mnist_task3.py`)

Low-rank mixer added to VectorLinear: y = g + U(V g), rank r, 2*D*r
params/neuron. At the 105k budget full-matrix D=64 is degenerate (matched
width = 1: on-ramp 784*64 = 50k + readout matrices 41k), so low-rank is what
makes D>=32 viable at all. Widths: lowrank-d16-r4 78, matrix-d32 23,
lowrank-d32-r4 53, lowrank-d64-r4 24.

(Per-seed json retrieved; paired counts confirm the mean-level reads below —
matrix-d32 rot45 vs mlp@matrix: +2.14 (5/5) / +1.84 (5/5) / +1.02 (5/5) at
500/2k/10k.)

Mean gaps (pp) vs round-1 baselines:

| arm | vs matrix, clean 500/2k | vs matrix, rot45 500/2k | vs mlp@matrix, rot45 500/2k |
|---|---|---|---|
| lowrank-d16-r4 | -0.23 / -0.32 | -0.87 / -0.93 | +0.84 / +0.92 |
| matrix-d32 (23 neurons) | -0.05 / -0.10 | **+0.42 / -0.01** | **+2.13 / +1.84** |
| lowrank-d32-r4 | -0.32 / -0.33 | -0.52 / -0.96 | +1.19 / +0.89 |
| lowrank-d64-r4 | -0.55 / -0.72 | -0.31 / -0.67 | +1.40 / +1.18 |

Findings:

1. **matrix-d32 with 23 neurons ties the D=16 flagship on clean and posts the
   best small-data rot45 of any arm tested** (+2.13 over the MLP at n=500).
   The rot45 edge tracks full per-neuron matrices at the largest affordable D
   — 23 fat vector neurons beat 64 thinner ones on robustness.
2. **Rank-4 mixing keeps the clean edge but gives up ~half the small-n rot45
   edge** (lowrank-d16 -0.9 pp vs matrix at 500/2k, converging by 60k). The
   clean sample-efficiency benefit is cheap (consistent with round 2's ring3);
   the robustness benefit needs high-rank mixing. Whatever matrix learns
   about pose, it uses many mixing directions.
3. **Low-rank does not rescue big D:** lowrank d32/d64 degrade monotonically
   with D at n>=10k. The 784*D on-ramp tax buys channel resolution that never
   pays for itself; D=64 is the worst large-n arm.
4. Combined sweet spot after rounds 2b+2c: full matrix, D in [16, 32].

## Experiment 2d — pure low-rank factorization (`mnist_task4.py`)

y = U(V g), no identity path: the whole mixer is rank-r (vs 2c's residual
form g + UVg). purelr-d16-r8 is param-IDENTICAL to the round-1 matrix
flagship (2*16*8 = 256 = 16^2 per neuron; same width 64, same 105,152 total)
— pure reparameterization test. purelr-d16-r4 gets width 78.

Paired per-seed gaps (pp):

| arm | vs mlp@matrix clean 500 | vs mlp@matrix rot45 500/2k/10k | vs matrix clean 500 | vs matrix rot45 500 |
|---|---|---|---|---|
| purelr-d16-r4 | +1.30 (5/5) | +1.65/+1.52/+1.29 (all 5/5) | +0.01 | -0.07 |
| purelr-d16-r8 | **+1.66 (5/5)** | +1.87/+1.73/+1.30 (all 5/5) | **+0.37 (5/5)** | +0.15 |

Findings:

1. **Prediction falsified: rank is NOT the constraint.** Pure rank-4 matches
   the full matrix everywhere, including rot45. Round 2c's "robustness needs
   high-rank mixing" was wrong — the residual form's weakness was never its
   rank.
2. **The residual identity path is what hurt.** Residual r4 lost ~0.9 pp
   rot45 vs matrix; pure r4 loses nothing. Likely init/optimization: residual
   starts with mixing ~ 0 (U,V both 0.01 -> product 1e-4) and must grow it
   against a working identity path; pure starts with variance-preserving
   full-strength random mixing it must immediately use. Mixing strength at
   init, not mixer rank, looks like the operative variable.
3. **purelr-d16-r8 slightly beats its param-identical full-matrix twin on
   clean small-data** (+0.37, 5/5 at n=500; best clean edge of any arm at
   +1.66 over the MLP). Factorized parameterization of the same map acts as
   a mild regularizer.
4. Practical: rank-4 pure halves mixer params/FLOPs with zero measured cost —
   the default mixer for the transformer FFN-slot test should be
   lowrank_pure r=D/4 (and it removes the D^2 obstacle to D=32/64 there).

## Experiment 3 — frozen DINO features, head swap (`cifar_head_task.py`)

CIFAR-10 through frozen DINO ViT-S/16 (cached 384-d CLS features,
`cifar_features.py`); heads at ~26k params, 5 seeds, sizes 500/2k/10k/50k.
The "guest in a scalar world" test: 384 features reshaped to 24 x 16 vectors,
NO on-ramp (vector_in=True). rot45 = images rotated before the backbone.

Means (clean): vec 82.1/87.6/91.2/93.4 vs mlp 90.2/92.5/94.2/95.4 —
**vector head loses by 5-8 pts everywhere**; rot45 worse (-14 pts at 500).
vec-perm == vec (grouping irrelevant).

Findings:

1. **Free-reshape drop-in is dead.** Diagnosis: input wiring starvation —
   the reshape gives 24x64 ~ 1.5k input weights (one scalar per GROUP of 16
   features) vs the MLP's 384x57 ~ 22k (every feature individually
   weighted). DINO's 384 dims are individually meaningful with no channel
   structure to exploit (perm control confirms), so forcing features to
   travel in gangs of 16 throws away most of the input resolution.
2. Container-not-generator again, now at the interface: a frozen scalar
   representation has nothing to pour into the channels. MNIST worked
   because the learned on-ramp MANUFACTURED channel structure; the reshape
   removed the on-ramp and the frozen host supplied nothing.
3. Not killed: co-adapting hosts (transformer FFN slot — the host trains
   with the vector layer and can develop channel structure to feed it).
4. Diagnostic arm (`vec-onramp`, width 26): learned per-feature 16-d
   directions on the same features — restores per-feature wiring + learned
   channel placement at the same param count. Result: 84.4/89.3/92.7/94.3
   clean — recovers ~2.3 of the ~8-pt gap, still ~6 behind the MLP
   everywhere. **The neuron itself underperforms on frozen feature-space
   inputs**, not just the reshape interface. Caveat: the interface cost
   squeezed hidden width to 26, so a FLOP-matched version might close more;
   but the shape of the verdict is clear.

**Verdict for the drop-in program:** vector neurons earn their keep when
they build the representation (pixels + learned on-ramp: sample-efficiency
and rotation-robustness edges, 5/5 seeds) and lose when consuming someone
else's scalar representation (frozen features: -6 to -8 pts). The open
middle case — a host that trains jointly and can co-adapt (transformer FFN
slot) — remains untested and is the only untried branch of the original
drop-in claim.

## Experiment 3b — mixer ablations on DINO heads (round-1 arms 5-6)

vec-none (no mixing = 16 disjoint thin MLPs, width 131) and vec-ring (K=5
circular conv, width 140), same 26k budget. Paired vs vec (clean):

    vec-none - vec:  -10.28 / -6.17 / -3.07 / -0.69   (0/5 everywhere)
    vec-ring - vec:  -10.43 / -4.37 / -1.97 / -0.78   (0/5 everywhere)

Even where the vector neuron loses to the MLP, its mixer is doing heavy
lifting: removing it costs another 10 pts at n=500. Ladder on frozen
features: none ~ ring < vec (rank-4) < vec-onramp < mlp. Cheap mixers do
NOT buy their width back here (opposite of MNIST round 2b, where ring3
recovered ~90% of the matrix edge) — on unstructured features, arbitrary
neighbor topology is worthless and only full learned mixing helps.

## Experiment 4 — end-to-end small CNN, co-adapting host (`cifar_e2e.py`)

SmallCNN backbone (94k) trained jointly with each head (~33k), CIFAR
pixels, sizes 2k/10k/50k. cnn-vec = reshape 2048 -> 128 x 16, purelr-r4.

    cnn-vec - cnn-mlp (clean):  +2.24 (4/5) / -1.52 (1/5) / -1.57 (0/5)
    rot45: ~0 everywhere.

**Co-adaptation is real but partial**: the trainable backbone erases the
catastrophic frozen-feature gap (-6 to -8 -> ~+2/-1.5) and restores the
small-data edge, but the old neuron still loses at scale. The MNIST
pattern (edge shrinks as n grows) reappears, now crossing zero.

## Experiment 5 — new neuron variants A/B (`*_task.py 2`, `cifar_e2e.py 2`)

Architecture pivot (docs/vector_neuron_variants_ab.md): ProjNet = Variant A
(signals are single D-vectors, modReLU magnitude gate, per-neuron DxD
projection — quadratic-form interference detector); TagNet = Variant B
(scalar path x bounded agreement gain; 'weighted' reuses connection weights
in the direction path, 'query' uses a per-neuron query on a shared field).
D=2-4, all param-matched to the round-1 vec targets, same seeds/subsets.

DINO heads (clean, means): proj-d2 88.9/91.6/93.0/95.1, proj-d4
88.7/91.5/92.9/95.0, tagw-d4 89.1/92.0/93.9/95.1, tagq-d4
86.9/90.8/93.2/94.8. Paired highlights:

1. **The new neurons leapfrog the whole old family.** proj-d4 - vec: +6.60
   / +3.87 / +1.64 / +1.61, tagw-d4 - vec: +6.99 / +4.41 / +2.69 / +1.71
   (all 5/5). proj-d4 beats even vec-onramp +4.33 (5/5) at n=500. The gap
   to the MLP shrinks from -8 to about -1 (still 0/5 clean — mlp keeps a
   small consistent edge on frozen features).
2. **First rot45 win on real images:** tagw-d4 - mlp rot45 = +1.27 (4/5)
   at n=10k, +1.63 (4/5) at n=50k (loses at small n). proj-d4 reaches
   parity at 50k. The agreement-gated scalar neuron generalizes to rotated
   inputs better than the MLP once it has data.
3. D=2 ~ D=4 for ProjNet on clean (geometry is cheap); d4 slightly better
   rot45. Ablate D upward only if a use case demands it.

End-to-end CNN (clean, means): cnn-proj-d4 47.9/62.4/73.1 — **best arm in
the whole e2e table at every size**, despite head width 11 (it pays the
full 2048-input wiring tax):

    cnn-proj-d4 - cnn-mlp: +4.18 (5/5) / +1.25 (4/5) / +0.26 (3/5)
    cnn-proj-d4 - cnn-vec: +1.94 / +2.77 / +1.83  (all 5/5)

First architecture in the project to beat the param-matched MLP with a
trainable host across all sizes (means; 50k is within noise). TagNets
collapse end-to-end (tagw -6 to -15, tagq -5 to -17, seed std up to 14):
the multiplicative gain destabilizes joint training with a backbone —
usable as a head on fixed features, fragile as a training partner.

**Verdict update:** Variant A (ProjNet) supersedes the channel-mixing
neuron: better on frozen features (+2 to +7), better end-to-end (+2 to +3),
and the only arm ever ahead of the MLP under co-adaptation. TagNet-weighted
holds the single rot45-beats-mlp result on frozen features. The old
neuron's remaining unique win is the MNIST pixel regime.

## Queue

- FLOP-matched MLP control for exp 2 (and now for proj-d4 e2e, where the
  width-11 squeeze makes FLOP-matching bite in the other direction).
- TagNet e2e instability: warmup or gain clamp if we ever need B in a
  trainable host.
- Rot45 gap vs rotation angle; pose linear probe (why is matrix robust?).
- Middle rungs of the structure ladder (init-not-frozen, soft radial tying,
  small-data angle task).
- Transformer FFN-slot drop-in (reshape d_model as N x D residual stream),
  tiny-shakespeare — the real "drop into an existing model" test. Consider
  GPU rental here, not before.
- Pretrained-backbone head test (cache DINO-family features; supervised
  backbones likely destroyed pose info).
