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

NOTE: run executed on a remote box; per-seed json (`mnist_results3.json`) not
retrieved yet, so gaps below are mean-level (still same seeded subsets), no
seed-pairing counts.

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

## Queue

- FLOP-matched MLP control for exp 2.
- Rot45 gap vs rotation angle; pose linear probe (why is matrix robust?).
- Middle rungs of the structure ladder (init-not-frozen, soft radial tying,
  small-data angle task).
- Transformer FFN-slot drop-in (reshape d_model as N x D residual stream),
  tiny-shakespeare — the real "drop into an existing model" test. Consider
  GPU rental here, not before.
- Pretrained-backbone head test (cache DINO-family features; supervised
  backbones likely destroyed pose info).
