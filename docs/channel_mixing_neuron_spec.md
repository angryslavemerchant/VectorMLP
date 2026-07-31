# Channel-Mixing Vector Neuron — Experiment Spec (v2)

Supersedes `vector_neuron_experiment_spec.md`. The original magnitude-gate
(‖r‖², NMDA-agreement) design was dropped after review; this branch bets on
**structured representation**, not multiplicative interactions.

## Core Idea

Each neuron carries a D-dimensional vector. The layer factorizes into two
mixing steps — across neurons with scalar weights, then across channels with a
small per-neuron learned transform — with a per-channel threshold gate between
them. Flattened, this is a **factorized MLP**: same piecewise-linear function
class as a plain MLP, but ~D²-fold parameter sharing (same structural bet as
MLP-Mixer / depthwise-separable convs). The hypothesis: "which neuron fires"
encodes feature *identity*, the vector encodes feature *attributes* (pose,
orientation, phase), and the per-neuron transform learns attribute-space maps.

## Neuron Definition

One neuron, N inputs, vector dim D (hyperparameter).
Learned: N scalar connection weights wᵢ, bias vector b ∈ ℝᴰ, matrix M ∈ ℝᴰˣᴰ
(or conv kernel k ∈ ℝᴷ in the ring variant).

1. **Receive** N incoming D-dim vectors x₁…xₙ.
2. **Mix across neurons:** r = Σᵢ wᵢxᵢ  (resultant, ℝᴰ).
3. **Gate per channel:** g = ReLU(r + b) elementwise. Initialize b negative so
   channels actually threshold; g is typically sparse.
4. **Mix across channels:** y = M·g (matrix variant) or y = k ⊛ g, circular
   conv with wraparound (ring variant).
5. **Emit** y ∈ ℝᴰ.

Ordering note: gate sits *between* the mixes (mix → gate → matrix) = "detect a
sparse pattern, then broadcast a chosen direction." Micro-variant: mix →
matrix → gate (sparse *outputs* instead). Default to gate-then-matrix.

### Ring variant

Channels get a circular ordering (bin k ≙ angle 2πk/D). Channel mixing is a
learned 1-D circular convolution (K ≪ D, e.g. 3–5 taps). Forces shift
equivariance: input angle rotates → channel pattern circularly shifts → output
shifts identically. Cheaper (K params vs D²) AND stronger inductive bias.

### Dropped from v1 (deliberately)

No ‖r‖² magnitude gate, no unit-direction output, no agreement/NMDA mechanism.
Optional recombination if ever wanted: multiply y by ReLU(‖r‖² + β).

## Layer Implementation (PyTorch)

```python
# x: [batch, N_in, D]   W: [N_in, N_out]   b: [N_out, D]   M: [N_out, D, D]
r = torch.einsum('bnd,nm->bmd', x, W)     # mix across neurons
g = F.relu(r + b)                          # per-channel gate
y = torch.einsum('mde,bme->bmd', M, g)     # per-neuron channel mix
# Ring variant: replace last line with grouped circular conv1d over the D axis.
```

Never materialize [batch, N_in, N_out, D] — the einsum above is just a batched
matmul.

### On/off-ramps (unchanged from v1)

- Input: per-input learned projection pᵢ ∈ ℝᴰ, vᵢ = xᵢ·pᵢ (input_dim × D params).
- Output: logit_k = output_vector · cₖ, cₖ ∈ ℝᴰ (num_classes × D params).

### Parameter accounting

Layer of N_in→N_out: N_in·N_out (weights) + N_out·D (biases) + N_out·D²
(matrices, or N_out·K ring). A plain MLP over the flattened N·D representation
would cost N_in·N_out·D². At matched parameters the vector net gets a ~D×
wider representation.

## Predictions

- Generic i.i.d. benchmarks (MNIST/CIFAR flat): ≈ tie with a param-matched MLP.
  Slight win per-parameter, slight loss per-FLOP. Not the interesting regime.
- The payoff, if real, is **OOD generalization over a transformation** and
  sample efficiency — tasks whose latent structure matches the
  identity × attribute factorization. Ring variant should show it hardest.

## Experiment Plan

Win condition is generalization, not i.i.d. accuracy. Ascending effort:

1. **Synthetic angle task (kill test — do first).** Noisy oriented feature
   (Gabor patch / 2-point pattern) at angle φ; classify pattern + regress φ.
   Train φ ∈ [0°,180°), test held-out ranges. Prediction: MLP fails held-out
   angles; ring variant generalizes (rotation = channel shift). No gap here →
   branch is dead. An afternoon of work.
2. **Rotated MNIST, limited-rotation training.** Train 0–45°, test 45–360°.
   Also linear-probe hidden vectors for rotation angle — if the vector really
   carries pose, a linear probe reads it out.
3. **smallNORB viewpoint generalization.** Train subset of azimuths/elevations,
   test unseen. Capsules' one demonstrated win; tests whether this cheaper
   structure recovers it.
4. **Skip CIFAR** — nothing there rewards pose factorization at MLP scale.

### Arms / baselines

- Param-matched plain MLP (must-have).
- Per-neuron D×D matrix vs ring-conv (does *topology* matter or just extra
  mixing?).
- Shared per-layer matrix (Mixer-style, near-free) vs per-neuron (do per-neuron
  matrices earn their keep?).
- No CNN baseline — it wins rotated images for its own reasons; muddies the
  isolated question.

### Reading the outcomes

- Ring wins held-out rotations + probe reads angle → structured-channel idea is
  real; frame as equivariance-on-a-budget.
- Matrix wins, ring ≤ matrix → extra mixing helped, topology didn't; a
  (smaller) factorization result.
- All ≈ MLP → structure earns nothing at this scale; the v1 magnitude-gate
  branch is a separate bet still standing.

## Context / prior-art map (from design review)

- Capsule nets (Sabour & Hinton 2017): magnitude×direction squash + per-
  connection D×D pose matrices + routing. Routing failed to scale; smallNORB
  viewpoint generalization was the real win. This design = the cheap rung of
  that ladder (matrix per *neuron*, not per connection; no routing).
- Vector Neurons (Deng et al., ICCV 2021): same name, different goal (SO(3)
  equivariance for point clouds); their nonlinearity is direction-clipping.
- MLP-Mixer: validates the mix-across-tokens / mix-across-channels
  factorization at scale.
- Group-equivariant CNNs / ring attractors: the principled relatives of the
  ring variant.
- v1's magnitude mechanism relatives: low-rank quadratic neurons, SwiGLU,
  squared ReLU (Primer), bilinear pooling — multiplicative interactions win
  when cheap and well-placed. Parked, not refuted.

## Open Questions

- Gate-then-matrix vs matrix-then-gate.
- D and K sweeps; does the ring need D ≥ ~8 to have room to shift?
- Initialization of b (how negative?), M (identity + noise? random?), input
  projections (unit norm?).
- How to encode the input angle onto the ring at the on-ramp — learned
  projection may not discover the circular structure on its own; may need (or
  want to compare against) a hand-wired circular embedding for the synthetic
  task.
- Does depth preserve the channel topology in the ring variant, or does it
  drift?
