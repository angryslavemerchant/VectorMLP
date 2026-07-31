# Vector Neuron Variants A & B — spec + as-built notes

Architecture pivot away from the channel-mixing neuron (which keeps D
independent scalar gates and touches vector geometry only linearly): these
variants make the **nonlinearity itself operate on the vector geometry**.

Shared setup: vector dimension D is a hyperparameter (target ballpark D=2–4,
ablatable); raw inputs/outputs are scalars.

## Variant A — Projection (`ProjNet` / `ProjLinear`)

Every inter-neuron signal is a single D-vector; magnitude = activation
strength, direction = feature identity, coupled.

Per neuron: N scalar connection weights, scalar bias, per-neuron D×D
projection P.

    v = Σ wᵢxᵢ          (sum of scaled incoming vectors)
    r = P v
    y = ReLU(‖r‖ + b) · r / (‖r‖ + ε)      ← as-built (modReLU)

Mechanism: ‖Pv‖² = vᵀPᵀPv — each neuron's firing is a learned (PSD)
quadratic form of the summed vector, i.e. constructive/destructive
interference of input directions. Capsule-network-like, minus routing, with
per-neuron (not per-connection) matrices.

On-ramp: learned unit direction per input, vᵢ = xᵢdᵢ (full per-feature
wiring at D scalars per feature — cheap at D=2–4).
Off-ramp: linear readout on flattened final vectors (subsumes the per-class
direction dot of the original spec).

**Deviations from the original spec, and why:**
- `‖r‖` instead of `‖r‖²` in the gate: squared norms square magnitudes per
  layer (4th powers after two layers). Degree-1 keeps scale stable.
- ε in the direction division; norm computed as sqrt(‖r‖²+1e-12): the raw
  r/‖r‖ has an exploding gradient at the origin.
- With negative bias (init −0.1, same recipe as the channel-mixing gates)
  the unit outputs exactly 0 below the magnitude threshold.
- Note: one nonlinear scalar per neuron (the radial gate) vs D gates in the
  channel-mixing neuron — A bets its D-dim linear side-channel is worth the
  reduced nonlinear bandwidth.

## Variant B — Decoupled (`TagNet` / `TagLinear`)

Signals are (scalar activation, fixed learned D-dim identity tag). Scalar
path is a standard weighted sum z; the direction path computes an agreement
gain that multiplies it.

**The spec as originally written had a collapse**: uᵢ = aᵢvᵢ doesn't depend
on the receiving neuron, so in a dense layer g = ‖Σaᵢvᵢ‖² is ONE scalar
shared by the whole layer — a global gain knob, not an agreement mechanism.
Two fixes, both built:

    mode='weighted':  rₙ = Σ wₙᵢaᵢvᵢ   (connection weights REUSED in the
                      direction path — agreement over the inputs this neuron
                      cares about; zero extra params)      gₙ = ‖rₙ‖²
    mode='query':     r  = Σ aᵢvᵢ      (one shared field per layer)
                      gₙ = (qₙ·r)²     (per-neuron query direction)

    gain = g/(1+g)    (bounded — raw z·g grows cubically per layer and has
                      a gradient dead-zone at g=0)
    a_out = ReLU(z · gain + b)

Off-ramp: standard linear classifier on final scalars. Consequently the last
hidden layer's output tags would never be read — built tagless
(`out_tags=False`) so they don't pad the param count as dead weights.

## Where they're wired in

- `experiments/cifar_head_task.py 2` — frozen-DINO heads: proj-d2, proj-d4,
  tagw-d4, tagq-d4, all width-matched to the round-1 vec-head target
  (26,304). Matched widths: 54 / 48 / 54 / 53 (mlp baseline was 57).
- `experiments/cifar_e2e.py 2` — same SmallCNN trainable backbone,
  new-neuron heads matched to the round-1 head target (32,960). All match
  at width 11: they pay the full 2048-input wiring tax like cnn-mlp does
  (cnn-vec's reshape dodges it) — narrow, but the honest match.

Compare against the round-1 arms (same seeds/subsets → exact per-seed
pairing): does geometry-coupled nonlinearity do anything the channel-mixing
neuron and the plain MLP don't, on features (head task) and with a
co-adapting host (e2e)?
