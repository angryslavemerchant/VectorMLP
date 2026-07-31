# Vector Neuron Experiment Spec

## Core Idea

Replace the standard scalar neuron with a vector neuron that carries directional information through the network. The squared magnitude of the vector sum introduces pairwise input interactions (NMDA-like supralinear amplification) without adding learnable parameters beyond what a standard neuron already has.

---

## Neuron Definitions

### Standard Neuron (Baseline)

- **Per connection:** 1 learned scalar weight wᵢ
- **Per neuron:** 1 learned scalar bias b
- **Forward pass:**
  1. Compute weighted sum: z = Σ wᵢxᵢ + b
  2. Apply activation: y = ReLU(z)
- **Output:** 1 scalar
- **Learnable parameters:** N + 1

### Vector Neuron (Proposed)

- **Per connection:** 1 learned scalar weight wᵢ
- **Per neuron:** 1 learned scalar bias b
- **Hyperparameter:** vector dimension D (not learned)
- **Forward pass:**
  1. Receive N incoming D-dimensional vectors: x₁, x₂, ..., xₙ
  2. Scale each by connection weight: wᵢxᵢ
  3. Sum scaled vectors: r = Σ wᵢxᵢ
  4. Compute squared magnitude: s = ‖r‖²
  5. Add bias: s = s + b
  6. Apply activation: a = ReLU(s)
  7. Compute output direction: d = r / ‖r‖ (unit vector)
  8. Output vector: y = a · d
- **Output:** D-dimensional vector
- **Learnable parameters:** N + 1 (same as baseline)

### Mathematical Relationship

The squared magnitude expands to:

    ‖Σ wᵢxᵢ‖² = Σᵢ Σⱼ wᵢwⱼ(xᵢ · xⱼ)

This is equivalent to a quadratic neuron y = xᵀAx where A = wwᵀ (rank-1 interaction matrix). The pairwise interaction structure comes from the vector geometry rather than from N² free parameters.

---

## Experiment Design

TBD — task, architecture, controlled comparisons, variables to sweep, and success criteria still to be figured out.

---

## Implementation Notes

### Input Projection (Scalar → Vector On-Ramp)

The network receives scalar pixel values but the hidden layers operate in D-dimensional vector space. Each input scalar xᵢ gets multiplied by a learned D-dimensional direction vector pᵢ to produce a D-dimensional vector:

    vᵢ = xᵢ · pᵢ    where pᵢ ∈ ℝᴰ is learned

Example: pixel value 0.7 × learned direction [0.3, -0.8] → vector [0.21, -0.56]

Cost: input_dim × D additional learnable parameters (3072 × D for CIFAR-10). These are learned end-to-end through normal backpropagation.

### Output Projection (Vector → Scalar Off-Ramp)

The final hidden layer outputs D-dimensional vectors, but the network needs 10 scalar class logits. Each class gets a learned D-dimensional direction vector cₖ. The logit for class k is the dot product between the output vector and the class direction:

    logit_k = output_vector · cₖ    where cₖ ∈ ℝᴰ is learned

Cost: num_classes × D additional learnable parameters (10 × D for CIFAR-10). Negligible.

### Total Parameter Budget

For a fair comparison, account for projection overhead:

- Standard MLP: input weights + hidden weights + output weights
- Vector neuron: input projection (3072 × D) + hidden weights (same count as MLP) + output projection (10 × D)

The input projection is the main overhead. At D=2, that's 6144 extra parameters. Adjust hidden layer width downward slightly to match total parameter count with the baseline if needed.

### Numerical Stability

The squared magnitude can grow large when many inputs agree, especially in early layers with high fan-in. Consider:

- Normalizing the resultant before squaring: r_norm = r / √N, then square
- Using ‖r‖ (magnitude, not squared) as a gentler alternative — still supralinear in agreement because magnitude of a sum of aligned unit vectors is N while magnitude of random vectors is √N
- Layer normalization on the vector magnitudes between layers

### Gradient Through the Squared Magnitude

The gradient of ‖r‖² with respect to wᵢ is 2(Σⱼ wⱼxⱼ) · xᵢ. This means gradient magnitude scales with the resultant magnitude, which could cause exploding gradients in deep networks. Monitor this. If problematic, switch to ‖r‖ (magnitude without squaring) which has better-behaved gradients.

### Framework

PyTorch. The vector neuron forward pass is straightforward tensor operations — no custom CUDA kernels needed.

The core layer is:

```
# Shapes:
# x: [batch, N_in, D]      — incoming vectors
# w: [N_in, N_out]          — scalar connection weights
# b: [N_out]                — scalar biases

# 1. Scale inputs by weights and sum
# weighted: [batch, N_in, N_out, D]
weighted = x.unsqueeze(2) * w.unsqueeze(0).unsqueeze(-1)
# resultant: [batch, N_out, D]
resultant = weighted.sum(dim=1)

# 2. Squared magnitude + bias + ReLU
magnitude_sq = (resultant ** 2).sum(dim=-1)  # [batch, N_out]
activated = F.relu(magnitude_sq + b)         # [batch, N_out]

# 3. Output vector = activated magnitude * unit direction
direction = F.normalize(resultant, dim=-1)   # [batch, N_out, D]
output = activated.unsqueeze(-1) * direction  # [batch, N_out, D]
```

Note: F.normalize will need a small epsilon for zero-length resultants.

---

## Open Questions

- Does unsquared magnitude (‖r‖ instead of ‖r‖²) work better in practice despite weaker supralinear amplification?
- How does this behave as depth increases — does the directional information survive many layers?
- Would adding the learned reference angle per neuron (selective filtering before summation) help at larger D?
- How does this compare at scale to an actual quadratic neuron with the full N² interaction matrix?
- Should input projection vectors be initialized with unit norm? Does initialization scheme matter?
- Could the input projection be shared (one learned projection matrix for all pixels) instead of per-pixel?
