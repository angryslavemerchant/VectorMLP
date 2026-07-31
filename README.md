# VectorMLP — channel-mixing vector neurons

An MLP whose neurons carry D-dimensional vectors instead of scalars. Each
layer: mix across neurons with scalar weights → per-channel ReLU gate → mix
across channels (per-neuron D×D matrix, low-rank factorization, or K-tap
circular "ring" convolution).

- Design rationale: `docs/channel_mixing_neuron_spec.md`
- Results and conclusions so far: `results/experiment_results.md`

## Layout

- `vector_mlp.py` — the library: `VectorLinear`, `VectorMLP`, plain-MLP
  baseline, parameter matchers
- `experiments/angle_task.py` — synthetic rotation-generalization kill test
- `experiments/mnist_grid.py` — MNIST sample-efficiency grids (rounds 1–4);
  all (size × seed) models of an arm train as one stacked `torch.func.vmap`
  computation, 20 models in lockstep
- `results/` — analysis doc + raw per-seed accuracies (paired across rounds:
  identical seeded data subsets)
- `tests/sanity_check.py` — shape/gradient/equivariance checks

## Running

Requires PyTorch with CUDA and torchvision (MNIST downloads to `data/` on
first run).

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'   # needed on some Windows/conda setups
python tests/sanity_check.py
python experiments/mnist_grid.py 4   # rounds 1-4
python experiments/angle_task.py
```

Each MNIST round trains 20 models per arm and takes ~10 min/arm on an
RTX 2060.
