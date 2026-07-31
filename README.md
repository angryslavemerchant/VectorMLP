# VectorMLP — channel-mixing vector neurons

An MLP whose neurons carry D-dimensional vectors instead of scalars. Each
layer: mix across neurons with scalar weights → per-channel ReLU gate → mix
across channels (per-neuron D×D matrix, low-rank factorization, or K-tap
circular "ring" convolution). Design rationale in
`channel_mixing_neuron_spec.md`, results so far in `experiment_results.md`.

## Layout

- `vector_mlp.py` — layers (`VectorLinear`, `VectorMLP`), plain-MLP baseline,
  parameter matchers
- `sanity_check.py` — shape/gradient/equivariance checks
- `angle_task.py` — experiment 1: synthetic rotation-generalization kill test
- `mnist_task.py` / `mnist_task2.py` / `mnist_task3.py` — MNIST
  sample-efficiency grids (round 1: variants vs matched MLPs; round 2:
  ring-kernel ladder + matrix D-sweep down; round 3: D-sweep up + low-rank).
  All (size × seed) models of an arm train as one stacked computation via
  `torch.func.vmap` — 20 models in lockstep per run.
- `mnist_results*.json` — raw per-seed accuracies (paired across rounds:
  identical seeded data subsets)

## Running

Requires PyTorch with CUDA and torchvision (MNIST downloads to `data/` on
first run).

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'   # needed on some Windows/conda setups
python sanity_check.py
python mnist_task.py               # then mnist_task2.py, mnist_task3.py
python angle_task.py
```

Each MNIST round trains 80–100 models and takes ~25–50 min on an RTX 2060.
