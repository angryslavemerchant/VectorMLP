# Guide: designing a new neuron for VectorMLP

This repo benchmarks alternative "neuron" designs (vector-valued hidden
units instead of scalar ReLU units) against each other and against
param-matched plain-MLP baselines. This doc tells you how to write a new
neuron so it drops into the existing benchmark harness with no changes to
the harness itself.

Read `vector_mlp.py` in full before writing anything — `VectorMLP`/
`VectorLinear`, `ProjNet`/`ProjLinear`, and `TagNet`/`TagLinear` are three
working examples of the pattern below. Copy whichever is closest to your
idea and modify it.

## The pattern: two classes, `*Linear` + `*Net`

Every neuron family is two classes:

1. **`<Name>Linear`** (or you can call it `<Name>Layer`) — one hidden
   layer, `nn.Module`. Takes `n_in`, `n_out`, `dim` plus any
   family-specific kwargs. Maps a batch of `n_in` vectors to a batch of
   `n_out` vectors (or whatever your signal type is — see "signal shape"
   below).
2. **`<Name>Net`** — the full model. Stacks `<Name>Linear` layers,
   handles the "on-ramp" (raw scalar features -> first layer's signal
   type) and "off-ramp" (last hidden layer -> class logits).

The harness only ever instantiates `<Name>Net`. `<Name>Linear` exists
for code organization, testing, and so the net can vary `channel_mix`,
`mode`, etc. per layer if you want.

## Required `<Name>Net` interface

```python
class YourNet(nn.Module):
    def __init__(self, in_features, hidden, num_classes, dim, **kw):
        super().__init__()
        ...

    def forward(self, x):
        # x: [B, in_features]  (plain scalar features, e.g. flattened
        #    pixels or a frozen backbone's feature vector)
        ...
        return logits  # [B, num_classes]
```

Hard constraints:

- **Constructor signature is `(in_features, hidden, num_classes, dim, **kw)`**
  in that positional order. `hidden` is a `list[int]` of hidden widths
  (one entry per hidden layer — support arbitrary depth, don't hardcode
  2 layers). `dim` is the vector/channel width `D`, the one hyperparameter
  that's compared apples-to-apples across all neuron families in a given
  experiment. Anything specific to your design (mixing mode, rank,
  init flags, etc.) goes in `**kw` with a keyword default.
- **`forward(x)` takes `[B, in_features]` and returns `[B, num_classes]`.**
  Whatever exotic signal representation you use internally (vectors,
  scalar+tag pairs, complex numbers, whatever) is fully contained between
  the on-ramp and off-ramp — the outside world only ever sees flat
  scalars in and flat logits out. See `ProjNet.forward` / `TagNet.forward`
  for the on-ramp/off-ramp pattern.
- **Pure `nn.Module`, plain tensors only, no Python-level control flow
  that depends on tensor values, no CUDA graphs, no external state.**
  The training harness stacks N independently-initialized copies of your
  model with `torch.func.stack_module_state` and runs training as one
  `vmap(grad(loss_fn))` call — effectively your `forward` gets traced
  under `vmap`/`grad`/`functional_call`. Concretely this means:
    - Every learnable value must be an `nn.Parameter` (not a plain
      tensor, not a Python float you mutate).
    - No `.item()`, no `if some_tensor > 0:` branching on data, no
      in-place ops that dodge autograd, no non-tensor mutable state
      (e.g. a running Python list you append to across calls).
    - `nn.Sequential`/`nn.ModuleList` of standard layers, `einsum`,
      `F.relu`, `F.conv1d` etc. are all fine — that's exactly what the
      existing variants use.
    - If you need a buffer (non-trained tensor, e.g. a fixed
      permutation), use `register_buffer`, same as `VectorLinear`'s
      `tie_index`.
- **`sum(p.numel() for p in model.parameters() if p.requires_grad)`
  must equal your "real" parameter count.** The harness width-matches
  arms by this count (`count_params` / `matched_width` in
  `vector_mlp.py:334-402`), so don't carry non-trainable parameters with
  `requires_grad=True`, and don't stash extra state as fake parameters.

## Signal shape — pick one, be consistent

Pick whatever internal representation your idea needs between layers,
but keep it uniform across all your `<Name>Linear` layers so they chain:

- Pure vector, like `VectorLinear`/`ProjLinear`: `[B, N, D]` in, `[B, N, D]`
  out. Simplest — model composition is just `for layer in self.layers: v
  = layer(v)`.
- Paired signal, like `TagLinear`: activations `[B, N]` + a side-channel
  `[N, D]` "tag" tensor, layer returns both `(activation, tag_out)`, and
  the net threads a `(a, tags)` pair through the loop instead of a single
  tensor. This is fine as long as `<Name>Net.forward` still reduces to
  scalar `[B, num_classes]` logits at the end.

Whichever you pick, write it in a docstring at the top of your
`<Name>Linear` class the way the existing ones do — the math notation
(what `w`, `r`, `g` etc. mean) matters more here than usual since these
aren't standard layers.

## On-ramp / off-ramp conventions

- **On-ramp** (`in_features` scalars -> your signal type): the existing
  convention is `v_i = x_i * d_i` where `d_i` is a learned (or
  `proj_init`-seeded, optionally frozen via `freeze_proj`) per-feature
  direction of shape `[in_features, dim]`. See `ProjNet.__init__`
  (`self.dirs`) and `VectorMLP.hidden` (`self.proj_in`). You don't have
  to reuse this exact scheme, but keep on-ramp params to
  `O(in_features * dim)` — that's the assumed baseline both against
  which "vector richness" is measured and against which param-matching
  is computed.
- **Off-ramp** (last hidden layer -> `[B, num_classes]`): simplest and
  most common is a plain `nn.Linear` on the flattened/pooled last-layer
  signal (`ProjNet.head`, `TagNet.head`). `VectorMLP` instead supports
  a learned per-class direction dot-product (`readout='dirs'`) or a
  channel-mean pool (`readout='pooled'`) — use one of those patterns if
  your signal type doesn't flatten cleanly.

## Wiring it into an experiment

Once `YourNet`/`YourLinear` exist in `vector_mlp.py` (or a new module
imported alongside it), you don't touch the harness (`train_stack`,
`eval_stack`, `vmap`/`stack_module_state` machinery) at all. You only
add an entry to whichever experiment script's arm dict you want to run
it in — e.g. `experiments/cifar_head_task.py`:

```python
arms = {
    name: (new_arm(name, build), tx, ex, ex_rot)
    for name, build in {
        'proj-d4':   lambda w: ProjNet(FEAT, [w] * len(HIDDEN), 10, 4),
        'yourarm-d4': lambda w: YourNet(FEAT, [w] * len(HIDDEN), 10, 4),
    }.items()}
```

`new_arm(name, build)` auto width-matches your net's hidden width `w` to
hit the round's target parameter budget via `matched_width` — you don't
pick `w` by hand, the factory does a binary search over it, which is why
the arm dict entry is `lambda w: YourNet(...)` rather than a built
instance.

For a quick correctness check before running a full grid, mirror
`tests/sanity_check.py`: instantiate with tiny sizes, run one
`forward`, check output shape is `[B, num_classes]`, check
`loss.backward()` populates `.grad` on every parameter, and (if your
design claims any equivariance/invariance property) check that
property numerically the way the existing tests do for rotation/shift.

## Checklist

- [ ] `YourLinear(n_in, n_out, dim, **kw)` — one layer, documented signal
      shape in the docstring.
- [ ] `YourNet(in_features, hidden, num_classes, dim, **kw)` — full model,
      `hidden` is a list, arbitrary depth.
- [ ] `forward(x)`: `[B, in_features]` -> `[B, num_classes]`, no internal
      state leaks past the class boundary.
- [ ] All learnable state is `nn.Parameter`; no data-dependent control
      flow; no non-tensor mutable state — must survive
      `stack_module_state` + `vmap(grad(...))`.
- [ ] `count_params(model)` reports only the params you intend to be
      counted in width-matching.
- [ ] Sanity-checked forward/backward on a tiny instance before wiring
      into a full experiment round.
