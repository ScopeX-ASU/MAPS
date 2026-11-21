# InvDes AutoTune, Circuit Simulation, and Example Surfaces

This page records the remaining top-level MAPS-InvDes surfaces that sit outside the core device/parametrization/objective loop.

## AutoTune

`core.invdes.autotune.AutoTune` wraps an Optuna study around a user-provided inverse-design evaluation callback.

The callback contract is conceptually:

```python
obj, invdes = eval_obj_fn(iteration, params)
```

where `params` is the sampled parameter dictionary, `obj` is the scalar value reported to Optuna, and `invdes` is the corresponding inverse-design object when the caller wants to preserve the best model/artifacts.

### Parameter schema

`params_cfgs` is keyed by parameter name. Each entry contains a `type` field plus the arguments used to construct the corresponding Optuna distribution. The implementation recognizes `float`, `int`, and `categorical` distributions.

A representative continuous parameter is:

```python
params_cfgs = {
    "design_region_size": {
        "type": "float",
        "low": 3,
        "high": 7,
        "step": 1,
        "log": False,
    }
}
```

### Trial lifecycle

The search loop asks the study for a trial, avoids repeated parameter tuples when possible, evaluates uncached trials through `eval_obj_fn`, reports the value with `study.tell`, and logs the current best/top-k trials.

The duplicate cache is keyed by `tuple(trial.params.values())`. Consequently, parameter ordering in the Optuna trial representation is part of the current cache implementation.

Initial guesses can be queued with `add_init_guesses`; every supplied key must already exist in the configured distributions.

## Circuit-level simulation

`core/invdes/circuit_sim` contains SAX/JAX-oriented circuit utilities that consume component S-parameters rather than electromagnetic field tensors.

### Component S-matrix schema

`Component` stores an S-matrix dictionary keyed by port pairs:

```python
{
    ("in0", "out0"): 0.5,
    ("in0", "out1"): 0.5,
}
```

For a wavelength-vectorized model, each value can instead be a JAX array with one complex/amplitude value per predefined wavelength:

```python
{
    ("in0", "out0"): jnp.array([...]),
    ("in0", "out1"): jnp.array([...]),
}
```

Calling a `Component` applies `sax.reciprocal` to the stored dictionary, so reciprocity/symmetric reverse paths are part of this wrapper's circuit contract.

### Netlist schema

The splitter helpers and schematic renderer operate on SAX-style netlists with three principal mappings:

- `instances`: instance name to component/model;
- `connections`: endpoint string such as `"s11,out0"` to another endpoint;
- `ports`: top-level port name to an instance endpoint.

`draw_schematic` infers per-instance port tables from these mappings and renders a directed Graphviz diagram. Port names beginning with `in`/`out` are used as a presentation heuristic, not as an electromagnetic convention.

## Example directory policy

`core/invdes/examples/` contains research scripts spanning 2-D and 3-D devices, material platforms, FDFD/FDTD-X paths, and specialized demonstrations. These scripts are valuable executable specifications but are not all suitable for generic GitHub-hosted CI because many require GPU memory, optional solvers, or long electromagnetic simulations.

The documentation policy is therefore:

1. keep representative lightweight/validated examples linked from getting-started documentation;
2. preserve specialized examples as research references;
3. add explicit configuration/shape/unit comments when an example introduces a new public workflow;
4. run expensive examples on dedicated GPU/release validation rather than on every documentation PR;
5. treat example breakage caused by an API change as documentation drift, because examples are part of the user-facing contract.

The canonical introductory paths remain the Bending workflows documented under `docs/getting-started/`, while the larger example directory provides device-specific extensions.
