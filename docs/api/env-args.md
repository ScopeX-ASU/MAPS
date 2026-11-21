# FDTD environment arguments

All MAPS FDTD environment switches are registered in
[`core/env_args.py`](../../core/env_args.py). Each entry defines its default,
allowed values, and explanation. Solver and device modules use this parser;
there is no new `sim_cfg` option or simulation interface.

List the complete catalog without importing simulation dependencies:

```bash
python -m core.env_args
```

Set environment variables **before importing** solver or device modules:

```bash
MAPS_ENABLE_BROADBAND_ADJOINT=1 \
MAPS_BROADBAND_ADJOINT_METHOD=chebyshev \
MAPS_BROADBAND_ADJOINT_PROFILE_ORDER=3 \
CUDA_VISIBLE_DEVICES=0 python your_script.py
```

Boolean switches accept `0` or `1`, not `true`/`false`. Invalid values now raise
`ValueError` naming the variable and its allowed domain instead of silently
disabling a feature. Numeric guards reject nonfinite or out-of-range values.
The adjoint method remains case-insensitive. Defaults and valid existing values
are unchanged.

The existing import-time behavior is preserved: changing `os.environ` after
import does not update already-imported solver/device settings. The parser itself
does not cache, so reloading a consumer reads the current environment as before.

One historical context-dependent default is preserved explicitly:
`MAPS_ENABLE_PHASOR_H` defaults to `1` in the backend but `0` in device monitor
setup. Setting it explicitly controls both. This consolidation does not change
which field components existing workflows record.

CUDA/JAX/library environment variables remain owned by those libraries; they are
not parsed here. Forward mode-source selection remains in the existing
`sim_cfg["fdtdx_mode_source_method"]` interface.

For broadband-adjoint behavior and numerical safeguards, see
[the design report](../design/broadband-adjoint.md).
