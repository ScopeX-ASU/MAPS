# Numerical Conventions

This document records cross-cutting conventions that should remain stable unless a change explicitly updates the convention, tests, and documentation.

## Current rule

Do not infer a convention from a single function. Confirm it from the relevant subsystem, tests, examples, and scientific notes before changing code.

## Conventions to document and maintain

The project should progressively record the following for each numerical subsystem:

- physical units and normalization;
- coordinate orientation and origin;
- grid spacing and indexing;
- frequency, wavelength, and angular-frequency relationships;
- complex-valued quantity conventions;
- boundary-condition conventions;
- field/component ordering;
- material/property representation;
- random seed behavior;
- objective and loss normalization;
- differentiation and gradient semantics;
- CPU/GPU precision expectations.

## Current documentation policy

Where the exact convention has not yet been verified and consolidated, documentation must say so rather than making a guess. The implementation and regression tests remain authoritative until the convention is explicitly captured here.

Subsystem-specific details should be added below dedicated headings as they are audited.

## FDFD

FDFD conventions require a dedicated audit covering grid ordering, field components, operator signs, material units, source normalization, and boundary conditions. Do not fill these details from generic FDFD assumptions; derive them from MAPS implementation and tests.

## FDTD / FDTDX

FDTD/FDTDX conventions require a dedicated audit covering time stepping, field component ordering, spatial indexing, material properties, source definitions, and normalization. Treat `v0.0.2_fdtdx` as the current release baseline for this documentation effort.
