# MAPS CI Testing Policy

CI should enforce the testing contract without requiring every expensive electromagnetic simulation on every pull request.

## PR gates

Every substantive source PR should run, as applicable:

1. static checks and formatting/lint checks already adopted by the repository;
2. unit tests;
3. focused integration tests for changed boundaries;
4. gradient/adjoint tests when differentiable code changes;
5. documentation/link checks when docs or public APIs change.

Numerical, GPU, and slow suites should run in PR CI when the changed subsystem requires them or when the repository has an appropriately sized runner. Otherwise they belong in scheduled/release validation with explicit status reporting.

## Nightly validation

Nightly CI should exercise the broader numerical surface:

- full unit suite;
- integration suite;
- gradient/adjoint suite;
- representative FDFD and FDTD-X numerical regressions;
- GPU regressions when a GPU runner is available;
- selected long-running examples and optimization smoke tests.

Nightly failures should be triaged as engineering or infrastructure failures rather than silently ignored.

## Release validation

Before publishing a release:

- run the release numerical regression set;
- run representative CPU/GPU tests for supported backends;
- verify canonical example workflows;
- verify documentation and generated API surfaces;
- record the commit/configuration used for numerical reference generation;
- confirm that no release branch contains unvalidated local-only behavior.

## Canonical commands

The project should converge on a small set of commands that run locally and in CI. Exact environment tooling may evolve, but the commands and markers should remain conceptually identical between developer and CI environments.

Example target commands:

```bash
pytest -m unit
pytest -m "integration or gradient"
pytest -m numerical
pytest -m "gpu and numerical"
pytest -m slow
```

A command may be narrowed during development, but a PR description should state the focused tests that were run and why broader suites were or were not required.

## Markers

Use pytest markers consistently:

- `unit`
- `integration`
- `gradient`
- `numerical`
- `gpu`
- `slow`

Markers are execution metadata, not a substitute for explaining the scientific purpose of a test.

## Required versus advisory checks

Critical correctness checks should eventually be required for merging. During migration, a check may be advisory while the test inventory is still incomplete, but the repository must document which checks are advisory and why.

Do not make a check permanently advisory simply because it is inconvenient or slow. Expensive tests should instead be scheduled at an appropriate frequency or run conditionally based on changed subsystems.

## Failure policy

A CI failure must be investigated before merge unless it is a recognized infrastructure failure with documented evidence.

Agents must not:

- disable a workflow to unblock a PR;
- skip failing tests without a reason tied to environment capability;
- lower thresholds globally because one platform is noisy;
- update numerical references without following `docs/development/numerical-regression.md`;
- report a suite as passed when it was not executed.

## Future evolution

As test coverage grows, the repository should move toward protected required checks for unit/integration/gradient quality and targeted numerical gates based on changed subsystems. The CI structure should remain explicit enough that an agent can determine which checks are required before modifying scientific code.
