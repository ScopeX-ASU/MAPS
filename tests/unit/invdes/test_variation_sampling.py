"""Sampling contracts independent of geometry and electromagnetic solvers."""

import numpy as np
import pytest

from core.invdes.variation import VariationSampler

pytestmark = pytest.mark.unit


def config(**kwargs):
    """Return deterministic scalar-distribution settings for sampler tests."""
    return dict(
        enabled=True,
        seed=42,
        num_samples=4,
        distributions=[dict(params=["arbitrary.dx"], type="norm", loc=0, scale=0.01)],
        **kwargs,
    )


def test_seed_cache_and_exact_resume():
    a, b = VariationSampler(config()), VariationSampler(config())
    assert a.get(0) == b.get(0)
    assert a.get(0) == a.get(0)
    state = a.state_dict()
    expected = a.get(1)
    b.load_state_dict(state)
    assert b.get(1) == expected
    assert expected != b.get(2)
    b.set_seed(43)
    assert b.get(0) != VariationSampler(config()).get(0)


def test_disable_entropy_freeze_intervals_and_corners():
    sampler = VariationSampler(dict(enabled=False, seed=42))
    before = sampler.state_dict()["rng"]
    assert sampler.get(10) == {"scenarios": [{}], "weights": [1.0]}
    assert sampler.state_dict()["rng"] == before
    sampler = VariationSampler(config(resample_every=10))
    assert sampler.get(0) == sampler.get(9)
    assert sampler.get(10) != sampler.get(9)
    sampler.freeze()
    assert sampler.get(10) == sampler.get(100)
    sampler.unfreeze()
    assert sampler.get(200) != sampler.get(100)
    seed = sampler.randomize_seed()
    assert isinstance(seed, int) and sampler.seed == seed
    assert VariationSampler({**sampler.config, "seed": seed}).get(0) == sampler.get(0)
    corners = VariationSampler(
        dict(enabled=True, scenarios=[{"x": -1}, {"x": 1}], weights=[1, 3])
    )
    assert corners.get(0) == corners.get(20)
    assert corners.get(0)["weights"] == [0.25, 0.75]


@pytest.mark.parametrize(
    "strategy", ["iid", "fixed", "sobol", "halton", "latin_hypercube", "antithetic"]
)
def test_native_strategy_reproducibility(strategy):
    settings = config(strategy=strategy)
    assert VariationSampler(settings).get(0) == VariationSampler(settings).get(0)
    if strategy == "fixed":
        sampler = VariationSampler(settings)
        assert sampler.get(0) == sampler.get(99)
    if strategy == "antithetic":
        batch = VariationSampler(settings).sample()["scenarios"]
        assert batch[0]["arbitrary.dx"] == pytest.approx(-batch[2]["arbitrary.dx"])


def test_independent_uniform_joint_and_mixture():
    distributions = [
        dict(
            params=["x", "y"],
            type="multivariate_normal",
            mean=[0, 0],
            cov=[[1, 0.8], [0.8, 1]],
        ),
        dict(params=["u"], type="uniform", loc=-2, scale=4),
        dict(
            params=["m"],
            type="mixture",
            weights=[1, 0],
            components=[
                dict(type="uniform", loc=3, scale=0.1),
                dict(type="norm", loc=-20, scale=1),
            ],
        ),
    ]
    sampler = VariationSampler(
        dict(enabled=True, seed=4, num_samples=4096, distributions=distributions)
    )
    batch = sampler.sample()["scenarios"]
    assert all(-2 <= s["u"] <= 2 and 3 <= s["m"] <= 3.1 for s in batch)
    assert np.corrcoef([[s["x"] for s in batch], [s["y"] for s in batch]])[
        0, 1
    ] == pytest.approx(0.8, abs=0.03)


def test_custom_numpy_torch_adapters_and_global_rng_isolation():
    import torch

    np.random.seed(55)
    numpy_state = np.random.get_state()
    torch.manual_seed(55)
    torch_state = torch.random.get_rng_state().clone()
    settings = dict(
        enabled=True,
        seed=7,
        num_samples=3,
        distributions=[
            dict(
                params=["a"],
                backend="custom",
                distribution=lambda rng, n: rng.normal(size=(n, 1)),
            ),
            dict(params=["b"], backend="numpy", type="uniform", low=-1, high=1),
            dict(
                params=["c"],
                backend="torch",
                distribution=torch.distributions.Normal(0.0, 1.0),
            ),
        ],
    )
    assert VariationSampler(settings).sample() == VariationSampler(settings).sample()
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    custom = VariationSampler(
        dict(
            enabled=True,
            strategy=lambda rng, n: [{"x": float(rng.normal())} for _ in range(n)],
            seed=2,
        )
    )
    assert len(custom.sample()["scenarios"]) == 1


@pytest.mark.parametrize(
    "settings",
    [
        dict(num_samples=0),
        dict(distributions=[dict(params=["x", "x"], type="norm")]),
        dict(weights=[-1]),
        dict(
            distributions=[
                dict(params=["x", "y"], type="multivariate_normal", mean=[0], cov=[[1]])
            ]
        ),
        dict(
            strategy="sobol",
            distributions=[
                dict(
                    params=["x", "y"],
                    type="multivariate_normal",
                    mean=[0, 0],
                    cov=[[1, 0], [0, 1]],
                )
            ],
        ),
    ],
)
def test_invalid_configuration_fails_clearly(settings):
    with pytest.raises(ValueError):
        VariationSampler({**config(), **settings}).sample()


@pytest.mark.parametrize(
    "strategy", ["iid", "sobol", "halton", "latin_hypercube", "antithetic"]
)
def test_checkpoint_is_safe_serializable_and_restores_each_strategy(strategy):
    import io

    import torch

    sampler = VariationSampler(config(strategy=strategy))
    sampler.get(0)
    stream = io.BytesIO()
    torch.save(sampler.state_dict(), stream)
    stream.seek(0)
    restored = VariationSampler(config(strategy=strategy))
    restored.load_state_dict(torch.load(stream, weights_only=True))
    assert restored.get(1) == sampler.get(1)


def test_explicit_resample_is_used_at_the_next_logical_step():
    sampler = VariationSampler(config())
    batch = sampler.resample()
    assert sampler.get(0) == batch
    sampler.freeze()
    previous = sampler.get(0)
    batch = sampler.resample()
    assert batch != previous and sampler.get(5) == batch


@pytest.mark.gpu
def test_cpu_distribution_adapter_preserves_cuda_random_state(require_cuda):
    import torch

    before = torch.cuda.get_rng_state(0).clone()
    sampler = VariationSampler(
        dict(
            enabled=True,
            seed=42,
            distributions=[
                dict(
                    params=["p"],
                    backend="torch",
                    distribution=torch.distributions.Normal(0.0, 1.0),
                )
            ],
        )
    )
    sampler.sample()
    assert torch.equal(torch.cuda.get_rng_state(0), before)


def test_numpy_joint_covariance_is_validated():
    settings = dict(
        enabled=True,
        seed=42,
        distributions=[
            dict(
                params=["x", "y"],
                backend="numpy",
                type="multivariate_normal",
                mean=[0.0, 0.0],
                cov=[[1.0, 2.0], [2.0, 1.0]],
            )
        ],
    )
    with pytest.raises(ValueError, match="covariance"):
        VariationSampler(settings).sample()


def test_native_frozen_scipy_joint_distribution_adapter():
    from scipy import stats

    config = dict(
        enabled=True,
        seed=42,
        num_samples=4,
        distributions=[
            dict(
                params=["x", "y"],
                backend="scipy",
                distribution=stats.multivariate_normal(
                    mean=[0.0, 0.0], cov=[[1.0, 0.2], [0.2, 1.0]]
                ),
            )
        ],
    )
    a, b = VariationSampler(config), VariationSampler(config)
    assert a.sample() == b.sample()
    batch = a.get(0)
    assert len(batch["scenarios"]) == 4
    checkpoint = a.state_dict()
    b.load_state_dict(checkpoint)
    assert a.get(1) == b.get(1)


@pytest.mark.parametrize(
    "backend,strategy",
    [
        ("scipy", "iid"),
        ("numpy", "iid"),
        ("scipy", "sobol"),
        ("scipy", "halton"),
        ("scipy", "latin_hypercube"),
        ("scipy", "antithetic"),
    ],
)
def test_uniform_min_max_preserves_native_samples_and_resume(backend, strategy):
    native_bounds = (
        dict(loc=-0.01, scale=0.02)
        if backend == "scipy"
        else dict(low=-0.01, high=0.01)
    )
    common = dict(enabled=True, seed=42, num_samples=4, strategy=strategy)
    native = VariationSampler(
        {
            **common,
            "distributions": [
                dict(
                    params=["etch_delta"],
                    backend=backend,
                    type="uniform",
                    **native_bounds,
                )
            ],
        }
    )
    intuitive_config = {
        **common,
        "distributions": [
            dict(
                params=["etch_delta"],
                backend=backend,
                type="uniform",
                min=-0.01,
                max=0.01,
            )
        ],
    }
    intuitive = VariationSampler(intuitive_config)
    assert intuitive.get(0) == native.get(0)
    assert all(-0.01 <= s["etch_delta"] <= 0.01 for s in intuitive.get(0)["scenarios"])
    restored = VariationSampler(intuitive_config)
    restored.load_state_dict(intuitive.state_dict())
    assert restored.get(1) == intuitive.get(1) == native.get(1)


@pytest.mark.parametrize("backend", ["scipy", "numpy"])
def test_uniform_per_parameter_min_max(backend):
    sampler = VariationSampler(
        dict(
            enabled=True,
            seed=42,
            num_samples=16,
            distributions=[
                dict(
                    params=["x", "y"],
                    type="uniform",
                    backend=backend,
                    min=[-1.0, 2.0],
                    max=[0.0, 3.0],
                )
            ],
        )
    )
    assert all(
        -1 <= s["x"] <= 0 and 2 <= s["y"] <= 3 for s in sampler.sample()["scenarios"]
    )


@pytest.mark.parametrize(
    "bounds",
    [
        dict(min=0.0),
        dict(max=1.0),
        dict(min=1.0, max=0.0),
        dict(min=1.0, max=1.0),
        dict(min=float("nan"), max=1.0),
        dict(min=0.0, max=float("inf")),
        dict(min=0.0, max=1.0, loc=0.0),
        dict(min=0.0, max=1.0, low=0.0),
        dict(min=[0.0, 1.0], max=2.0),
    ],
)
def test_uniform_min_max_invalid_bounds_are_rejected(bounds):
    sampler = VariationSampler(
        dict(enabled=True, distributions=[dict(params=["x"], type="uniform", **bounds)])
    )
    with pytest.raises(ValueError, match="[Uu]niform"):
        sampler.sample()


def test_uniform_min_max_in_mixture_components():
    sampler = VariationSampler(
        dict(
            enabled=True,
            seed=42,
            num_samples=16,
            distributions=[
                dict(
                    params=["x"],
                    type="mixture",
                    weights=[1.0],
                    components=[dict(type="uniform", min=2.0, max=3.0)],
                )
            ],
        )
    )
    assert all(2 <= s["x"] <= 3 for s in sampler.sample()["scenarios"])


@pytest.mark.parametrize("backend,kind", [("scipy", "norm"), ("numpy", "normal")])
@pytest.mark.parametrize(
    "strategy", ["iid", "sobol", "halton", "latin_hypercube", "antithetic"]
)
def test_truncated_gaussian_matches_conditioned_inverse_cdf_and_resume(
    backend, kind, strategy
):
    """Truncation follows the conditional law without endpoint point masses."""
    from scipy import stats

    spec = dict(
        params=["x", "y"],
        backend=backend,
        type=kind,
        loc=[0.0, 1.0],
        scale=[1.0, 2.0],
        min=[-0.25, -1.0],
        max=[0.75, 2.0],
    )
    settings = dict(
        enabled=True, seed=17, num_samples=8, strategy=strategy, distributions=[spec]
    )
    sampler = VariationSampler(settings)
    if strategy == "iid":
        expected = stats.truncnorm(
            [-0.25, -1.0], [0.75, 0.5], loc=[0.0, 1.0], scale=[1.0, 2.0]
        ).rvs(size=(8, 2), random_state=np.random.default_rng(17))
        actual = np.array([[s["x"], s["y"]] for s in sampler.get(0)["scenarios"]])
        np.testing.assert_array_equal(actual, expected)
    assert all(
        -0.25 < s["x"] < 0.75 and -1.0 < s["y"] < 2.0
        for s in sampler.get(0)["scenarios"]
    )
    restored = VariationSampler(settings)
    restored.load_state_dict(sampler.state_dict())
    assert restored.get(1) == sampler.get(1)
    assert sampler.support_bounds() == {"x": (-0.25, 0.75), "y": (-1.0, 2.0)}


def test_truncated_joint_gaussian_preserves_conditional_correlations():
    """Joint rejection retains correlations and places no mass at bounds."""
    sampler = VariationSampler(
        dict(
            enabled=True,
            seed=42,
            num_samples=4096,
            distributions=[
                dict(
                    params=["x", "y"],
                    type="multivariate_normal",
                    mean=[0, 0],
                    cov=[[1, 0.8], [0.8, 1]],
                    min=-3.0,
                    max=3.0,
                )
            ],
        )
    )
    values = np.array([[s["x"], s["y"]] for s in sampler.sample()["scenarios"]])
    assert np.all((values > -3) & (values < 3))
    assert np.corrcoef(values.T)[0, 1] == pytest.approx(0.8, abs=0.03)
    assert sampler.support_bounds() == {"x": (-3.0, 3.0), "y": (-3.0, 3.0)}


def test_truncated_gaussian_has_conditional_mean_without_clipping():
    """Half-normal mean is analytic; clipping would instead create a zero atom."""
    sampler = VariationSampler(
        dict(
            enabled=True,
            seed=42,
            num_samples=32768,
            distributions=[
                dict(params=["x"], type="norm", loc=0.0, scale=1.0, min=0.0, max=8.0)
            ],
        )
    )
    values = np.array([s["x"] for s in sampler.sample()["scenarios"]])
    assert (values > 0).all()
    assert values.mean() == pytest.approx(np.sqrt(2 / np.pi), abs=0.01)


def test_support_bounds_uses_theoretical_support_and_missing_corner_zeros():
    sampler = VariationSampler(
        dict(
            enabled=True,
            seed=42,
            distributions=[
                dict(params=["x"], type="uniform", min=-1.0, max=1.0),
                dict(params=["y"], type="norm", loc=0.0, scale=1.0),
                dict(params=["z"], type="truncnorm", a=-2, b=3, loc=1.0, scale=0.5),
            ],
        )
    )
    before = sampler.state_dict()["rng"]
    assert sampler.support_bounds() == {
        "x": (-1.0, 1.0),
        "y": (-np.inf, np.inf),
        "z": (0.0, 2.5),
    }
    assert sampler.state_dict()["rng"] == before
    assert VariationSampler(
        dict(enabled=True, scenarios=[{"x": 1.0}, {"y": -2.0}])
    ).support_bounds() == {"x": (0.0, 1.0), "y": (-2.0, 0.0)}


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan")])
def test_truncated_gaussian_invalid_scale_rejected(scale):
    sampler = VariationSampler(
        dict(
            enabled=True,
            distributions=[dict(params=["x"], type="norm", scale=scale, min=-1, max=1)],
        )
    )
    with pytest.raises(ValueError, match="scale"):
        sampler.sample()
