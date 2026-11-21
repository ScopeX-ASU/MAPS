import pytest
import torch
import torch.nn.functional as F

from core.invdes.models.materialization.resolution import convert_resolution_tensor


def _legacy_subpixel_conversion(
    x: torch.Tensor,
    *,
    eps_r: float,
    eps_bg: float,
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Reference implementation copied from the v0.0.5 subpixel branch.

    This test-only oracle intentionally preserves the legacy arithmetic order
    so forward values and the autograd graph can be compared independently.
    Do not use this helper in production code.
    """
    x = eps_bg + (eps_r - eps_bg) * x
    x = F.adaptive_avg_pool2d(
        x[None, None],
        output_size=target_size,
    )[0, 0]
    return (x - eps_bg) / (eps_r - eps_bg)


def test_subpixel_downsampling_matches_permittivity_space_average():
    hr_density = torch.tensor(
        [
            [0.0, 1.0, 0.25, 0.75],
            [0.5, 0.5, 0.0, 1.0],
            [0.25, 0.75, 0.1, 0.9],
            [1.0, 0.0, 0.8, 0.2],
        ],
        dtype=torch.float64,
    )
    eps_bg = 2.25
    eps_r = 12.25

    lr_density = convert_resolution_tensor(
        hr_density,
        target_size=(2, 2),
        subpixel_smoothing=True,
        eps_r=eps_r,
        eps_bg=eps_bg,
    )

    hr_eps = eps_bg + (eps_r - eps_bg) * hr_density
    expected_eps = torch.tensor(
        [
            [hr_eps[:2, :2].mean(), hr_eps[:2, 2:].mean()],
            [hr_eps[2:, :2].mean(), hr_eps[2:, 2:].mean()],
        ],
        dtype=hr_eps.dtype,
    )
    expected_density = (expected_eps - eps_bg) / (eps_r - eps_bg)

    torch.testing.assert_close(lr_density, expected_density, rtol=0, atol=1e-12)


def test_subpixel_downsampling_preserves_gradient_path():
    hr_density = torch.tensor(
        [[0.1, 0.4, 0.7, 0.9], [0.2, 0.3, 0.8, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    lr_density = convert_resolution_tensor(
        hr_density,
        target_size=(1, 2),
        subpixel_smoothing=True,
        eps_r=12.25,
        eps_bg=2.25,
    )
    loss = (lr_density**2).sum()
    loss.backward()

    assert hr_density.grad is not None
    assert hr_density.grad.shape == hr_density.shape
    assert torch.isfinite(hr_density.grad).all()
    assert torch.count_nonzero(hr_density.grad).item() == hr_density.numel()


def test_subpixel_conversion_matches_legacy_forward_and_backward_graph():
    torch.manual_seed(17)
    x_legacy = torch.rand(9, 7, dtype=torch.float64, requires_grad=True)
    x_new = x_legacy.detach().clone().requires_grad_(True)

    legacy = _legacy_subpixel_conversion(
        x_legacy,
        eps_r=12.25,
        eps_bg=2.25,
        target_size=(4, 3),
    )
    new = convert_resolution_tensor(
        x_new,
        subpixel_smoothing=True,
        eps_r=12.25,
        eps_bg=2.25,
        target_size=(4, 3),
    )

    torch.testing.assert_close(new, legacy, rtol=0, atol=0)

    legacy_loss = (legacy**2).sum()
    new_loss = (new**2).sum()
    legacy_loss.backward()
    new_loss.backward()
    torch.testing.assert_close(x_new.grad, x_legacy.grad, rtol=0, atol=0)


def test_same_shape_is_identity():
    density = torch.rand(4, 5, dtype=torch.float64)
    output = convert_resolution_tensor(density, target_size=density.shape)
    assert output is density


def test_resolution_requires_target_information():
    with pytest.raises(ValueError, match="target_size|target_resolution"):
        convert_resolution_tensor(torch.ones(4, 4))


def test_subpixel_mode_requires_material_permittivities():
    with pytest.raises(ValueError, match="eps_r and eps_bg"):
        convert_resolution_tensor(
            torch.ones(4, 4),
            target_size=(2, 2),
            subpixel_smoothing=True,
        )


def test_subpixel_mode_rejects_equal_material_permittivities():
    with pytest.raises(ValueError, match="eps_r and eps_bg"):
        convert_resolution_tensor(
            torch.ones(4, 4),
            target_size=(2, 2),
            subpixel_smoothing=True,
            eps_r=3.0,
            eps_bg=3.0,
        )


def test_interpolation_path_matches_torch_reference():
    torch.manual_seed(23)
    x = torch.rand(3, 5, 7, dtype=torch.float64)
    output = convert_resolution_tensor(
        x,
        intplt_mode="bilinear",
        target_size=(4, 6),
    )
    expected = F.interpolate(
        x.unsqueeze(0),
        size=(4, 6),
        mode="bilinear",
    )[0]
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
