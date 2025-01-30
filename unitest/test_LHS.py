import numpy as np

def generate_lhs_samples(n_samples, n_dimensions, lower_bounds=None, upper_bounds=None, seed=None):
    from scipy.stats import qmc
    """
    Generate Latin Hypercube Samples (LHS) with specified dimensions and sample size.

    Parameters:
    - n_samples (int): Number of samples to generate.
    - n_dimensions (int): Number of dimensions for each sample.
    - lower_bounds (array-like, optional): Lower bounds for each dimension. Defaults to 0 for all dimensions.
    - upper_bounds (array-like, optional): Upper bounds for each dimension. Defaults to 1 for all dimensions.
    - seed (int or np.random.Generator, optional): Seed for reproducibility. Defaults to None.

    Returns:
    - samples (np.ndarray): Array of shape (n_samples, n_dimensions) containing the LHS samples.
    """
    # Initialize the LatinHypercube sampler
    sampler = qmc.LatinHypercube(d=n_dimensions, seed=seed)

    # Generate samples in the unit hypercube
    unit_samples = sampler.random(n=n_samples)

    # If bounds are not specified, default to [0, 1] for all dimensions
    if lower_bounds is None:
        lower_bounds = np.zeros(n_dimensions)
    if upper_bounds is None:
        upper_bounds = np.ones(n_dimensions)

    # Scale samples to the specified bounds
    samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)

    return samples

# Example usage
n_samples = 123  # Number of samples
n_dimensions = 2000  # Number of dimensions
lower_bounds = [-0.2] * n_dimensions  # Lower bounds for each dimension
upper_bounds = [0.2] * n_dimensions  # Upper bounds for each dimension
seed = 42  # Seed for reproducibility

lhs_samples = generate_lhs_samples(n_samples, n_dimensions, lower_bounds, upper_bounds, seed)
print(lhs_samples.shape)
print(lhs_samples)
