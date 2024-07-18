"""
This module contains functions to calculate distribution distances.
"""
import numpy as np

try:
    import jax.numpy as jnp
except ImportError:
    print("JAX is not installed. Using numpy instead, which may be slower.")
    import numpy as jnp
from scipy import linalg


def wasserstein_distance(x, y):
    """
    See: https://en.wikipedia.org/wiki/Wasserstein_metric
    """
    means = []
    np.random.seed(0)
    for _ in range(10):
        if x.shape[0] != y.shape[0]:
            # sample from the larger distribution
            if x.shape[0] > y.shape[0]:
                x = x[np.random.choice(x.shape[0], y.shape[0], replace=False)]
            else:
                y = y[np.random.choice(y.shape[0], x.shape[0], replace=False)]
        means.append(jnp.mean((jnp.sort(x) - jnp.sort(y)) ** 2) ** 0.5)
    return np.mean(means)


def frechet_distance(x, y, eps=1e-6):
    if isinstance(x, tuple):
        mu1, sigma1 = x
    else:
        mu1 = jnp.mean(x, axis=0)
        sigma1 = jnp.cov(x, rowvar=False)
    if isinstance(y, tuple):
        mu2, sigma2 = y
    else:
        mu2 = jnp.mean(y, axis=0)
        sigma2 = jnp.cov(y, rowvar=False)

    mu1 = jnp.atleast_1d(mu1)
    mu2 = jnp.atleast_1d(mu2)

    sigma1 = jnp.atleast_2d(sigma1)
    sigma2 = jnp.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2).astype(complex), disp=False)
    if not jnp.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = jnp.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset).astype(complex))

    # Numerical error might give slight imaginary component
    if jnp.iscomplexobj(covmean):
        if not jnp.allclose(jnp.diagonal(covmean).imag, 0, atol=1e-2):
            m = jnp.max(jnp.abs(covmean.imag))
            print(f"Warning: Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = jnp.trace(covmean)

    result = diff.dot(diff) + jnp.trace(sigma1) + jnp.trace(sigma2) - 2 * tr_covmean

    return result
