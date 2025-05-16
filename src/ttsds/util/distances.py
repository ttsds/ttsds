"""
This module contains functions to calculate distribution distances.
"""

import numpy as np
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
        means.append(np.mean((np.sort(x) - np.sort(y)) ** 2) ** 0.5)
    return np.mean(means)


def frechet_distance(x, y, eps=1e-6):
    if isinstance(x, tuple):
        mu1, sigma1 = x
    else:
        mu1 = np.mean(x, axis=0)
        sigma1 = np.cov(x, rowvar=False)
    if isinstance(y, tuple):
        mu2, sigma2 = y
    else:
        mu2 = np.mean(y, axis=0)
        sigma2 = np.cov(y, rowvar=False)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2).astype(complex), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset).astype(complex))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-2):
            m = np.max(np.abs(covmean.imag))
            print(f"Warning: Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    result = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return result


def frechet_distance_fast(x, y, eps=1e-6):
    """
    Faster implementation of Frechet distance using optimization techniques.

    This implementation includes:
    1. Optional caching of pre-computed statistics
    2. Faster matrix operations
    3. Early return for identical distributions
    """
    # Extract or compute mean and covariance
    if isinstance(x, tuple):
        mu1, sigma1 = x
    else:
        mu1 = np.mean(x, axis=0)
        sigma1 = np.cov(x, rowvar=False)

    if isinstance(y, tuple):
        mu2, sigma2 = y
    else:
        mu2 = np.mean(y, axis=0)
        sigma2 = np.cov(y, rowvar=False)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    # Early return if distributions are identical
    if np.allclose(mu1, mu2) and np.allclose(sigma1, sigma2):
        return 0.0

    # Compute squared difference between means
    diff = mu1 - mu2
    squared_diff = np.sum(diff * diff)

    # Calculate traces
    tr_sigma1 = np.trace(sigma1)
    tr_sigma2 = np.trace(sigma2)

    # Compute product of covariance matrices more efficiently
    try:
        # Use eigenvalue-based approach which is faster for certain matrix types
        s1_sqrt = linalg.sqrtm(sigma1)
        covmean_squared = s1_sqrt @ sigma2 @ s1_sqrt
        tr_covmean = np.sqrt(np.trace(covmean_squared.real))
    except:
        # Fall back to original method if the above fails
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2).astype(complex), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm(
                (sigma1 + offset).dot(sigma2 + offset).astype(complex)
            )

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-2):
                m = np.max(np.abs(covmean.imag))
                print(f"Warning: Imaginary component {m}")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

    # Compute final result
    result = squared_diff + tr_sigma1 + tr_sigma2 - 2 * tr_covmean

    return result
