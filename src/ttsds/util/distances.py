"""
This module contains functions to calculate distribution distances.

These distance metrics are used to measure the similarity between distributions
produced by benchmarks across different datasets.
"""

from typing import Tuple, Union

import numpy as np
from scipy import linalg


def wasserstein_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the 2-Wasserstein distance between two 1D distributions.

    This implementation uses the sorted samples approach for empirical distributions.
    To ensure stability, it runs multiple times with different random subsamples
    when the distributions have different sizes.

    Args:
        x: First distribution samples
        y: Second distribution samples

    Returns:
        float: The Wasserstein distance between the distributions

    Reference:
        https://en.wikipedia.org/wiki/Wasserstein_metric
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


def frechet_distance(
    x: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    y: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    eps: float = 1e-6,
) -> float:
    """
    Calculate the Fréchet distance between two multivariate Gaussian distributions.

    The Fréchet distance (also known as Wasserstein-2 distance) measures the
    similarity between two probability distributions over a feature space.

    Args:
        x: First distribution (either samples or tuple of (mean, covariance))
        y: Second distribution (either samples or tuple of (mean, covariance))
        eps: Small epsilon to add to covariance matrices in case of numerical instability

    Returns:
        float: The Fréchet distance between the distributions

    Raises:
        AssertionError: If distributions have incompatible dimensions
    """
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

    return float(result)


def frechet_distance_fast(
    x: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    y: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    eps: float = 1e-6,
) -> float:
    """
    Faster implementation of Fréchet distance using optimization techniques.

    This implementation includes:
    1. Early return for identical distributions
    2. More efficient matrix operations
    3. Direct calculation of terms

    Args:
        x: First distribution (either samples or tuple of (mean, covariance))
        y: Second distribution (either samples or tuple of (mean, covariance))
        eps: Small epsilon to add to covariance matrices in case of numerical instability

    Returns:
        float: The Fréchet distance between the distributions

    Raises:
        AssertionError: If distributions have incompatible dimensions
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

    return float(result)
