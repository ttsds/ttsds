"""
This module contains functions to calculate distribution distances.
"""
import numpy as np

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
