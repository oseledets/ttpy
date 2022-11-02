import numpy as np


def normalize_domain(base, domain, ndim=None) -> np.ndarray:
    """Normalize domain specification.

    >>> normalize_domain((-1, 1), None, 1)
    array([[-1,  1]])
    >>> normalize_domain((-1, 1), (-10, 10), 3)
    array([[-10,  10],
           [-10,  10],
           [-10,  10]])
    """
    if domain is None:
        domain = base
    domain = np.asarray(domain, np.float64)
    if domain.ndim == 1:
        domain = domain[None, :]
    if domain.shape[1] != 2:
        raise ValueError('Subdomain expected to be a pair of '
                         'the left and right bounds.')
    if (domain[:, 0] >= domain[:, 1]).all():
        raise ValueError('Left bound must be less than the right one.')
    if ndim is not None:
        domain = np.tile(domain, (ndim, 1))
    return domain


def normalize_cheb_domain(domain, ndim=None) -> np.ndarray:
    return normalize_domain((-1, 1), domain, ndim)


def normalize_spl_domain(domain, ndim=None) -> np.ndarray:
    return normalize_domain((0, 1), domain, ndim)
