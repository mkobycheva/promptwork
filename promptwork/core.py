"""Gaussian covariance updates in nats, and a slow test oracle."""
import numpy as np


def _validate_prior(dim, sigma0_sq):
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError('dim must be a positive integer')
    if not np.isfinite(sigma0_sq) or sigma0_sq <= 0:
        raise ValueError('sigma0_sq must be finite and positive')


def _active(z, w, dim):                                            #check dimensions, w>0, discard w=0
    z = np.asarray(z, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != dim or w.shape != (z.shape[0],):
        raise ValueError('Expected embeddings (m, d) and w (m,)')
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError('Embeddings and nonnegative w must be finite')
    active = w > 0
    return z[active], w[active]


class GaussianPosterior:
    def __init__(self, dim: int, sigma0_sq: float = 1.0):
        _validate_prior(dim, sigma0_sq)
        self.dim = dim
        self.Sigma = np.eye(dim, dtype=np.float64) * sigma0_sq

    def update(self, z: np.ndarray, alpha: np.ndarray) -> float:   # alpha = w / sigma_sq, the caller divides
        z, alpha = _active(z, alpha, self.dim)                     #validate shapes/finite/non-negative; drop w == 0
        if not len(alpha):
            return 0.0
        b = np.sqrt(alpha)[:, None] * z                            #multiply vectors by sqrt from their weights to use Woodbury formula 
        cross = self.Sigma @ b.T                                   # (d, m): new vectors seen through current uncertainty
        m = np.eye(len(alpha)) + b @ cross                         # (m, m): I + overlap table
        m = (m + m.T) * 0.5                                        #removing floating-point rounding errors
        sign, logdet = np.linalg.slogdet(m)                        #sign and log of module of matrix determinant
        if sign <= 0 or not np.isfinite(logdet):                   #sanity check: matrix must be positive definite
            raise np.linalg.LinAlgError('Woodbury matrix must have a finite positive determinant')
        if logdet < -1e-10:                                        #sanity check: logdet cannot be negative
            raise np.linalg.LinAlgError('Negative information gain indicates numerical instability')
        updated = self.Sigma - cross @ np.linalg.solve(m, cross.T) #Sigma update (not J, because finding inverses each time is costly);
                                                                   #updating using inverse of the smaller m (only new vectors)
        self.Sigma = (updated + updated.T) * 0.5                   #removing floating-point rounding errors
        return float(max(0.0, 0.5 * logdet))                       #returns information gain


class ReferenceGaussianPosterior:                                  #canonical implementation from paper, slow
    """Slow precision implementation for tests only; not used by scoring."""
    def __init__(self, dim: int, sigma0_sq: float = 1.0):
        _validate_prior(dim, sigma0_sq)
        self.dim = dim
        self.J = np.eye(dim, dtype=np.float64) / sigma0_sq

    @property
    def Sigma(self):
        return np.linalg.solve(self.J, np.eye(self.dim))          #find inverse of J

    def update(self, z, alpha) -> float:
        z, alpha = _active(z, alpha, self.dim)
        if not len(alpha):
            return 0.0
        updated = self.J + z.T @ (alpha[:, None] * z)             #update J (precision matrix)
        sign0, ld0 = np.linalg.slogdet(self.J)                    #sign and log of module of J determinant
        sign1, ld1 = np.linalg.slogdet(updated)                   #sign and log of module of updated J determinant
        if sign0 <= 0 or sign1 <= 0:                              #sanity check
            raise np.linalg.LinAlgError('Precision must have a positive determinant')
        self.J = updated                                          #reassigning J to updated J
        return float(0.5 * (ld1 - ld0))                           #return info gain IG=1/2(logdetJ_new - logdetJ_old)
