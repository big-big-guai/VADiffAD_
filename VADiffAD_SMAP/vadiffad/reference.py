"""Normal Path Bank: a single mean/covariance per variable (Eqs. 19-21)."""
import numpy as np
import torch


class NormalPathBank:
    def __init__(self, shrinkage=0.1, eps=1e-6):
        self.shrinkage = shrinkage
        self.eps = eps
        self.mean = None
        self.inverse = None
        self.counts = None

    def fit(self, signatures):
        """Fit [N, D, 2K-1] normal signatures, no clustering or test labels."""
        h = np.asarray(signatures, dtype=np.float64)
        if h.ndim != 3 or h.shape[0] < h.shape[2] + 2:
            raise ValueError('Insufficient normal signatures for covariance estimation')
        n, d, m = h.shape
        self.mean = h.mean(axis=0)
        self.inverse = np.empty((d, m, m), dtype=np.float64)
        eye = np.eye(m)
        for dimension in range(d):
            sigma = np.cov(h[:, dimension, :], rowvar=False)
            # Exact isotropic shrinkage from the manuscript: tr(Sigma)/m * I.
            sigma_tilde = ((1-self.shrinkage)*sigma +
                           self.shrinkage*(np.trace(sigma)/m)*eye + self.eps*eye)
            self.inverse[dimension] = np.linalg.inv(sigma_tilde)
        self.counts = [n] * d
        return self

    def components(self, magnitude, variation, signature, compute_device=None):
        """Return uncalibrated local (V, C, D), retaining every time/variable."""
        if torch.is_tensor(signature):
            m = torch.as_tensor(self.mean, device=signature.device, dtype=signature.dtype)
            inv = torch.as_tensor(self.inverse, device=signature.device, dtype=signature.dtype)
            residual = signature - m[None, None]
            distance = torch.einsum('bldm,dmn,bldn->bld', residual, inv, residual)
            return torch.stack((magnitude, variation, distance.clamp_min(0)), -1)
        residual = signature.astype(np.float64) - self.mean[None, None]
        distance = np.einsum('bldm,dmn,bldn->bld', residual, self.inverse, residual, optimize=True)
        return np.stack((magnitude, variation, np.maximum(distance, 0)), axis=-1)

    def save(self, path):
        np.savez_compressed(path, mean=self.mean, inverse=self.inverse,
                            counts=np.asarray(self.counts, dtype=int),
                            shrinkage=self.shrinkage)
