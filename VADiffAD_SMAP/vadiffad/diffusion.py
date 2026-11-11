"""Variable-dependent diffusion schedule with a mean-preserving noise budget."""
import numpy as np
import torch


def robust_difference_mad(train, boundaries=None, eps=1e-6):
    """Compute differences only within each independent normal SMAP entity."""
    segments = boundaries if boundaries is not None else [(0, len(train))]
    difference = np.concatenate([np.diff(train[start:end], axis=0)
                                 for start, end in segments], axis=0).astype(np.float64)
    return np.median(np.abs(difference - np.median(difference, axis=0)), axis=0)


def variable_coefficients(train, cfg, boundaries=None):
    """r_d, c_hat_d, c_d (paper Eqs. 7-10). Only normal training samples."""
    r = robust_difference_mad(train, boundaries, cfg.eps)
    rbar = np.median(r)
    unscaled = ((rbar + cfg.eps) / (r + cfg.eps)) ** cfg.rho
    low, high = 0.0, 1.0
    while np.clip(high * unscaled, cfg.cmin, cfg.cmax).mean() < 1:
        high *= 2
    for _ in range(80):
        mid = (low + high) / 2
        if np.clip(mid * unscaled, cfg.cmin, cfg.cmax).mean() < 1:
            low = mid
        else:
            high = mid
    c = np.clip(((low+high)/2)*unscaled, cfg.cmin, cfg.cmax).astype(np.float32)
    if not np.isclose(c.mean(), 1, atol=1e-5):
        raise RuntimeError('Mean-preserving coefficient projection failed')
    return c


class VariableAwareSchedule:
    def __init__(self, coefficients, cfg, device):
        self.cfg = cfg
        self.coeff = torch.as_tensor(coefficients, dtype=torch.float32, device=device)
        t_beta = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps,
                                device=device, dtype=torch.float64)[:, None]
        beta = t_beta * self.coeff.double()[None, :]
        if (beta <= 0).any() or (beta >= 1).any():
            raise ValueError('Invalid per-variable beta values')
        log_alpha = torch.log1p(-beta).cumsum(0)
        self.log_alpha = torch.cat((torch.zeros_like(log_alpha[:1]), log_alpha), 0).float()

    def alpha(self, t):
        t = t.clamp(0, self.cfg.timesteps)
        lower = t.long()
        upper = (lower+1).clamp(max=self.cfg.timesteps)
        w = (t-lower.float())[:, None]
        return torch.exp((1-w)*self.log_alpha[lower] + w*self.log_alpha[upper])

    def coordinates(self, t):
        alpha = self.alpha(t)
        return alpha, torch.sqrt(((1-alpha)/alpha).clamp_min(0.))

    def perturb(self, x0, t, epsilon):
        alpha = self.alpha(t)[:, None, :]
        return alpha.sqrt()*x0 + (1-alpha).sqrt()*epsilon
