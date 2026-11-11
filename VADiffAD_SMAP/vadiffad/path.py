"""Deterministic coarse probing and localized noise-prediction response paths.

This uses epsilon_theta, NOT the reconstructed-clean residual, at each path node.
The numerical discretization follows the manuscript's finite path update.
"""
import torch


def temporal_retention(original, perturbed, eps=1e-6):
    """Mean signed centered cosine across variables (manuscript Eq. 13)."""
    a = original - original.mean(dim=1, keepdim=True)
    b = perturbed - perturbed.mean(dim=1, keepdim=True)
    numerator = (a * b).sum(dim=1)
    denominator = a.square().sum(dim=1).sqrt() * b.square().sum(dim=1).sqrt() + eps
    return (numerator / denominator).mean(dim=-1)


@torch.no_grad()
def probe_depth(model, schedule, x, cfg):
    """Earliest probe with R <= tau_R and |R_p-R_{p-1}| <= tau_Delta."""
    b = len(x)
    grid = torch.linspace(0, cfg.timesteps, cfg.probe_nodes+1, device=x.device)
    state = x.clone()  # y_0 = X_0 / sqrt(alpha_0) = X_0
    observed = [temporal_retention(x, x, cfg.eps)]
    for i in range(cfg.probe_nodes):
        t = grid[i].expand(b)
        tn = grid[i+1].expand(b)
        a, gamma = schedule.coordinates(t)
        an, gamman = schedule.coordinates(tn)
        response = model(state*a[:, None, :].sqrt(), t, schedule.coeff)
        state = state + (gamman-gamma)[:, None, :] * response
        observed.append(temporal_retention(x, state*an[:, None, :].sqrt(), cfg.eps))
    retention = torch.stack(observed, dim=1)
    delta = (retention[:, 1:] - retention[:, :-1]).abs()
    eligible = (retention[:, 1:] <= cfg.retention_threshold) & (delta <= cfg.retention_delta_threshold)
    first_index = eligible.long().argmax(dim=1) + 1
    hit = eligible.any(dim=1)
    depth = torch.where(hit, grid[first_index], grid[-1])
    return depth, retention, hit


def local_signature(responses, delta_gamma, eps=1e-6):
    """Eqs. (17-19): weighted energy, response change, and signed 2K-1 signature.

    responses: [B,L,D,K]; delta_gamma: [B,D,K].
    """
    if responses.shape[-1] < 2:
        raise ValueError('At least two path responses are required')
    interval_weight = delta_gamma / delta_gamma.sum(dim=-1, keepdim=True).clamp_min(eps)
    magnitude = (responses.square()*interval_weight[:, None]).sum(dim=-1)
    change = (responses[..., 1:]-responses[..., :-1]) / (delta_gamma[:, None, :, :-1]+eps)
    change_weight = delta_gamma[..., :-1] / delta_gamma[..., :-1].sum(dim=-1, keepdim=True).clamp_min(eps)
    variation = (change.square()*change_weight[:, None]).sum(dim=-1)
    signature = torch.cat((responses, change), dim=-1)
    return magnitude, variation, signature


class DiffusionPathExtractor:
    def __init__(self, predictor, schedule, cfg):
        self.predictor, self.schedule, self.cfg = predictor, schedule, cfg

    @torch.no_grad()
    def __call__(self, x):
        b = len(x)
        depth, retention, hit = probe_depth(self.predictor, self.schedule, x, self.cfg)
        kmax = self.cfg.path_intervals
        state = x.clone()
        responses = []
        increments = []
        for k in range(kmax):
            t = depth * k / kmax
            tn = depth * (k+1) / kmax
            alpha, gamma = self.schedule.coordinates(t)
            _next_alpha, next_gamma = self.schedule.coordinates(tn)
            # Corresponds exactly to E_k = epsilon_theta(X_{t_k}, t_k, c).
            e = self.predictor(state * alpha[:, None, :].sqrt(), t, self.schedule.coeff)
            delta = (next_gamma-gamma).clamp_min(0)
            responses.append(e)
            increments.append(delta)
            state = state + delta[:, None, :] * e
        v, c, h = local_signature(torch.stack(responses, -1), torch.stack(increments, -1), self.cfg.eps)
        diagnostics = {'depth': depth.cpu().numpy(), 'stop_found': hit.cpu().numpy(),
                       'retention': retention.cpu().numpy()}
        return v, c, h, diagnostics
