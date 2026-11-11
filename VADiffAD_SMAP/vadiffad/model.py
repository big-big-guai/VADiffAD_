"""Shared temporal CNN noise field and auxiliary normal-context reconstruction."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TemporalResidualBlock(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.proj = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, h, embedding):
        update = self.proj(F.silu(self.conv(F.silu(self.norm(h) + embedding[..., None]))))
        return (h + update) / math.sqrt(2)


class TemporalNoisePredictor(nn.Module):
    """epsilon_theta(X_t, t, c), shared across variables (paper Eq. 12)."""
    def __init__(self, dimensions, hidden=64, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        self.encoder = nn.Conv1d(dimensions * 2, hidden, kernel_size=1)
        self.time_embedding = nn.Sequential(nn.Linear(32, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList(TemporalResidualBlock(hidden, d) for d in (1, 2, 4, 8))
        self.decoder = nn.Conv1d(hidden, dimensions, kernel_size=1)

    def forward(self, state, t, coeff):
        freqs = torch.exp(torch.linspace(0, math.log(1000), 16, device=state.device))
        theta = t[:, None] / self.timesteps * freqs[None]
        time_emb = self.time_embedding(torch.cat((theta.sin(), theta.cos()), dim=-1))
        condition = coeff.log()[None, None, :].expand(state.shape[0], state.shape[1], -1)
        h = self.encoder(torch.cat((state, condition), dim=-1).transpose(1, 2))
        for block in self.blocks:
            h = block(h, time_emb)
        return self.decoder(h).transpose(1, 2)


class MaskedContextReconstructor(nn.Module):
    """Trainable residual scorer; distinct from the diffusion noise predictor."""
    def __init__(self, dimensions, hidden=64):
        super().__init__()
        self.encoder = nn.Conv1d(dimensions * 2, hidden, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(TemporalResidualBlock(hidden, d) for d in (1, 2, 4, 8, 16))
        self.decoder = nn.Conv1d(hidden, dimensions, kernel_size=1)

    def forward(self, sample, mask):
        h = self.encoder(torch.cat((sample * (1-mask), mask), dim=-1).transpose(1, 2))
        zero_emb = torch.zeros((len(sample), h.shape[1]), device=h.device, dtype=h.dtype)
        for block in self.blocks:
            h = block(h, zero_emb)
        return self.decoder(h).transpose(1, 2)
