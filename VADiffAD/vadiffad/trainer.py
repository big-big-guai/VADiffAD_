"""Normal-only training of the shared noise field and context estimator."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from vadiffad.model import TemporalNoisePredictor, MaskedContextReconstructor


class NormalTrainer:
    def __init__(self, dimensions, cfg, schedule, device):
        self.cfg, self.schedule, self.device = cfg, schedule, device
        self.predictor = TemporalNoisePredictor(dimensions, cfg.hidden_dim, cfg.timesteps).to(device)
        self.context = MaskedContextReconstructor(dimensions, cfg.hidden_dim).to(device)
        self.optimizer = torch.optim.AdamW(
            list(self.predictor.parameters()) + list(self.context.parameters()),
            lr=cfg.learning_rate, weight_decay=1e-4)

    def _batch_loss(self, x):
        cfg, sch = self.cfg, self.schedule
        timesteps = torch.randint(1, cfg.timesteps+1, (len(x),), device=x.device).float()
        epsilon = torch.randn_like(x)
        state = sch.perturb(x, timesteps, epsilon)
        noise_loss = (self.predictor(state, timesteps, sch.coeff)-epsilon).square().mean()
        mask = (torch.rand_like(x) < cfg.mask_probability).float()
        prediction = self.context(x, mask)
        context_loss = ((prediction-x).square()*mask).sum()/mask.sum().clamp_min(1)
        return noise_loss + cfg.context_loss_weight*context_loss

    def train(self, normal_train, normal_validation, output):
        cfg = self.cfg
        train = torch.as_tensor(normal_train, device=self.device)
        validation = torch.as_tensor(normal_validation, device=self.device)
        best, not_improved, history = np.inf, 0, []
        path = Path(output)/'checkpoint.pt'
        for epoch in range(cfg.max_epochs):
            self.predictor.train()
            self.context.train()
            epoch_loss = []
            for ids in torch.randperm(len(train), device=self.device).split(cfg.batch_size):
                self.optimizer.zero_grad(set_to_none=True)
                loss = self._batch_loss(train[ids])
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.predictor.parameters()) + list(self.context.parameters()), max_norm=1.)
                self.optimizer.step()
                epoch_loss.append(float(loss.detach().cpu()))
            self.predictor.eval()
            self.context.eval()
            with torch.no_grad(), torch.random.fork_rng(devices=[self.device.index or 0] if self.device.type == 'cuda' else []):
                torch.manual_seed(2026)
                val_loss = np.mean([float(self._batch_loss(batch).cpu())
                                    for batch in validation.split(cfg.batch_size)])
            row = {'epoch': epoch+1, 'training_loss': float(np.mean(epoch_loss)),
                   'validation_loss': float(val_loss)}
            history.append(row)
            if val_loss < best - 1e-5:
                best, not_improved = val_loss, 0
                torch.save({'predictor': self.predictor.state_dict(),
                            'context': self.context.state_dict(),
                            'epoch': epoch+1, 'validation_loss': float(val_loss)}, path)
            else:
                not_improved += 1
            if epoch == 0 or (epoch+1) % 5 == 0:
                print(json.dumps(row), flush=True)
            if not_improved >= cfg.patience:
                break
        (Path(output)/'training_history.json').write_text(
            json.dumps(history, indent=2), encoding='utf-8')
        try:
            saved = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            saved = torch.load(path, map_location=self.device)
        self.predictor.load_state_dict(saved['predictor'])
        self.context.load_state_dict(saved['context'])
        self.predictor.eval()
        self.context.eval()
        return saved['epoch']

    @torch.no_grad()
    def masked_context_error(self, batch):
        """Every position is masked exactly once over four deterministic masks."""
        result = torch.zeros_like(batch)
        temporal = torch.arange(batch.shape[1], device=batch.device)[:, None]
        variables = torch.arange(batch.shape[2], device=batch.device)[None, :]
        for phase in range(4):
            mask = (((temporal + variables) % 4) == phase).to(batch.dtype)[None].expand_as(batch)
            result += (self.context(batch, mask)-batch).abs()*mask
        return result
