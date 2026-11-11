"""Non-experimental unit checks for manuscript equations and leakage boundaries.

Run: python -m unittest discover -s tests -v
These are formula-level tests, NOT benchmark experiments or data preparation.
"""
import unittest
import numpy as np
import torch
from vadiffad.diffusion import variable_coefficients
from vadiffad.path import local_signature, temporal_retention
from vadiffad.reference import NormalPathBank
from vadiffad.scoring import EmpiricalUpperTail, local_top_q, warmup_thresholds
from config.parser import SMAPConfig


class MainMethodFormulaTests(unittest.TestCase):
    def test_variable_schedule_preserves_mean_noise(self):
        t = np.linspace(0, 18, 1000)
        series = np.stack([np.sin(t*j)*(.1+j*.2) for j in range(1, 26)], axis=1)
        cfg = SMAPConfig()
        c = variable_coefficients(series, cfg)
        self.assertEqual(c.shape, (25,))
        self.assertAlmostEqual(float(c.mean()), 1., places=5)
        self.assertGreaterEqual(float(c.min()), cfg.cmin-1e-5)
        self.assertLessEqual(float(c.max()), cfg.cmax+1e-5)

    def test_signed_responses_are_kept_in_normal_signature(self):
        # K=3 -> m=2K-1=5; negative E values must remain signed.
        e = torch.tensor([[[[-1., 2., -3.]]]])
        dg = torch.tensor([[[1., 1., 1.]]])
        v, c, h = local_signature(e, dg)
        self.assertEqual(h.shape, (1, 1, 1, 5))
        np.testing.assert_allclose(h[0, 0, 0, :3].numpy(), [-1, 2, -3])
        self.assertAlmostEqual(float(v[0, 0, 0]), (1+4+9)/3, places=5)

    def test_signed_temporal_retention(self):
        x = torch.tensor([[[1., 1.], [2., 2.], [3., 3.]]])
        r = temporal_retention(x, -x)
        self.assertLess(float(r.item()), -.99)

    def test_single_normal_reference_per_variable(self):
        rng = np.random.default_rng(1)
        h = rng.normal(size=(80, 3, 5))
        bank = NormalPathBank(shrinkage=.1).fit(h)
        self.assertEqual(bank.mean.shape, (3, 5))
        self.assertEqual(bank.inverse.shape, (3, 5, 5))
        self.assertTrue(np.all(np.linalg.eigvalsh(bank.inverse) > 0))

    def test_upper_tail_ecdf_and_topq_all_variables(self):
        cal = np.arange(40, dtype=float).reshape(10, 2, 2)
        z = EmpiricalUpperTail().fit(cal)
        out = z.transform(cal[-1:]+100)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(np.all(out > 1))
        local = np.asarray([[0, 1, 8, 6], [4, 5, 6, 7]], dtype=float)
        np.testing.assert_allclose(local_top_q(local, 2), [7, 6.5])

    def test_unlabeled_entity_specific_thresholds(self):
        scores = np.arange(200, dtype=float)
        cfg = SMAPConfig(warmup_fraction=.05, warmup_quantile=.99)
        thresholds, meta = warmup_thresholds(scores, [(0, 100), (100, 200)], cfg)
        self.assertEqual([r['warmup_points'] for r in meta], [5, 5])
        self.assertLess(float(thresholds[0]), float(thresholds[-1]))


if __name__ == '__main__':
    unittest.main()
