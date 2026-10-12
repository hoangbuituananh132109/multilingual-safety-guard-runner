import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn

from decision_guard.checkpoint import save_checkpoint, load_checkpoint
import decision_guard.checkpoint as checkpoint
from decision_guard.model import MasterAdam


class CheckpointTests(unittest.TestCase):
    def test_smoke_verifies_restoration_after_mutating_weights_and_optimizer(self):
        # Break caught: a smoke claims save success without restoring trained state.
        self.assertTrue(hasattr(checkpoint, 'verify_roundtrip'), 'GPU smoke needs a real save/load verification')
        model = nn.Linear(3, 2)
        optim = MasterAdam(list(model.parameters()), lr=1e-3)
        model(torch.ones(1, 3)).sum().backward()
        optim.step(clip=1)
        expected = model(torch.ones(1, 3)).detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint-000001.pt'
            contract = {'arm': 'pointer36', 'smoke': True}
            save_checkpoint(path, model, optim, contract, 1, [])
            report = checkpoint.verify_roundtrip(path, model, optim, contract, 1)
            self.assertTrue(report['passed'])
            self.assertEqual(report['step'], 1)
            self.assertGreater(report['bytes'], 0)
            self.assertEqual(len(report['sha256']), 64)
            self.assertTrue(torch.equal(model(torch.ones(1, 3)), expected))
            self.assertTrue(optim.optimizer.state)

    def test_roundtrip_retains_head_optimizer_and_position(self):
        torch.manual_seed(3)
        model = nn.Sequential(nn.Linear(3, 2))
        optim = MasterAdam(list(model.parameters()), lr=1e-3)
        model(torch.ones(2, 3)).sum().backward()
        optim.step(clip=1)
        expected = model(torch.ones(2, 3)).detach().clone()
        contract = {'arm': 'binary18', 'data': 'hash', 'world_size': 1}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pt'
            save_checkpoint(path, model, optim, contract, step=12, rng_states=[{'torch': torch.get_rng_state()}])
            for p in model.parameters():
                p.data.zero_()
            state = load_checkpoint(path, model, optim, contract)
            self.assertEqual(state['step'], 12)
            self.assertTrue(torch.equal(model(torch.ones(2, 3)), expected))
            self.assertTrue(optim.optimizer.state)
            with self.assertRaises(ValueError):
                load_checkpoint(path, model, optim, dict(contract, data='different'))


if __name__ == '__main__':
    unittest.main()
