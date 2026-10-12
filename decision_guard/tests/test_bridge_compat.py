import unittest
from types import SimpleNamespace
import torch
from torch import nn

from decision_guard.bridge_backend import fill_missing_key_adapter


class SplitQKV(nn.Module):
    """CPU double of the relevant native wrapper calls; not a Bridge runtime test."""
    def __init__(self):
        super().__init__()
        self.to_wrap = nn.Linear(3, 8, bias=False)
        self.to_wrap.config = SimpleNamespace(kv_channels=2, num_query_groups=1)
        self.adapter = nn.ModuleDict(dict(adapter_q=nn.Linear(3, 4, bias=False),
                                         adapter_k=None, adapter_v=nn.Linear(3, 2, bias=False)))

    def forward(self, x):
        q, k, v = [self.adapter[name](x) for name in ('adapter_q', 'adapter_k', 'adapter_v')]
        return self.to_wrap(x) + torch.cat((q, k, v), -1)


class CompatibilityTests(unittest.TestCase):
    def test_qv_only_native_missing_key_is_zero_not_trainable(self):
        model = SplitQKV()
        x = torch.randn(2, 3)
        with self.assertRaises(TypeError):
            model(x)
        self.assertEqual(fill_missing_key_adapter(model, SplitQKV), 1)
        self.assertEqual(fill_missing_key_adapter(model, SplitQKV), 0)
        self.assertEqual(list(model.adapter.adapter_k.parameters()), [])
        out = model(x)
        torch.testing.assert_close(out[:, 4:6], model.to_wrap(x)[:, 4:6])
        out.sum().backward()
        self.assertIsNotNone(model.adapter.adapter_q.weight.grad)
        self.assertIsNotNone(model.adapter.adapter_v.weight.grad)


if __name__ == '__main__':
    unittest.main()
