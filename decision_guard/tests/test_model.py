import unittest
from types import SimpleNamespace
import torch
from torch import nn

from decision_guard.model import BinaryHead, PointerHead, cut_backbone, collate, MasterAdam, per_example_loss


class ModelTests(unittest.TestCase):
    def test_binary_reads_last_real_token_not_padding(self):
        head = BinaryHead(4)
        hidden = torch.randn(5, 2, 4)
        picks = torch.tensor([[1], [3]])
        first = head(hidden, picks)
        hidden[2:, 0] = 10000
        hidden[4:, 1] = 10000
        self.assertTrue(torch.equal(first, head(hidden, picks)))

    def test_pointer_scores_both_options_with_trainable_q_and_k(self):
        torch.manual_seed(10)
        head = PointerHead(4, 3)
        hidden = torch.randn(6, 2, 4, requires_grad=True)
        picks = torch.tensor([[1, 3, 5], [0, 2, 4]])
        scores = head(hidden, picks)
        self.assertEqual(tuple(scores.shape), (2, 2))
        nn.functional.cross_entropy(scores, torch.tensor([0, 1])).backward()
        self.assertGreater(head.query.weight.grad.abs().sum().item(), 0)
        self.assertGreater(head.key.weight.grad.abs().sum().item(), 0)
        self.assertGreater(hidden.grad.abs().sum().item(), 0)

    def test_cut_removes_unused_layers_and_lm_projection(self):
        model = nn.Module()
        model.config = SimpleNamespace(num_layers=4)
        model.decoder = nn.Module()
        model.decoder.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
        model.decoder.num_layers = 4
        model.decoder.final_layernorm = nn.LayerNorm(4)
        norm = model.decoder.final_layernorm
        model.output_layer = nn.Linear(4, 20)
        model.post_process = True
        cut_backbone(model, 2)
        self.assertEqual(len(model.decoder.layers), 2)
        self.assertEqual(model.decoder.num_layers, 2)
        self.assertEqual(model.config.num_layers, 2)
        self.assertIs(model.decoder.final_layernorm, norm)
        self.assertFalse(model.post_process)
        self.assertIsNone(model.output_layer)
        self.assertFalse(any(name.startswith('decoder.layers.2') for name, _ in model.named_parameters()))

    def test_padding_has_zero_completion_loss_and_valid_last_positions(self):
        a = dict(ids=[4, 5, 6], targets=[5, 6, 7], loss_start=1, label=0, picks=[2])
        b = dict(ids=[8], label=1, picks=[0])
        batch = collate([a, b], pad_id=0)
        self.assertEqual(batch['input_ids'].tolist(), [[4, 5, 6], [8, 0, 0]])
        self.assertEqual(batch['loss_mask'].tolist(), [[0, 1, 1], [0, 0, 0]])
        self.assertEqual(batch['picks'].tolist(), [[2], [0]])

    def test_fp32_master_preserves_small_updates_to_bf16_weight(self):
        value = nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
        optim = MasterAdam([value], lr=1e-4, weight_decay=0)
        for _ in range(30):
            value.grad = torch.ones_like(value)
            optim.step(clip=1)
            optim.zero_grad()
        self.assertLess(value.item(), 1.0)
        self.assertEqual(optim.masters[0].dtype, torch.float32)
        self.assertEqual(optim.optimizer.state[optim.masters[0]]['exp_avg'].dtype, torch.float32)

    def test_generation_weights_each_example_not_completion_length(self):
        losses = torch.tensor([[2., 2., 99.], [4., 99., 99.]], requires_grad=True)
        mask = torch.tensor([[1., 1., 0.], [1., 0., 0.]])
        result = per_example_loss(losses, dict(loss_mask=mask), 'json')
        self.assertEqual(result.tolist(), [2., 4.])
        result.sum().backward()
        self.assertEqual(losses.grad.tolist(), [[.5, .5, 0.], [1., 0., 0.]])


if __name__ == '__main__':
    unittest.main()
