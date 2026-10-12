"""Trainable heads, physical backbone truncation, and FP32 optimizer state."""
from __future__ import annotations

import math
import torch
from torch import nn


class BinaryHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.projection = nn.Linear(hidden_size, 2, dtype=torch.float32)

    def forward(self, hidden, picks):
        batch = torch.arange(hidden.shape[1], device=hidden.device)
        return self.projection(hidden[picks[:, -1], batch].float())


class PointerHead(nn.Module):
    """KEV-style two-option pointer, not the official pretrained KEV implementation."""
    def __init__(self, hidden_size, pointer_dim=256):
        super().__init__()
        self.query = nn.Linear(hidden_size, pointer_dim, bias=False, dtype=torch.float32)
        self.key = nn.Linear(hidden_size, pointer_dim, bias=False, dtype=torch.float32)
        self.scale = math.sqrt(pointer_dim)

    def forward(self, hidden, picks):
        batch = torch.arange(hidden.shape[1], device=hidden.device)
        decision = hidden[picks[:, 2], batch].float()
        options = torch.stack([hidden[picks[:, 0], batch], hidden[picks[:, 1], batch]], dim=1).float()
        return (self.query(decision)[:, None, :] * self.key(options)).sum(-1) / self.scale


def cut_backbone(backbone, depth):
    layers = backbone.decoder.layers
    if not 1 <= depth <= len(layers):
        raise ValueError('Invalid physical depth')
    backbone.decoder.layers = nn.ModuleList(list(layers[:depth]))
    backbone.decoder.num_layers = depth
    backbone.config.num_layers = depth
    # Keep decoder.post_process=True: it controls the pretrained final norm.
    backbone.post_process = False
    backbone.output_layer = None


def collate(rows, pad_id):
    length = max(len(row['ids']) for row in rows)
    batch = len(rows)
    ids = torch.full((batch, length), pad_id, dtype=torch.long)
    targets = torch.zeros((batch, length), dtype=torch.long)
    loss_mask = torch.zeros((batch, length), dtype=torch.float32)
    for i, row in enumerate(rows):
        n = len(row['ids'])
        ids[i, :n] = torch.tensor(row['ids'])
        if 'targets' in row:
            targets[i, :n] = torch.tensor(row['targets'])
            loss_mask[i, row['loss_start']:n] = 1
    return dict(input_ids=ids, position_ids=torch.arange(length).expand(batch, -1),
                targets=targets, loss_mask=loss_mask,
                picks=torch.tensor([row['picks'] for row in rows], dtype=torch.long),
                labels=torch.tensor([row['label'] for row in rows], dtype=torch.long),
                weights=torch.tensor([row.get('weight', 1.0) for row in rows], dtype=torch.float32))


def per_example_loss(output, batch, kind):
    if kind in ('json', 'label'):
        if output.shape != batch['loss_mask'].shape:
            raise RuntimeError('Megatron LM loss must have batch/sequence layout')
        mask = batch['loss_mask']
        return (output.float() * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    return nn.functional.cross_entropy(output.float(), batch['labels'], reduction='none')


class MasterAdam:
    """BF16 adapter weights with FP32 masters/moments; heads already use FP32."""
    def __init__(self, parameters, lr=1e-5, weight_decay=.01):
        self.parameters = list(parameters)
        self.masters = [nn.Parameter(p.detach().float().clone()) for p in self.parameters]
        self.optimizer = torch.optim.AdamW(self.masters, lr=lr, betas=(.9, .999), eps=1e-8,
                                           weight_decay=weight_decay)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None
        self.optimizer.zero_grad(set_to_none=True)

    def step(self, clip=1.0):
        for parameter, master in zip(self.parameters, self.masters):
            if parameter.grad is None:
                raise RuntimeError('Unused trainable parameter: head/LoRA graph is disconnected')
            master.grad = parameter.grad.detach().float()
        norm = nn.utils.clip_grad_norm_(self.masters, clip, error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            for parameter, master in zip(self.parameters, self.masters):
                parameter.copy_(master)
        return float(norm)

    def state_dict(self):
        return dict(masters=[p.detach().cpu() for p in self.masters], optimizer=self.optimizer.state_dict())

    def load_state_dict(self, state):
        if len(state['masters']) != len(self.masters):
            raise ValueError('Optimizer parameter count changed')
        self.optimizer.load_state_dict(state['optimizer'])
        with torch.no_grad():
            for parameter, master, saved in zip(self.parameters, self.masters, state['masters']):
                master.copy_(saved)
                parameter.copy_(master)
