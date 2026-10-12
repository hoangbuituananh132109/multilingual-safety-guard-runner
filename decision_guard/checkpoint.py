"""Portable DP checkpoints; never discard the new decision head."""
from __future__ import annotations

import os
import hashlib
from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer, contract, step, rng_states):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Frozen base weights are reconstructed from the fingerprinted pretrained model.
    parameters = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
    buffers = {name: b.detach().cpu().clone() for name, b in model.named_buffers()}
    state = dict(format='decision_guard_dp_v1', contract=contract, step=step,
                 parameters=parameters, buffers=buffers, optimizer=optimizer.state_dict(), rng=rng_states)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, model, optimizer, contract):
    # Local trusted study checkpoints only: optimizer/RNG use Python objects.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('format') != 'decision_guard_dp_v1' or state['contract'] != contract:
        raise ValueError('Checkpoint contract differs: data/model/settings/world size must not change')
    parameters = dict(model.named_parameters())
    expected = {name for name, p in parameters.items() if p.requires_grad}
    if set(state['parameters']) != expected:
        raise ValueError('Trainable adapter/head keys differ')
    with torch.no_grad():
        for name, value in state['parameters'].items():
            parameters[name].copy_(value)
        buffers = dict(model.named_buffers())
        if set(buffers) != set(state['buffers']):
            raise ValueError('Backbone buffers differ')
        for name, value in state['buffers'].items():
            buffers[name].copy_(value)
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    return state


def _assert_state_equal(expected, actual):
    if isinstance(expected, torch.Tensor):
        if not torch.equal(expected.cpu(), actual.cpu()):
            raise RuntimeError('Checkpoint tensor did not round-trip exactly')
    elif isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise RuntimeError('Checkpoint state keys changed')
        for key in expected:
            _assert_state_equal(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        if len(expected) != len(actual):
            raise RuntimeError('Checkpoint state length changed')
        for left, right in zip(expected, actual):
            _assert_state_equal(left, right)
    elif expected != actual:
        raise RuntimeError('Checkpoint scalar state changed')


def verify_roundtrip(path, model, optimizer, contract, step):
    """Smoke-only destructive probe, restored from the checkpoint before returning.

    Mutates adapter/head parameters and clears optimizer state, then loads the
    real saved checkpoint. Checks both BF16 weights and FP32 masters/moments.
    """
    import copy
    path = Path(path)
    expected = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    if not expected or not optimizer.optimizer.state:
        raise RuntimeError('No trained adapter/head or optimizer state to verify')
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.zero_()
        for p in optimizer.masters:
            p.zero_()
    optimizer.optimizer.state.clear()
    state = load_checkpoint(path, model, optimizer, contract)
    if state['step'] != step:
        raise RuntimeError('Saved optimizer step is incorrect')
    actual = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
    _assert_state_equal(expected, actual)
    _assert_state_equal(expected_optimizer, optimizer.state_dict())
    for value in actual.values():
        if not torch.isfinite(value).all():
            raise RuntimeError('Nonfinite saved trainable parameter')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(passed=True, step=step, bytes=path.stat().st_size, sha256=digest.hexdigest(),
                trainable_tensors=len(actual), optimizer_restored=True,
                reload_scope='same process after zeroing trainables and clearing optimizer')
