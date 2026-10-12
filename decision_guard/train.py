"""Single-node DP training with a genuine Bridge backbone and trained decision heads."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

import torch
import torch.distributed as dist

from .checkpoint import load_checkpoint, save_checkpoint, verify_roundtrip
from .data import Cache, offline, sha256
from .model import MasterAdam, collate, per_example_loss


def accumulation(global_batch, micro_batch, world_size):
    if min(global_batch, micro_batch, world_size) < 1:
        raise ValueError('Batch sizes and world size must be positive')
    divisor = micro_batch * world_size
    if global_batch % divisor:
        raise ValueError('global_batch must be divisible by micro_batch * GPU count')
    return global_batch // divisor


def step_slots(order, step, micro, rank, world_size, micro_batch, global_batch):
    start = step * global_batch + (micro * world_size + rank) * micro_batch
    return [order[i] if i < len(order) else None for i in range(start, start + micro_batch)]


def validate_arm(arm):
    match = re.fullmatch(r'(json|label|binary|pointer)(36|27|18)', arm)
    if not match or (match[1] in ('json', 'label') and match[2] != '36'):
        raise ValueError('Use json36 / optional label36 / binary36,27,18 / pointer36,27,18')
    return match[1], int(match[2])


def planned_steps(rows, global_batch, smoke=False):
    return min(1, math.ceil(rows / global_batch)) if smoke else math.ceil(rows / global_batch)


def training_code_hash():
    return hashlib.sha256(''.join(sha256(Path(__file__).with_name(name)) for name in
                          ('train.py', 'model.py', 'bridge_backend.py', 'checkpoint.py', 'protocol.py', 'data.py', 'preflight.py')).encode()).hexdigest()


def check_smoke(certificate, contract):
    previous = certificate.get('contract', {})
    allowed_differences = {'smoke', 'cache_sha256'}
    expected = {k: v for k, v in contract.items() if k not in allowed_differences}
    actual = {k: v for k, v in previous.items() if k not in allowed_differences}
    verification = certificate.get('checkpoint_roundtrip', {})
    if (certificate.get('steps') != 1 or verification.get('passed') is not True or
            verification.get('step') != 1 or not verification.get('sha256') or
            previous.get('smoke') is not True or actual != expected):
        raise ValueError('Smoke certificate is stale or incompatible; rerun smoke with these exact settings')


def rank_zero_check(check):
    """Propagate local integrity failures to every rank before building the model."""
    message = [None]
    if dist.get_rank() == 0:
        try:
            check()
        except Exception as exc:
            message[0] = f'{type(exc).__name__}: {exc}'
    dist.broadcast_object_list(message, src=0)
    if message[0]:
        raise RuntimeError(message[0])


def sync_trainables(model):
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    metadata = [(name, tuple(p.shape), str(p.dtype)) for name, p in trainable]
    peers = [None] * dist.get_world_size()
    dist.all_gather_object(peers, metadata)
    if any(peer != metadata for peer in peers):
        raise RuntimeError('Trainable structure differs across DP ranks')
    with torch.no_grad():
        for _, parameter in trainable:
            dist.broadcast(parameter, src=0)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def summarize(totals, classification=True):
    tn, fp, fn, tp, loss, count = totals
    result = dict(examples=int(count), loss=loss / count if count else None)
    if classification:
        safe = tn / (tn + fp) if tn + fp else None
        unsafe = tp / (tp + fn) if tp + fn else None
        result.update(accuracy=(tn + tp) / count if count else None, safe_recall=safe,
                      unsafe_recall=unsafe,
                      balanced_accuracy=(safe + unsafe) / 2 if safe is not None and unsafe is not None else None,
                      tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return result


def _batch(data, slots, device):
    rows = []
    for index in slots:
        item = dict(data[0 if index is None else index])
        item['weight'] = 0.0 if index is None else 1.0
        rows.append(item)
    return {name: value.to(device) for name, value in collate(rows, data.manifest['pad_id']).items()}


def rng_snapshot():
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(), megatron=get_cuda_rng_tracker().get_states())


def rng_restore(state):
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])
    get_cuda_rng_tracker().set_states(state['megatron'])


def collective_save(path, model, optimizer, contract, step):
    # Tiny per-rank RNG objects only, never gather full model weights.
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, rng_snapshot())
    if dist.get_rank() == 0:
        save_checkpoint(path, model, optimizer, contract, step, states)
        print(f'CHECKPOINT SAVED step={step} path={path}', flush=True)
    dist.barrier()


@torch.no_grad()
def validate(model, data, micro_batch, kind, device):
    model.eval()
    rank, world = dist.get_rank(), dist.get_world_size()
    indices = list(range(rank, len(data), world))
    totals = torch.zeros(6, dtype=torch.float64, device=device)
    for start in range(0, len(indices), micro_batch):
        batch = _batch(data, indices[start:start + micro_batch], device)
        output = model(batch)
        losses = per_example_loss(output, batch, kind)
        totals[4] += losses.sum().double()
        totals[5] += len(losses)
        if kind not in ('json', 'label'):
            truth, pred = batch['labels'], output.argmax(-1)
            totals[0] += ((truth == 0) & (pred == 0)).sum()
            totals[1] += ((truth == 0) & (pred == 1)).sum()
            totals[2] += ((truth == 1) & (pred == 0)).sum()
            totals[3] += ((truth == 1) & (pred == 1)).sum()
    dist.all_reduce(totals)
    model.train()
    return summarize(totals.tolist(), kind not in ('json', 'label'))


def main():
    offline()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--model-receipt', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--global-batch', type=int, default=32)
    parser.add_argument('--micro-batch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--save-every', type=int, default=2000)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-certificate', type=Path)
    args = parser.parse_args()
    kind, depth = validate_arm(args.arm)
    if args.lr <= 0 or args.save_every < 1:
        raise ValueError('Learning rate and save interval must be positive')
    if not args.smoke and args.smoke_certificate is None:
        raise ValueError('Full training requires --smoke-certificate from the same arm/settings')
    rank, world, local = (int(os.environ.get(key, default)) for key, default in
                          (('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')))
    if int(os.environ.get('LOCAL_WORLD_SIZE', str(world))) != world:
        raise ValueError('This first version supports single-node data parallel only')
    if not torch.cuda.is_available():
        raise RuntimeError('Real training requires the approved NeMo image and CUDA; no CPU/HF fallback')
    grad_accum = accumulation(args.global_batch, args.micro_batch, world)
    training = Cache(args.cache / f'train.{kind}.sqlite')
    validation = Cache(args.cache / f'validation.{kind}.sqlite')
    request = training.manifest['request']
    if request != validation.manifest['request']:
        raise ValueError('Train and validation cache contracts differ')
    expected_sizes = (128, 16) if args.smoke else (274760, 1000)
    if (len(training), len(validation)) != expected_sizes:
        raise ValueError(f'Expected {expected_sizes} rows for this phase, refusing accidental full smoke or partial train')
    receipt = json.loads(args.model_receipt.read_text())
    if str(args.model.resolve()) != receipt['model_path'] or receipt['tokenizer'] != request['tokenizer']:
        raise ValueError('Local model/preflight receipt/tokenizer cache mismatch')
    code_hash = training_code_hash()
    protocol = {k: v for k, v in request.items() if k not in ('train_limit', 'validation_limit')}
    contract = dict(arm=args.arm, model=receipt['model_sha256'], cache_sha256=sha256(args.cache / 'cache_manifest.json'),
                    code_sha256=code_hash, cache_protocol=protocol,
                    environment=receipt['environment']['versions'], world_size=world, global_batch=args.global_batch,
                    micro_batch=args.micro_batch, lr=args.lr, seed=args.seed, epochs=1,
                    lora=dict(targets=['linear_q', 'linear_v'], rank=8, alpha=32, dropout=.05),
                    dtype='bf16', optimizer='AdamW_fp32_master', smoke=args.smoke)
    args.output.mkdir(parents=True, exist_ok=True)
    complete = args.output / 'run_complete.json'
    if complete.exists():
        completed = json.loads(complete.read_text())
        if completed['contract'] != contract:
            raise ValueError('Completed output belongs to a different experiment; choose another output')
        final = args.output / 'final.pt'
        if not final.is_file() or sha256(final) != completed.get('final_sha256'):
            raise ValueError('Completed run has a missing/changed final checkpoint; preserve it for inspection')
        print(f'SKIP completed {args.arm}', flush=True)
        return
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    try:
        def integrity():
            from .preflight import verify_receipt, image_api_checks
            verify_receipt(args.model, receipt)
            if image_api_checks()['versions'] != receipt['environment']['versions']:
                raise ValueError('Image package versions changed since preflight')
            cache_manifest = json.loads((args.cache / 'cache_manifest.json').read_text())
            if cache_manifest['request'] != request:
                raise ValueError('Cache manifest and SQLite contracts differ')
            for split in ('train', 'validation'):
                name = f'{split}.{kind}.sqlite'
                if sha256(args.cache / name) != cache_manifest['files'][name]:
                    raise ValueError(f'Token cache changed: {name}')
            if not args.smoke:
                check_smoke(json.loads(args.smoke_certificate.read_text()), contract)
        rank_zero_check(integrity)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        from .bridge_backend import build_model
        if rank == 0:
            print(json.dumps(dict(contract=contract, train_rows=len(training), validation_rows=len(validation),
                                  grad_accum=grad_accum, initialization='fresh pretrained Qwen3-4B')), flush=True)
        model = build_model(args.model, kind, depth, request['max_length'], args.seed)
        # Base tensors are imported from the same fingerprinted files. Synchronize
        # only new trainables, not all frozen 4B weights during DDP construction.
        sync_trainables(model)
        # Standard torch DDP handles only DP. MCore Q/K/V and attention still run in the backbone.
        ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False,
                                                        find_unused_parameters=False, init_sync=False)
        optimizer = MasterAdam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        total_steps = planned_steps(len(training), args.global_batch, args.smoke)
        order = list(range(len(training)))
        random.Random(args.seed).shuffle(order)
        saved = sorted(args.output.glob('checkpoint-*.pt'), key=lambda p: int(p.stem.split('-')[-1]))
        start = 0
        if saved:
            state = load_checkpoint(saved[-1], model, optimizer, contract)
            start = state['step']
            rng_restore(state['rng'][rank])
            if rank == 0:
                print(f'RESUME step={start} from {saved[-1]}', flush=True)
        # Different dropout streams per DP rank; initialization was identical before DDP broadcast.
        else:
            torch.manual_seed(args.seed + rank)
            torch.cuda.manual_seed(args.seed + rank)
        begin = time.monotonic()
        for step in range(start, total_steps):
            optimizer.zero_grad()
            real_count = min(args.global_batch, len(order) - step * args.global_batch)
            running = torch.zeros(1, dtype=torch.float64, device=device)
            for micro in range(grad_accum):
                slots = step_slots(order, step, micro, rank, world, args.micro_batch, args.global_batch)
                batch = _batch(training, slots, device)
                sync = ddp.no_sync() if micro != grad_accum - 1 else nullcontext()
                with sync:
                    output = ddp(batch)
                    loss_sum = (per_example_loss(output, batch, kind) * batch['weights']).sum()
                    # DDP averages gradients; undo world averaging then normalize by real global rows.
                    (loss_sum * world / real_count).backward()
                running += loss_sum.detach().double()
            norm = optimizer.step(clip=1.0)
            dist.all_reduce(running)
            if rank == 0 and ((step + 1) % 10 == 0 or step < 2 or step + 1 == total_steps):
                print(json.dumps(dict(step=step + 1, total_steps=total_steps, loss=float(running / real_count),
                                      grad_norm=norm, elapsed_seconds=time.monotonic() - begin,
                                      allocated_MB=round(torch.cuda.memory_allocated() / 2**20))), flush=True)
            if (step + 1) % args.save_every == 0 or step + 1 == total_steps:
                collective_save(args.output / f'checkpoint-{step + 1:06d}.pt', model, optimizer, contract, step + 1)
        metrics = validate(model, validation, args.micro_batch, kind, device)
        collective_save(args.output / 'final.pt', model, optimizer, contract, total_steps)
        roundtrip = None
        if args.smoke:
            model.eval()
            probe = _batch(validation, [rank % len(validation)], device)
            with torch.no_grad():
                expected_output = model(probe).detach().cpu().clone()
            roundtrip = verify_roundtrip(args.output / 'final.pt', model, optimizer, contract, total_steps)
            with torch.no_grad():
                restored_output = model(probe).detach().cpu()
            torch.testing.assert_close(restored_output, expected_output, rtol=1e-6, atol=1e-6)
            roundtrip['forward_after_reload_passed'] = True
            model.train()
            dist.barrier()
        if rank == 0:
            result = dict(contract=contract, steps=total_steps, resumed_from_step=start, validation=metrics,
                          elapsed_seconds=time.monotonic() - begin,
                          trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                          note='Validation split only, not held-out benchmark evaluation')
            result['final_sha256'] = sha256(args.output / 'final.pt')
            if roundtrip is not None:
                result['checkpoint_roundtrip'] = roundtrip
            write_json(args.output / 'train_results.json', result)
            if args.smoke:
                write_json(args.output / 'smoke_pass.json', result)
            else:
                write_json(complete, result)
            print(json.dumps(result, indent=2), flush=True)
        dist.barrier()
    finally:
        training.connection.close()
        validation.connection.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
