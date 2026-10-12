"""Binary head benchmark eval with Megatron Bridge DP; no vLLM/generation/network."""
from __future__ import annotations

import argparse
import csv
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import time

from . import benchmark as bench
from .data import offline, sha256


def implementation_hash():
    return hashlib.sha256(''.join(sha256(Path(__file__).with_name(name)) for name in
                          ('evaluate.py', 'benchmark.py', 'protocol.py', 'model.py',
                           'bridge_backend.py', 'checkpoint.py', 'preflight.py')).encode()).hexdigest()


def source_contract(train_run, receipt, arm, model_path):
    from .train import training_code_hash
    path = train_run / 'run_complete.json'
    if not path.exists():
        path = train_run / 'smoke_pass.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    contract = report['contract']
    if contract['arm'] != arm or contract['model'] != receipt['model_sha256']:
        raise ValueError('Eval arm or pretrained model differs from the trained checkpoint')
    if str(model_path.resolve()) != receipt['model_path']:
        raise ValueError('Model path differs from preflight receipt')
    if contract['code_sha256'] != training_code_hash():
        raise ValueError('Training implementation changed; preserve the original code with its checkpoint')
    if contract['environment'] != receipt['environment']['versions']:
        raise ValueError('Image versions changed since training')
    final = train_run / 'final.pt'
    if not final.is_file() or sha256(final) != report['final_sha256']:
        raise ValueError('Trained final checkpoint missing or changed')
    return contract, report['final_sha256']


def publish(directory, contract, predictions):
    metrics = bench.metrics(predictions)
    bench.write_json(directory / 'metrics.json', metrics)
    path = directory / 'predictions.jsonl'
    temporary = path.with_suffix('.jsonl.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        for item in predictions:
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temporary, path)
    csv_path = directory / 'metrics.csv'
    temporary = csv_path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    os.replace(temporary, csv_path)
    manifest = dict(contract=contract, examples=len(predictions),
                    truncated=sum(p['truncated'] for p in predictions),
                    files={name: sha256(directory / name) for name in
                           ('predictions.jsonl', 'metrics.json', 'metrics.csv')})
    bench.write_json(directory / 'run_manifest.json', manifest)  # Completion marker LAST.
    print_table(metrics)
    print(json.dumps(manifest, indent=2), flush=True)


def print_table(metrics):
    print(f'{"benchmark":27} {"n":>8} {"safe":>7} {"unsafe":>7} {"BA":>8} {"Safe-R":>8} {"Unsafe-R":>9}', flush=True)
    for item in metrics:
        if item['group'] != 'ALL' and not item['group'].endswith('|ALL|ALL'):
            continue
        values = ['-' if item[k] is None else f'{item[k]:.2f}' for k in
                  ('balanced_accuracy', 'safe_recall', 'unsafe_recall')]
        print(f'{item["group"].split("|")[0]:27} {item["examples"]:8} {item["safe_examples"]:7} '
              f'{item["unsafe_examples"]:7} {values[0]:>8} {values[1]:>8} {values[2]:>9}', flush=True)


def main():
    offline()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-receipt', type=Path, required=True)
    parser.add_argument('--train-run', type=Path, required=True)
    parser.add_argument('--arm', choices=('binary36', 'binary18'), required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-length', type=int, default=40000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-batch-tokens', type=int, default=32768)
    parser.add_argument('--limit-per-benchmark', type=int, default=0)
    args = parser.parse_args()
    if not 2048 <= args.max_length <= 40960 or min(args.batch_size, args.max_batch_tokens) < 1 or args.limit_per_benchmark < 0:
        raise ValueError('Eval max-length must be 2048..40960, batches positive, limit nonnegative')
    rank, world, local = (int(os.environ.get(key, default)) for key, default in
                         (('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')))
    if int(os.environ.get('LOCAL_WORLD_SIZE', str(world))) != world:
        raise ValueError('Only single-node DP eval is supported')
    import torch
    import torch.distributed as dist
    from .train import rank_zero_check
    from .checkpoint import load_checkpoint
    from .protocol import Encoder, POLICY
    from .model import collate
    from .preflight import verify_receipt, image_api_checks
    if not torch.cuda.is_available():
        raise RuntimeError('Binary benchmark eval requires the approved NeMo image and CUDA')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=timedelta(hours=2))
    try:
        receipt = json.loads(args.model_receipt.read_text(encoding='utf-8'))
        train_contract, checkpoint_sha = source_contract(args.train_run, receipt, args.arm, args.model)
        contract = dict(format='decision_guard_binary_eval_v1', arm=args.arm,
                        checkpoint_sha256=checkpoint_sha, training_contract=train_contract,
                        bundle_sha256=bench.BUNDLE_SHA, eval_code_sha256=implementation_hash(),
                        policy_sha256=hashlib.sha256(POLICY.encode()).hexdigest(),
                        protocol='same binary chat wrapper; serialized conversation head+tail if truncated',
                        max_length=args.max_length, batch_size=args.batch_size,
                        max_batch_tokens=args.max_batch_tokens, limit_per_benchmark=args.limit_per_benchmark,
                        world_size=world, decision='softmax FP32, unsafe_probability > 0.5, tie=safe',
                        backend='Megatron Bridge / Transformer Engine / BF16 / data parallel',
                        network_policy='local files only, no model loading via HF runtime, no generation')

        def integrity():
            verify_receipt(args.model, receipt)
            if image_api_checks()['versions'] != train_contract['environment']:
                raise ValueError('Image package versions differ from training')
            print(json.dumps(bench.audit(args.bundle), indent=2), flush=True)
            bench.ensure_contract(args.output, contract)
        rank_zero_check(integrity)
        # Compact metadata only; do not retain all source prompts in RAM.
        expected = [{key: item[key] for key in ('index', 'id', 'label', 'benchmark', 'language', 'view')}
                    for item in bench.rows(args.bundle, args.limit_per_benchmark)]
        complete = args.output / 'run_manifest.json'
        if complete.exists():
            report = json.loads(complete.read_text(encoding='utf-8'))
            if report['contract'] != contract or any(sha256(args.output / name) != digest for name, digest in report['files'].items()):
                raise ValueError('Completed eval output changed; preserve it for inspection')
            if rank == 0:
                print('SKIP completed binary benchmark eval', flush=True)
                print_table(json.loads((args.output / 'metrics.json').read_text(encoding='utf-8')))
            return
        done = bench.completed_indices(args.output, rank)
        local_expected = [item for item in expected if item['index'] % world == rank]
        if not done <= {item['index'] for item in local_expected}:
            raise ValueError('Saved eval parts belong to another rank/shard')
        saved_predictions = list(bench.read_parts(args.output, rank))
        bench.check_predictions(saved_predictions, [item for item in local_expected if item['index'] in done])
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), local_files_only=True, trust_remote_code=False)
        encoder = Encoder(tokenizer, args.max_length)
        from .bridge_backend import build_model
        model = build_model(args.model, 'binary', int(args.arm[6:]), args.max_length, train_contract['seed'])
        load_checkpoint(args.train_run / 'final.pt', model, None, train_contract)
        model.eval()
        # DP inference has no parameter sync or collective per forward.
        def encoded_rows():
            buffer = []
            for item in bench.rows(args.bundle, args.limit_per_benchmark):
                if item['index'] % world == rank and item['index'] not in done:
                    buffer.append(bench.encode(encoder, item))
                    if len(buffer) >= 128:
                        yield from sorted(buffer, key=lambda r: len(r['ids']))
                        buffer = []
            yield from sorted(buffer, key=lambda r: len(r['ids']))
        parts = list(args.output.glob(f'rank-{rank:02d}-part-*.json'))
        part_number = max((int(p.stem.split('-')[-1]) for p in parts), default=-1) + 1
        pending = []
        begin = time.monotonic()
        with torch.inference_mode():
            for items in bench.batches(encoded_rows(), args.batch_size, args.max_batch_tokens):
                batch = {key: value.to(device) for key, value in collate(items, tokenizer.pad_token_id).items()}
                probabilities = model(batch).float().softmax(-1)
                if not torch.isfinite(probabilities).all():
                    raise RuntimeError('Nonfinite binary probabilities; eval is not valid')
                for item, probability in zip(items, probabilities[:, 1].cpu().tolist()):
                    pending.append(dict(index=item['index'], id=item['id'], benchmark=item['benchmark'],
                                        language=item['language'], view=item['view'], label=item['label'],
                                        expected_output=('safe', 'unsafe')[item['label']],
                                        unsafe_probability=probability, prediction=int(probability > .5),
                                        decision=('safe', 'unsafe')[int(probability > .5)],
                                        input_tokens=len(item['ids']), truncated=item['truncated']))
                if len(pending) >= 128:
                    bench.save_part(args.output, rank, part_number, pending)
                    done.update(item['index'] for item in pending)
                    pending, part_number = [], part_number + 1
                    print(f'eval rank={rank} saved={len(done)}/{len(local_expected)} elapsed={time.monotonic()-begin:.1f}s', flush=True)
            if pending:
                bench.save_part(args.output, rank, part_number, pending)
        dist.barrier()

        def finalize():
            predictions = bench.check_predictions(list(bench.read_parts(args.output)), expected)
            publish(args.output, contract, predictions)
        rank_zero_check(finalize)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
