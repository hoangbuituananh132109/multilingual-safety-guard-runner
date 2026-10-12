"""Offline image/API/input checks; no training, no model weight allocation on GPU."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path

from .data import offline, sha256, tokenizer_fingerprint

IMAGE = 'nvcr.io/nvidia/nemo:26.08'
QWEN4B = dict(model_type='qwen3', num_hidden_layers=36, hidden_size=2560,
              intermediate_size=9728, num_attention_heads=32, num_key_value_heads=8,
              head_dim=128, vocab_size=151936, tie_word_embeddings=True)


def inspect_model(model_dir):
    model_dir = Path(model_dir).resolve()
    config = json.loads((model_dir / 'config.json').read_text())
    if any(config.get(key) != value for key, value in QWEN4B.items()):
        raise ValueError('Only the approved Qwen3-4B / 36-layer Qwen3 backbone is supported')
    if config.get('quantization_config'):
        raise ValueError('Use the original unquantized pretrained checkpoint for this BF16 study')
    tokenizer = tokenizer_fingerprint(model_dir)
    if not (model_dir / 'tokenizer.json').is_file():
        raise FileNotFoundError('Local tokenizer.json required')
    index = model_dir / 'model.safetensors.index.json'
    if index.is_file():
        mapping = json.loads(index.read_text())['weight_map']
        names = sorted(set(mapping.values()))
    elif (model_dir / 'model.safetensors').is_file():
        names = ['model.safetensors']
    else:
        raise FileNotFoundError('Local safetensors checkpoint / shard index required')
    hashes = {}
    for name in names:
        path = (model_dir / name).resolve()
        if model_dir not in path.parents or not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'Missing/unsafe local shard: {name}')
        print(f'Fingerprint local model shard: {name}', flush=True)
        hashes[name] = sha256(path)
    if index.is_file():
        hashes[index.name] = sha256(index)
    digest = hashlib.sha256(json.dumps(dict(weights=hashes, tokenizer=tokenizer), sort_keys=True).encode()).hexdigest()
    return dict(model_path=str(model_dir), model_sha256=digest, weights=hashes, tokenizer=tokenizer)


def verify_receipt(model_dir, receipt):
    current = inspect_model(model_dir)
    if any(current[key] != receipt[key] for key in current):
        raise ValueError('Pretrained model changed since preflight; no stale receipt is accepted')
    return current


def image_api_checks():
    if os.environ.get('DG_IMAGE', IMAGE) != IMAGE:
        raise RuntimeError(f'The study is locked to {IMAGE}')
    import torch
    import transformer_engine.pytorch
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.model_provider import ModelProviderMixin
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA, LoRALinearSplitQKV
    from .bridge_backend import probe_native_qv_wrapper
    # Check the installed image, not just documentation examples.
    if not hasattr(AutoBridge, 'to_megatron_provider'):
        raise RuntimeError('Installed Bridge lacks the required provider API')
    params = inspect.signature(ModelProviderMixin.provide_distributed_model).parameters
    if not {'wrap_with_ddp', 'mixed_precision_wrapper'} <= params.keys():
        raise RuntimeError('Bridge provider API incompatible with this package')
    CanonicalLoRA(target_modules=['linear_q', 'linear_v'], dim=8, alpha=32, dropout=.05)
    wrapper_probe = probe_native_qv_wrapper(LoRALinearSplitQKV)
    if 'init_sync' not in inspect.signature(torch.nn.parallel.DistributedDataParallel).parameters:
        raise RuntimeError('Installed PyTorch lacks explicit DDP init_sync control')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no fallback or installation will be attempted')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 GPU support required')
    versions = {}
    for package in ('torch', 'transformers', 'megatron-bridge', 'megatron-core', 'transformer-engine'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'source checkout (API import succeeded)'
    return dict(expected_image=IMAGE, versions=versions, visible_gpus=torch.cuda.device_count(),
                qv_wrapper_probe=wrapper_probe,
                device_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])


def main():
    offline()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    environment = image_api_checks()
    receipt = inspect_model(args.model)
    from transformers import AutoTokenizer
    from .protocol import Encoder, MARKERS
    tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), local_files_only=True, trust_remote_code=False)
    for marker in MARKERS:
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.convert_ids_to_tokens(ids[0]) != marker:
            raise RuntimeError(f'Pointer marker not present as a single existing token: {marker}')
    Encoder(tokenizer, int(os.environ.get('DG_MAX_LENGTH', '2048')))
    receipt['environment'] = environment
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(dict(status='PREFLIGHT_PASS', **receipt), indent=2))


if __name__ == '__main__':
    main()
