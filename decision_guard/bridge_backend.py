"""Local HF-weight import through Megatron Bridge; no HF model/Trainer fallback."""
from __future__ import annotations

import torch
from torch import nn
from .model import BinaryHead, PointerHead, cut_backbone


class ZeroKeyAdapter(nn.Module):
    """No K update or parameters: compatibility for native subset-QKV wrappers."""
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, x, *args, **kwargs):
        return x.new_zeros((*x.shape[:-1], self.width))


def fill_missing_key_adapter(model, wrapper_type):
    """Keep Bridge's native Q/V adapters/interleaving; supply the missing zero K.

    Bridge v0.6 creates adapter_k=None for Q/V-only targets but its wrapper calls
    all three branches unconditionally. This local shim adds no trainable K and
    does not monkey-patch the installed package or copy its forward implementation.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, wrapper_type) and module.adapter.adapter_k is None:
            config = module.to_wrap.config
            module.adapter['adapter_k'] = ZeroKeyAdapter(config.kv_channels * config.num_query_groups)
            count += 1
    return count


def probe_native_qv_wrapper(wrapper_type):
    """CPU-only probe of the installed native wrapper, including backward."""
    from types import SimpleNamespace
    class PackedLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(kv_channels=2, num_attention_heads=2, num_query_groups=1)
        def forward(self, x, *args, **kwargs):
            return x.new_zeros((*x.shape[:-1], 8)), None
    adapters = nn.ModuleDict(dict(adapter_q=nn.Linear(3, 4, bias=False),
                                 adapter_k=None, adapter_v=nn.Linear(3, 2, bias=False)))
    module = wrapper_type(PackedLinear(), adapters)
    fill_missing_key_adapter(module, wrapper_type)
    x = torch.randn(2, 1, 3)
    output, _ = module(x)
    expected = torch.cat((adapters.adapter_q(x), x.new_zeros(2, 1, 2), adapters.adapter_v(x)), -1)
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    if adapters.adapter_q.weight.grad is None or adapters.adapter_v.weight.grad is None:
        raise RuntimeError('Installed CanonicalLoRA Q/V wrapper backward failed')
    return 'installed_native_qv_wrapper_cpu_forward_backward_pass'


class DecisionModel(nn.Module):
    def __init__(self, backbone, kind, head=None):
        super().__init__()
        self.backbone, self.kind, self.head = backbone, kind, head

    def forward(self, batch):
        # Right padding is future-only under causal attention; real token states cannot read it.
        # Padding positions are never pooled and have zero completion loss.
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = self.backbone(input_ids=batch['input_ids'], position_ids=batch['position_ids'],
                                   attention_mask=None,
                                   labels=batch['targets'] if self.kind in ('json', 'label') else None)
        if self.head is None:
            return output
        if tuple(output.shape[:2]) != (batch['input_ids'].shape[1], batch['input_ids'].shape[0]):
            raise RuntimeError('Expected MCore hidden states [sequence,batch,hidden]; refusing incorrect pooling')
        # Probability projections/loss in FP32, independent of BF16 backbone computation.
        return self.head(output, batch['picks'])


def build_model(model_path, kind, depth, max_length, seed=3407):
    from megatron.bridge import AutoBridge
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA, LoRALinearSplitQKV
    from megatron.core import parallel_state
    bridge = AutoBridge.from_hf_pretrained(str(model_path.resolve()), trust_remote_code=False,
                                          local_files_only=True)
    provider = bridge.to_megatron_provider(load_weights=True)
    overrides = dict(tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                     context_parallel_size=1, sequence_parallel=False, params_dtype=torch.bfloat16,
                     pipeline_dtype=torch.bfloat16, bf16=True, fp16=False,
                     transformer_impl='transformer_engine', seq_length=max_length,
                     gradient_accumulation_fusion=False, hidden_dropout=0.0, attention_dropout=0.0)
    for name, value in overrides.items():
        if not hasattr(provider, name):
            raise RuntimeError(f'NeMo/Bridge API differs: provider lacks {name}; no silent fallback')
        setattr(provider, name, value)
    provider.finalize()
    if not parallel_state.is_initialized():
        provider.initialize_model_parallel(seed=seed)
    models = provider.provide_distributed_model(wrap_with_ddp=False, mixed_precision_wrapper=None)
    if len(models) != 1:
        raise RuntimeError('Only single-stage data parallel is supported')
    backbone = models[0]
    if len(backbone.decoder.layers) != 36:
        raise ValueError('Expected the approved Qwen3-4B 36-layer backbone')
    if kind in ('binary', 'pointer'):
        cut_backbone(backbone, depth)
    elif depth != 36:
        raise ValueError('Generation baseline must retain all 36 layers')
    lora = CanonicalLoRA(target_modules=['linear_q', 'linear_v'], dim=8, alpha=32, dropout=.05)
    transformed = lora(backbone)
    if isinstance(transformed, list):
        if len(transformed) != 1:
            raise RuntimeError('Unexpected PEFT model list')
        transformed = transformed[0]
    backbone = transformed
    shim_count = fill_missing_key_adapter(backbone, LoRALinearSplitQKV)
    print(f'Native Q/V-only compatibility: zero K branches={shim_count}', flush=True)
    trainable = [(name, p) for name, p in backbone.named_parameters() if p.requires_grad]
    q = sum('adapter_q' in name for name, _ in trainable)
    v = sum('adapter_v' in name for name, _ in trainable)
    if not trainable or q != 2 * depth or v != 2 * depth:
        raise RuntimeError(f'Q/V LoRA did not match exactly {depth} layers: q={q}, v={v}')
    if any('adapter' not in name or 'adapter_k' in name for name, _ in trainable):
        raise RuntimeError('Unexpected unfrozen pretrained parameters')
    hidden = backbone.config.hidden_size
    head = BinaryHead(hidden) if kind == 'binary' else PointerHead(hidden, 256) if kind == 'pointer' else None
    model = DecisionModel(backbone, kind, head).cuda()
    model.train()
    print(f'Bridge model built: kind={kind} physical_layers={len(backbone.decoder.layers)} '
          f'trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}', flush=True)
    return model
