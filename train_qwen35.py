#!/usr/bin/env python3
"""Standalone LoRA/full trainer for multimodal Qwen3.5-4B (text-only).

Trains only the language-model backbone on text safety data. Vision encoder
and projector (if present) are frozen and never activated because inputs are
pure text (no image tokens). Does NOT touch core/train.py.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from accelerate.state import PartialState
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments

from core.prompt import N23, render_prompt
from core.train import configure_vram_limit, log, sync_resume_intervals, target_text, vram_mb


class StepLogger(TrainerCallback):
    def __init__(self) -> None:
        self.last_step = 0
        self.step_start = time.time()

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step != self.last_step:
            self.last_step = state.global_step
            self.step_start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.step_start
        log(f"STEP {state.global_step} done in {elapsed:.1f}s {vram_mb()}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        elapsed = time.time() - self.step_start
        log(f"step={state.global_step} elapsed_since_last={elapsed:.1f}s {vram_mb()} logs={json.dumps(logs)}")


def load_model_and_tokenizer(model_path: str):
    """Load a multimodal-or-text causal LM, freezing vision parts if present."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    log(f"loading model from {model_path} ...")
    load_start = time.time()
    model = None
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True
        )
        log("loaded via AutoModelForImageTextToText")
    except Exception as exc:  # noqa: BLE001
        log(f"AutoModelForImageTextToText failed ({exc}); falling back to AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True
        )
        log("loaded via AutoModelForCausalLM")
    model.config.use_cache = False

    # Freeze vision encoder + projector if present.
    frozen = []
    for attr in ("vision_tower", "visual", "vision_model", "image_encoder", "multi_modal_projector", "merger"):
        obj = getattr(model, attr, None)
        if obj is not None:
            for param in obj.parameters():
                param.requires_grad = False
            frozen.append(attr)
    if frozen:
        log(f"froze vision components: {frozen}")
    else:
        log("no vision component detected; treating as text-only model")

    log(f"model loaded in {time.time()-load_start:.1f}s, {vram_mb()}")
    return model, tokenizer, bool(frozen)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--no-checkpoints", action="store_true", help="Disable periodic checkpoints for disposable smoke/probe runs")
    parser.add_argument("--no-final-save", action="store_true", help="Skip the final model export; intended only for disposable capacity probes")
    parser.add_argument("--vram-fraction", type=float, help="Cap the PyTorch allocator to a fraction of each visible GPU")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", "-1")))
    args = parser.parse_args()
    distributed_state = PartialState()
    configure_vram_limit(distributed_state, args.vram_fraction)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_cfg, data_cfg, train_cfg = cfg["model"], cfg["data"], cfg["training"]

    log(f"config loaded from {args.config}")
    log(f"model_id={model_cfg['id']} tuning={model_cfg['tuning']} family={model_cfg['family']}")
    log(f"train={data_cfg['train']} validation={data_cfg['validation']} max_length={data_cfg['max_length']}")
    log(f"epochs={train_cfg['epochs']} lr={train_cfg['learning_rate']} batch={train_cfg['per_device_batch_size']} grad_accum={train_cfg['gradient_accumulation_steps']} max_steps={args.max_steps}")

    model, tokenizer, has_frozen_vision = load_model_and_tokenizer(model_cfg["id"])

    if model_cfg["tuning"] == "lora":
        log(f"wrapping with LoRA (r={model_cfg['lora_r']}, alpha={model_cfg['lora_alpha']}, dropout={model_cfg['lora_dropout']})...")
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(model_cfg["lora_r"]),
                lora_alpha=int(model_cfg["lora_alpha"]),
                lora_dropout=float(model_cfg["lora_dropout"]),
                target_modules=list(model_cfg["target_modules"]),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        log(f"LoRA applied, {vram_mb()}")
    elif model_cfg["tuning"] != "full":
        raise ValueError(f"Unsupported Qwen3.5 tuning method: {model_cfg['tuning']}")

    log("loading dataset...")
    with distributed_state.local_main_process_first():
        raw = load_dataset("json", data_files={"train": data_cfg["train"], "validation": data_cfg["validation"]})
    log(f"dataset loaded: train={len(raw['train'])} validation={len(raw['validation'])}")
    effective_batch = int(train_cfg["per_device_batch_size"]) * int(train_cfg["gradient_accumulation_steps"]) * distributed_state.num_processes
    update_steps = (len(raw["train"]) + effective_batch - 1) // effective_batch
    log(f"world_size={distributed_state.num_processes} effective_global_batch={effective_batch} estimated_updates_per_epoch={update_steps}")
    max_length = int(data_cfg["max_length"])

    def tokenize(row: dict[str, Any]) -> dict[str, Any]:
        prompt = render_prompt(tokenizer, model_cfg["family"], str(row["prompt"]), row.get("response"))
        target = target_text(row, model_cfg["family"])
        target_ids = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        budget = max_length - len(target_ids)
        if len(prompt_ids) > budget:
            head = budget // 2
            prompt_ids = prompt_ids[:head] + prompt_ids[-(budget - head):]
        ids = prompt_ids + target_ids
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(prompt_ids) + target_ids}

    log("tokenizing dataset (may take a few minutes)...")
    tok_start = time.time()
    with distributed_state.local_main_process_first():
        tokenized = raw.map(tokenize, remove_columns=raw["train"].column_names, num_proc=max(1, min(8, os.cpu_count() or 1)), desc="Tokenizing")
    log(f"tokenization done in {time.time()-tok_start:.1f}s")

    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report = {"method": model_cfg["tuning"], "total_parameters": total, "trainable_parameters": trainable, "trainable_percent": 100.0 * trainable / total}
    if distributed_state.is_main_process:
        (output / "trainable_parameters.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(f"trainable params: {trainable:,} ({report['trainable_percent']:.4f}%)")

    import inspect as _inspect
    from transformers import TrainingArguments as _TA
    _sig = set(_inspect.signature(_TA.__init__).parameters)
    _ta = dict(
        output_dir=str(output), run_name=cfg["run_name"], num_train_epochs=float(train_cfg["epochs"]), max_steps=args.max_steps,
        per_device_train_batch_size=int(train_cfg["per_device_batch_size"]), per_device_eval_batch_size=int(train_cfg["per_device_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]), learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=train_cfg["lr_scheduler_type"], lr_scheduler_kwargs={"warmup_ratio": float(train_cfg["warmup_ratio"])}, bf16=True, tf32=True,
        gradient_checkpointing=bool(train_cfg["gradient_checkpointing"]), gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=int(train_cfg["logging_steps"]), save_steps=int(train_cfg["save_steps"]), eval_steps=int(train_cfg["eval_steps"]),
        eval_strategy="no", save_strategy="no" if args.no_checkpoints else "steps", save_total_limit=int(train_cfg["save_total_limit"]), load_best_model_at_end=False,
        report_to=[], seed=int(train_cfg["seed"]), data_seed=int(train_cfg["seed"]), ddp_find_unused_parameters=has_frozen_vision,
        remove_unused_columns=False, label_names=["labels"], save_safetensors=True,
    )
    if train_cfg.get("optim"):
        _ta["optim"] = str(train_cfg["optim"])
    _ta = {k: v for k, v in _ta.items() if k in _sig}
    training_args = _TA(**_ta)
    log("building Trainer...")
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=tokenized["train"], eval_dataset=None,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8),
    )
    if trainer.is_world_process_zero():
        trainer.add_callback(StepLogger())
        resume = sync_resume_intervals(output, args.resume, train_cfg)
    else:
        resume = None
    distributed_state.wait_for_everyone()
    if not trainer.is_world_process_zero():
        resume = sync_resume_intervals(output, args.resume, train_cfg)
    log(f"starting training (resume={resume}, eval=skipped)...")
    start = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    log(f"training finished in {time.time()-start:.1f}s")
    if args.no_final_save:
        log("skipping final model export (--no-final-save)")
    else:
        log("saving model...")
        trainer.save_model(str(output / "final"))
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(output / "final")
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    distributed_state.wait_for_everyone()
    if not args.no_final_save:
        log(f"model saved to {output / 'final'}")


if __name__ == "__main__":
    main()
