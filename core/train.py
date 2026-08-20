from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from prompt import N23, render_prompt


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[train] {stamp} {message}", flush=True)


def vram_mb() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    return f"alloc={allocated:.0f}MB reserved={reserved:.0f}MB"


class StepLogger(TrainerCallback):
    """Log EVERY optimizer step with elapsed time and VRAM usage.

    Uses on_step_end so output appears even when logging_steps is large
    (otherwise the first line only prints after logging_steps steps).
    """

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


def target_text(row: dict[str, Any], family: str) -> str:
    label = str(row["safety_label"])
    if family == "qwen_binary":
        return "Safety: " + ("Unsafe" if label == "unsafe" else "Safe")
    payload = {"User Safety": str(row["prompt_safety_label"])}
    if str(row["view"]) == "PR":
        payload["Response Safety"] = label
    categories = [name for name in N23 if name in set(row.get("categories") or [])]
    if categories:
        payload["Safety Categories"] = ", ".join(categories)
    return json.dumps(payload, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--skip-eval", action="store_true", help="Do not run eval during training (default: skip eval to avoid hanging on large valid set)")
    parser.add_argument("--no-checkpoints", action="store_true", help="Only save the final model; skip periodic checkpoints to save disk/I/O")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_cfg, data_cfg, train_cfg = cfg["model"], cfg["data"], cfg["training"]
    log(f"config loaded from {args.config}")
    log(f"model_id={model_cfg['id']} tuning={model_cfg['tuning']} family={model_cfg['family']} attention={model_cfg.get('attention', 'sdpa')}")
    log(f"train={data_cfg['train']} validation={data_cfg['validation']} max_length={data_cfg['max_length']}")
    log(f"epochs={train_cfg['epochs']} lr={train_cfg['learning_rate']} batch={train_cfg['per_device_batch_size']} grad_accum={train_cfg['gradient_accumulation_steps']} max_steps={args.max_steps}")
    if model_cfg["tuning"] == "qlora" and os.environ.get("ACCELERATE_USE_DEEPSPEED") == "true":
        raise ValueError("QLoRA is intentionally restricted to single GPU/DDP here; use LoRA/full for DeepSpeed ZeRO.")

    log("loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.get("tokenizer_id", model_cfg["id"]),
        revision=model_cfg.get("revision", "main"),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    log(f"tokenizer loaded ({type(tokenizer).__name__}), vocab={getattr(tokenizer, 'vocab_size', '?')}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tuning = model_cfg["tuning"]
    quant = None
    if tuning == "qlora":
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    load_kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation=model_cfg.get("attention", "sdpa"), trust_remote_code=model_cfg.get("trust_remote_code", False), quantization_config=quant)
    if quant is not None:
        load_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    log(f"loading base model from {model_cfg['id']} (dtype=bf16, attention={load_kwargs['attn_implementation']})...")
    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["id"], revision=model_cfg.get("revision", "main"), **load_kwargs
    )
    log(f"base model loaded in {time.time() - load_start:.1f}s, {vram_mb()}")
    model.config.use_cache = False
    if tuning == "qlora":
        log("preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=train_cfg.get("gradient_checkpointing", True))
    if tuning in {"lora", "qlora"}:
        log(f"wrapping model with LoRA (r={model_cfg['lora_r']}, alpha={model_cfg['lora_alpha']}, dropout={model_cfg['lora_dropout']}, modules={model_cfg['target_modules']})...")
        model = get_peft_model(model, LoraConfig(r=int(model_cfg["lora_r"]), lora_alpha=int(model_cfg["lora_alpha"]), lora_dropout=float(model_cfg["lora_dropout"]), target_modules=list(model_cfg["target_modules"]), bias="none", task_type="CAUSAL_LM"))
        log(f"LoRA applied, {vram_mb()}")

    log("loading dataset...")
    raw = load_dataset("json", data_files={"train": data_cfg["train"], "validation": data_cfg["validation"]})
    log(f"dataset loaded: train={len(raw['train'])} validation={len(raw['validation'])}")
    allowed = {"P", "PR"}
    if any(view not in allowed for view in set(raw["train"]["view"])):
        raise ValueError("Clean contract violation: only P/PR are allowed; response-only R is forbidden.")
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
    tokenize_start = time.time()
    tokenized = raw.map(tokenize, remove_columns=raw["train"].column_names, num_proc=max(1, min(8, os.cpu_count() or 1)), desc="Tokenizing completion-only safety targets")
    log(f"tokenization done in {time.time() - tokenize_start:.1f}s")
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    parameter_report = {
        "method": tuning,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_percent": 100.0 * trainable_parameters / total_parameters,
    }
    (output / "trainable_parameters.json").write_text(json.dumps(parameter_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(parameter_report, indent=2))
    log(f"trainable params: {trainable_parameters:,} ({parameter_report['trainable_percent']:.4f}%)")
    import inspect as _inspect
    from transformers import TrainingArguments as _TA
    _sig_params = set(_inspect.signature(_TA.__init__).parameters)
    _ta_kwargs = dict(
        output_dir=str(output), run_name=cfg["run_name"], num_train_epochs=float(train_cfg["epochs"]), max_steps=args.max_steps,
        per_device_train_batch_size=int(train_cfg["per_device_batch_size"]), per_device_eval_batch_size=int(train_cfg["per_device_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]), learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=train_cfg["lr_scheduler_type"], lr_scheduler_kwargs={"warmup_ratio": float(train_cfg["warmup_ratio"])}, bf16=True, tf32=True,
        gradient_checkpointing=bool(train_cfg["gradient_checkpointing"]), gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=int(train_cfg["logging_steps"]), save_steps=int(train_cfg["save_steps"]), eval_steps=int(train_cfg["eval_steps"]),
        eval_strategy="no" if args.skip_eval else "steps", save_strategy="no" if args.no_checkpoints else "steps", save_total_limit=int(train_cfg["save_total_limit"]), load_best_model_at_end=False,
        report_to=[], seed=int(train_cfg["seed"]), data_seed=int(train_cfg["seed"]), ddp_find_unused_parameters=False,
        remove_unused_columns=False, label_names=["labels"], save_safetensors=True,
    )
    _ta_kwargs = {k: v for k, v in _ta_kwargs.items() if k in _sig_params}
    training_args = _TA(**_ta_kwargs)
    log("building Trainer...")
    eval_dataset = None if args.skip_eval else tokenized["validation"]
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized["train"], eval_dataset=eval_dataset, data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8))
    trainer.add_callback(StepLogger())
    resume = True if args.resume == "auto" else args.resume
    log(f"starting training (resume={resume}, eval={'skipped' if args.skip_eval else 'enabled'})...")
    train_start = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    log(f"training finished in {time.time() - train_start:.1f}s")
    log("saving model...")
    trainer.save_model(str(output / "final"))
    tokenizer.save_pretrained(output / "final")
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    log(f"model saved to {output / 'final'}")


if __name__ == "__main__":
    main()
