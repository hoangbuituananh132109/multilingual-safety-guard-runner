# A30 distributed LoRA training

The `--gpus` launcher is for GPUs visible on one node. Before training, inspect
the allocation:

```bash
hostname
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,mig.mode.current --format=csv
nvidia-smi topo -m
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
```

`--gpus` means GPUs per node. If four GPUs span two nodes with two GPUs each,
run the same command on both nodes with `--gpus 2 --nnodes 2`, a different
`--node-rank`, and the same master address/port.
If the allocation exposes 6 GB or 12 GB MIG slices, the current BF16 LoRA
trainer is not suitable for Qwen3-8B; use full 24 GB A30 GPUs or add a sharded
FSDP/ZeRO/QLoRA configuration.

## Batch invariant

The original one-GPU configuration uses an effective batch of 32:

```text
4 samples/GPU x 1 GPU x 8 accumulation steps = 32
```

For four GPUs, preserve the same optimizer-step semantics with:

```text
1 sample/GPU x 4 GPUs x 8 accumulation steps = 32
```

## Four-GPU smoke tests

```bash
python3 run.py train --model qwen3_8b --method lora --mode smoke --gpus 4 --per-device-batch-size 1 --gradient-accumulation-steps 8
python3 run.py train --model qwen3_4b --method lora --mode smoke --gpus 4 --per-device-batch-size 1 --gradient-accumulation-steps 8
python3 run.py train --model qwen3_1_7b --method lora --mode smoke --gpus 4 --per-device-batch-size 1 --gradient-accumulation-steps 8
python3 run.py train_qwen35 --model qwen35_4b --method lora --mode smoke --gpus 4 --per-device-batch-size 1 --gradient-accumulation-steps 8
```

For a two-node group with two GPUs per node, run this on node 0:

```bash
python3 run.py train --model qwen3_8b --method lora --mode smoke --gpus 2 --nnodes 2 --node-rank 0 --master-addr NODE0_IP --master-port 29500 --per-device-batch-size 1 --gradient-accumulation-steps 8
```

Run the same job on node 1, changing only the rank:

```bash
python3 run.py train --model qwen3_8b --method lora --mode smoke --gpus 2 --nnodes 2 --node-rank 1 --master-addr NODE0_IP --master-port 29500 --per-device-batch-size 1 --gradient-accumulation-steps 8
```

## Full runs

Resume Qwen3-8B from the numerically latest checkpoint:

```bash
python3 run.py train --model qwen3_8b --method lora --mode full --resume --gpus 4 --per-device-batch-size 1 --gradient-accumulation-steps 8
```

Start the other models from their base weights:

```bash
python3 run.py train --model qwen3_4b --method lora --mode full --gpus 4 --per-device-batch-size 1 --gradient-accumulation-steps 8
python3 run.py train --model qwen3_1_7b --method lora --mode full --gpus 4 --per-device-batch-size 1 --gradient-accumulation-steps 8
python3 run.py train_qwen35 --model qwen35_4b --method lora --mode full --gpus 4 --per-device-batch-size 1 --gradient-accumulation-steps 8
```

Changing from one-GPU to four-GPU DDP is a valid continuation when the global
batch remains 32, but it is not bit-for-bit reproducible because data sharding
and per-rank RNG state change.
