#!/usr/bin/env bash
# Setup environment for multilingual-safety-guard-runner
# Usage: bash setup_env.sh
set -euo pipefail

echo '=== Detecting CUDA driver ==='
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo '')
echo "Driver: ${DRIVER:-unknown}"

TORCH_INDEX='cpu'
if command -v nvidia-smi >/dev/null 2>&1; then
  MAJOR=$(echo "${DRIVER}" | cut -d. -f1)
  if [ "${MAJOR}" -ge 538 ]; then
    TORCH_INDEX='cu130'
  else
    TORCH_INDEX='cu126'
  fi
fi
echo "Torch index: ${TORCH_INDEX}"

echo '=== Creating venv ==='
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

echo '=== Installing requirements ==='
pip install -r requirements.txt

echo '=== Installing torch matching driver ==='
if [ "${TORCH_INDEX}" != 'cpu' ]; then
  pip install --force-reinstall torch --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"
fi

echo '=== Verify ==='
.venv/bin/python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())'
.venv/bin/python -c 'import transformers, peft, accelerate; print("transformers", transformers.__version__, "peft", peft.__version__, "acc", accelerate.__version__)'
echo '=== Setup complete ==='
