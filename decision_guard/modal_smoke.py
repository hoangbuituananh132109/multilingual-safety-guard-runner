"""Bounded Modal integration smoke, fixed NeMo image; never full training.

Use MODAL_PROFILE=pippa3. Reuses the existing model cache read-only.
"""
from pathlib import Path
import modal

app = modal.App('decision-guard-nemo2608-one-step')
model_volume = modal.Volume.from_name('guard-decision-lab-20261010')
study_volume = modal.Volume.from_name('guard-bridge-smoke-20261011', create_if_missing=True)
image = (modal.Image.from_registry('nvcr.io/nvidia/nemo:26.08')
         .env({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
               'HF_HUB_DISABLE_TELEMETRY': '1', 'WANDB_MODE': 'disabled',
               'DG_IMAGE': 'nvcr.io/nvidia/nemo:26.08', 'PYTHONUNBUFFERED': '1',
               'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false'})
         .add_local_dir(Path(__file__).parent, '/workspace/decision_guard',
                        ignore=['**/__pycache__/**', '**/verification/**']))
CACHED_MODEL = '/models/hf/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c'
MODEL = '/study/decision-guard-one-step-v1/model'


@app.function(image=image, volumes={'/models': model_volume, '/study': study_volume},
              cpu=4, memory=16384, timeout=1800, retries=0, max_containers=1)
def prepare():
    import importlib.metadata
    import json
    import os
    import subprocess
    import sys
    import shutil
    import tempfile
    sys.path.insert(0, '/workspace')
    os.chdir('/workspace')
    from decision_guard.preflight import inspect_model
    from decision_guard.data import install, prepare as prepare_cache
    versions = {}
    for name in ('torch', 'transformers', 'megatron-bridge', 'megatron-core', 'transformer-engine'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'source checkout'
    print(json.dumps({'phase': 'CPU_IMAGE_CHECK', 'versions': versions,
                      'python': sys.version, 'image': 'nvcr.io/nvidia/nemo:26.08'}, indent=2), flush=True)
    print('Bridge/TE API probe deferred to GPU preflight: native TE requires libcuda.so.1', flush=True)
    root = Path('/study/decision-guard-one-step-v1')
    root.mkdir(parents=True, exist_ok=True)
    if not Path(MODEL).exists():
        source = Path(CACHED_MODEL)
        cache_repo = source.parents[1].resolve()
        # HF snapshots point at sibling blobs. Preserve the original cache and
        # materialize an ordinary local model folder, rather than relax preflight.
        for item in source.iterdir():
            resolved = item.resolve()
            print(f'Cached file {item.name}: symlink={item.is_symlink()} exists={resolved.is_file()} '
                  f'target={resolved} allowed_root={cache_repo}', flush=True)
            if not resolved.is_file() or cache_repo not in resolved.parents:
                raise ValueError(f'Unsafe/missing cache target: {item.name}')
        temporary = Path(tempfile.mkdtemp(prefix='model-stage-', dir=root))
        shutil.copytree(source, temporary, symlinks=False, dirs_exist_ok=True)
        temporary.rename(MODEL)
    receipt = inspect_model(Path(MODEL))
    (root / 'cpu_model_receipt.json').write_text(json.dumps(receipt, indent=2))
    print(install('/study/input/nemotron_wildguard_scaled_max_70_30_v2.gated.zip', root / 'data'))
    print(prepare_cache(root / 'data', MODEL, root / 'cache', 2048, 128, 16))
    study_volume.commit()
    return {'root': str(root), 'versions': versions, 'model_sha256': receipt['model_sha256']}


@app.function(image=image, gpu='L40S', volumes={'/models': model_volume, '/study': study_volume},
              cpu=8, memory=32768, timeout=1800, retries=0, max_containers=1,
              scaledown_window=2)
def smoke():
    import json
    import os
    import subprocess
    import sys
    import time
    os.chdir('/workspace')
    sys.path.insert(0, '/workspace')
    root = Path('/study/decision-guard-one-step-v1')
    if not (root / 'cache/cache_manifest.json').is_file():
        raise FileNotFoundError('CPU preparation must pass first')
    output = root / 'pointer36'
    output.mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTHONPATH'] = '/workspace'
    print(subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                  text=True), flush=True)
    commands = [
        [sys.executable, '-m', 'decision_guard.preflight', '--model', MODEL,
         '--output', str(root / 'model_receipt.json')],
        [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1',
         '-m', 'decision_guard.train', '--model', MODEL, '--model-receipt', str(root / 'model_receipt.json'),
         '--cache', str(root / 'cache'), '--arm', 'pointer36', '--output', str(output),
         '--global-batch', '32', '--micro-batch', '1', '--save-every', '1', '--smoke']]
    started = time.monotonic()
    try:
        for index, command in enumerate(commands):
            with (output / f'phase-{index}.log').open('a') as log:
                print('EXEC', command, flush=True)
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1)
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                if process.wait() != 0:
                    raise RuntimeError(f'Smoke phase {index} failed with exit code {process.returncode}')
        result = json.loads((output / 'smoke_pass.json').read_text())
        if result['steps'] != 1 or result['checkpoint_roundtrip']['passed'] is not True:
            raise RuntimeError('Missing one-step saved-and-reloaded checkpoint evidence')
        result['modal_gpu_seconds'] = round(time.monotonic() - started, 2)
        result['fixed_image'] = 'nvcr.io/nvidia/nemo:26.08'
        (output / 'modal_result.json').write_text(json.dumps(result, indent=2))
        return result
    finally:
        study_volume.commit()


@app.local_entrypoint()
def main(phase: str = 'prepare'):
    import json
    if phase not in ('prepare', 'smoke'):
        raise ValueError('Only CPU prepare or one-step GPU smoke is allowed')
    if phase == 'prepare':
        archive = Path(__file__).resolve().parents[1] / 'input/decision-guard-70-30/hf-download/nemotron_wildguard_scaled_max_70_30_v2.gated.zip'
        try:
            present = any(Path(item.path).name == archive.name for item in study_volume.iterdir('/input'))
        except FileNotFoundError:
            present = False
        if not present:
            with study_volume.batch_upload() as upload:
                upload.put_file(archive, '/input/' + archive.name)
        result = prepare.remote()
    else:
        result = smoke.remote()
    output = Path(__file__).parent / 'verification' / f'modal-{phase}-result.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
