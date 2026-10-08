"""Launch the pinned official FuXi-Linear KuaiRec run with an auditable envelope.

The official main/train/evaluator implement all modeling and data protocol logic.
Run output is local and gitignored. Use --prepare-only without CUDA.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party/fuxi-linear'
CONFIG = UPSTREAM / 'configs/kuairec/linear-4b-l1024-b64x2.gin'
PROCESSED = ROOT / 'tmp/processed/kuairec/sasrec_format.csv'
RAW = ROOT / 'tmp/kuairec/big_matrix.csv'


def digest(path):
    h = sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git(args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('Seed must be nonnegative')
    run_dir = ROOT / 'outputs/baseline/kuairec/linear-4b-l1024-b64x2' / f'seed-{args.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / 'epoch_metrics.jsonl').exists() or (run_dir / 'checkpoint.pt').exists():
        parser.error(f'Run already has results; refusing to overwrite: {run_dir}')
    config_text = CONFIG.read_text()
    if 'train_fn.random_seed' in config_text:
        parser.error('Official config now sets a seed; inspect before proceeding')
    config_text += f'\ntrain_fn.random_seed = {args.seed}\n'
    (run_dir/'config.gin').write_text(config_text)
    commit = git(['rev-parse','HEAD'])
    dirty = git(['status','--porcelain'])
    if dirty and not args.prepare_only:
        parser.error('Commit the Stage 2 runner/source changes before a real baseline run')
    data = {}
    for name, path in [('raw',RAW),('processed',PROCESSED)]:
        data[name] = {'exists':path.is_file(),'sha256':digest(path) if path.is_file() else None,
                      'bytes':path.stat().st_size if path.is_file() else None}
    metadata = {
        'status':'PREPARED' if args.prepare_only else 'STARTING',
        'dataset':'kuairec', 'config':'linear-4b-l1024-b64x2.gin',
        'seed':args.seed, 'code_commit':commit, 'git_dirty_at_prepare':bool(dirty),
        'config_sha256':digest(run_dir/'config.gin'),
        'data':data, 'created_utc':datetime.now(timezone.utc).isoformat(),
        'host_os':platform.platform(), 'python':sys.version.split()[0],
        'python_executable':sys.executable,
    }
    launch = [sys.executable, '-u', str(UPSTREAM/'main.py'),
              f'--gin_config_file={run_dir/"config.gin"}', '--master_port=12345']
    (run_dir/'command.txt').write_text('PYTHONPATH=<project>/third_party/fuxi-linear '
                                       + ' '.join(['python3','-u','third_party/fuxi-linear/main.py',
                                                   '--gin_config_file=outputs/baseline/kuairec/linear-4b-l1024-b64x2/'
                                                   + f'seed-{args.seed}/config.gin','--master_port=12345']) + '\n')
    if args.prepare_only:
        save_json(run_dir/'run_metadata.json',metadata)
        print('Prepared:',run_dir)
        print('Raw data ready:',data['raw']['exists'],'Processed data ready:',data['processed']['exists'])
        return
    if not RAW.is_file() or not PROCESSED.is_file():
        parser.error('Official KuaiRec raw and processed data must be ready before training')
    import torch
    if not torch.cuda.is_available():
        parser.error('Official main.py requires NVIDIA CUDA; no CUDA GPU visible')
    try:
        import fbgemm_gpu  # noqa: F401
    except Exception as exc:
        parser.error(f'fbgemm_gpu import failed: {exc}')
    try: metadata['fbgemm_gpu_version'] = md.version('fbgemm-gpu')
    except md.PackageNotFoundError: metadata['fbgemm_gpu_version'] = 'unavailable'
    metadata.update({'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
        'gpus':[{'name':torch.cuda.get_device_name(i),
                 'compute_capability':list(torch.cuda.get_device_capability(i)),
                 'total_memory_bytes':torch.cuda.get_device_properties(i).total_memory}
                for i in range(torch.cuda.device_count())],
        'visible_cuda_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
    })
    tmp_link = run_dir/'tmp'
    if not tmp_link.exists():
        tmp_link.symlink_to(ROOT/'tmp',target_is_directory=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(UPSTREAM) + os.pathsep + env.get('PYTHONPATH','')
    env['PYTHONHASHSEED'] = str(args.seed)
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    env['DRIFTFUXI_RUN_DIR'] = str(run_dir)
    metadata['started_utc'] = datetime.now(timezone.utc).isoformat()
    metadata['status'] = 'RUNNING'
    save_json(run_dir/'run_metadata.json',metadata)
    start = time.perf_counter()
    with (run_dir/'train.log').open('w') as log:
        proc = subprocess.run(launch,cwd=run_dir,env=env,stdout=log,stderr=subprocess.STDOUT)
    metadata['wall_clock_seconds'] = time.perf_counter() - start
    metadata['finished_utc'] = datetime.now(timezone.utc).isoformat()
    metadata['exit_code'] = proc.returncode
    metadata['status'] = 'FINISHED' if proc.returncode == 0 else 'FAILED'
    if proc.returncode == 0:
        checkpoints = sorted((run_dir/'ckpts').rglob('*_ep*'),key=lambda x:x.stat().st_mtime)
        if checkpoints:
            final = run_dir/'checkpoint.pt'
            try: os.link(checkpoints[-1],final)
            except OSError: shutil.copy2(checkpoints[-1],final)
            metadata['checkpoint_relative_path'] = str(final.relative_to(run_dir))
            metadata['checkpoint_sha256'] = digest(final)
        else:
            metadata['status'] = 'FAILED_NO_CHECKPOINT'
    save_json(run_dir/'run_metadata.json',metadata)
    print(metadata['status'], 'wall_seconds=',metadata['wall_clock_seconds'],'output=',run_dir)
    if proc.returncode:
        raise SystemExit(proc.returncode)
    if metadata['status'] != 'FINISHED':
        raise SystemExit(1)

if __name__ == '__main__':
    main()
