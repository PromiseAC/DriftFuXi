# Stage 2 AutoDL handoff

This is the first mandatory compute stop. Run these commands personally on the rented host; do not treat pilot output as the Stage 2 reproduction.

## Instance

- Use one NVIDIA RTX 4090 24 GB for the compatibility/cost pilot. Do not choose a Moore Threads/MUSA instance.
- Place the repository and data under `/root/autodl-tmp/DriftFuXi`; AutoDL documents this as the persistent data disk path. Stop the instance when idle because billing starts while it is running.
- Choose a CUDA 12.1 PyTorch image with Python 3.10 or 3.11 when available. The project creates an isolated environment and installs the paper repository's pinned PyTorch 2.2.2, so the image's bundled PyTorch is not used as experimental evidence.

Official references: [PyTorch 2.2.2 cu121 command](https://docs.pytorch.org/get-started/previous-versions/), [FBGEMM-GPU 0.6.0 package](https://pypi.org/project/fbgemm-gpu/0.6.0/), [AutoDL quick start](https://www.autodl.com/docs/quick_start/), and [AutoDL data persistence](https://www.autodl.com/docs/instance_data/).

## 1. Clone the prepared branch

```bash
cd /root/autodl-tmp
git clone --branch codex/stage2-reproduction https://github.com/PromiseAC/DriftFuXi.git
cd DriftFuXi
git rev-parse HEAD
```

The expected commit will be stated in the Stage 2 handoff message after the preparation changes are committed and pushed.

## 2. Create the isolated CUDA environment

```bash
python3 -m venv .venv-stage2
source .venv-stage2/bin/activate
python -m pip install pip==24.3.1
python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-stage2-cu121.txt
python -m pip check
```

This is a pinned minimal FuXi-Linear path. Do not install or upgrade the entire upstream requirements file: it also contains unrelated TensorFlow, TensorRT and optional-baseline packages.

## 3. Obtain and preprocess the public KuaiRec data

```bash
source .venv-stage2/bin/activate
python scripts/download_kuairec.py
MPLBACKEND=Agg python third_party/fuxi-linear/preprocess_kuairec_data.py
python scripts/data_stats.py tmp/processed/kuairec/sasrec_format.csv
```

The expected post-preprocessing counts are 7,176 users, 9,958 observed items and 12,529,113 interactions. Stop if hashes or counts differ; do not start training.

## 4. GPU preflight

```bash
source .venv-stage2/bin/activate
python scripts/check_stage2_gpu_env.py
```

Continue only when it prints `"status": "PASS"`. Send back `outputs/preflight/gpu_preflight.json` if it fails. This path is gitignored, so the audit runner still sees a clean source checkout.

## 5. One-epoch pilot

```bash
source .venv-stage2/bin/activate
python scripts/run_stage2_baseline.py --seed 42 --pilot-epochs 1
```

The pilot has its own `outputs/pilot/...` directory and metadata field `official_reproduction=false`. Use its wall time and memory only to estimate the full budget. Do not copy its NDCG/HR/MRR into `docs/reproduction_report.md`.

After the pilot, send back:

```text
outputs/pilot/kuairec/linear-4b-l1024-b64x2/epochs-1/seed-42/
```

Do not start the 101-epoch run until the pilot's compatibility, memory and estimated cost have been reviewed.

## 6. Full Stage 2 run (after pilot review)

```bash
source .venv-stage2/bin/activate
python scripts/run_stage2_baseline.py --seed 42
```

The wrapper streams output to the terminal and saves the same lines to `train.log`. A valid full run must contain config, metadata, command, epoch metrics, train log and a final checkpoint under:

```text
outputs/baseline/kuairec/linear-4b-l1024-b64x2/seed-42/
```

Archive the run before shutting down or resetting the instance:

```bash
tar -C outputs -czf /root/autodl-tmp/driftfuxi-stage2-seed42.tar.gz baseline/kuairec/linear-4b-l1024-b64x2/seed-42
```

Stage 2 remains incomplete until the real run is reviewed and the reproduction table is populated from its saved artifacts.
