"""Fail-fast check for the pinned Stage 2 Linux/CUDA environment.

Run this on the rented NVIDIA host before starting either the pilot or baseline.
It does not train a model or report recommendation metrics.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/fuxi-linear"


def package_version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/preflight/gpu_preflight.json",
    )
    args = parser.parse_args()
    failures = []
    report = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": {},
    }

    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        report["nvidia_smi"] = [line for line in smi.stdout.splitlines() if line]
    except Exception as exc:
        report["nvidia_smi_error"] = repr(exc)
        failures.append("nvidia-smi failed")

    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }
        if not torch.__version__.startswith("2.2.2"):
            failures.append(f"expected torch 2.2.2, found {torch.__version__}")
        if torch.version.cuda != "12.1":
            failures.append(f"expected PyTorch CUDA runtime 12.1, found {torch.version.cuda}")
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            failures.append("PyTorch cannot see an NVIDIA CUDA GPU")
        else:
            report["gpus"] = []
            for device in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(device)
                report["gpus"].append(
                    {
                        "index": device,
                        "name": props.name,
                        "compute_capability": list(torch.cuda.get_device_capability(device)),
                        "total_memory_bytes": props.total_memory,
                    }
                )
            probe = torch.ones((32, 32), device="cuda")
            probe = probe @ probe
            torch.cuda.synchronize()
            report["cuda_tensor_probe_sum"] = float(probe.sum().item())
    except Exception as exc:
        report["torch_error"] = repr(exc)
        failures.append("torch/CUDA probe failed")

    for distribution in [
        "absl-py",
        "einops",
        "fbgemm-gpu",
        "gin-config",
        "matplotlib",
        "numpy",
        "pandas",
        "seaborn",
        "tensorboard",
        "tqdm",
    ]:
        report["packages"][distribution] = package_version(distribution)

    try:
        import fbgemm_gpu  # noqa: F401
        import torch

        required_ops = [
            "asynchronous_complete_cumsum",
            "dense_to_jagged",
            "jagged_to_padded_dense",
        ]
        report["fbgemm_ops"] = {
            op: hasattr(torch.ops.fbgemm, op) for op in required_ops
        }
        if report["packages"]["fbgemm-gpu"] != "0.6.0":
            failures.append(
                f"expected fbgemm-gpu 0.6.0, found {report['packages']['fbgemm-gpu']}"
            )
        missing_ops = [op for op, present in report["fbgemm_ops"].items() if not present]
        if missing_ops:
            failures.append(f"missing FBGEMM operators: {missing_ops}")
    except Exception as exc:
        report["fbgemm_import_error"] = repr(exc)
        failures.append("fbgemm_gpu import failed")

    sys.path.insert(0, str(UPSTREAM))
    try:
        from generative_recommenders.trainer.train import train_fn  # noqa: F401
        from generative_recommenders.modeling.sequential.fuxi_linear import FuXiLinear  # noqa: F401

        report["fuxi_import"] = True
    except Exception as exc:
        report["fuxi_import"] = False
        report["fuxi_import_error"] = repr(exc)
        failures.append("FuXi training imports failed")

    required_files = {
        "raw": ROOT / "tmp/kuairec/big_matrix.csv",
        "processed": ROOT / "tmp/processed/kuairec/sasrec_format.csv",
        "config": UPSTREAM / "configs/kuairec/linear-4b-l1024-b64x2.gin",
    }
    report["files"] = {
        name: {
            "path": str(path),
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
        }
        for name, path in required_files.items()
    }
    for name, info in report["files"].items():
        if not info["exists"]:
            failures.append(f"missing {name} file: {info['path']}")

    report["environment"] = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    report["failures"] = failures
    report["status"] = "PASS" if not failures else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
