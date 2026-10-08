"""Read-only runtime/import inventory; no package installation or data download."""
import ast
import importlib
import importlib.metadata as md
import importlib.util
import json
import platform
from pathlib import Path
import subprocess
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
source = ROOT/'third_party/fuxi-linear'
imports = {}
for path in source.rglob('*.py'):
    for node in ast.walk(ast.parse(path.read_text())):
        names = [n.name for n in node.names] if isinstance(node,ast.Import) else ([node.module] if isinstance(node,ast.ImportFrom) and node.level==0 and node.module else [])
        for name in names:
            top = name.split('.')[0]
            if top not in sys.stdlib_module_names and top != 'generative_recommenders':
                imports.setdefault(top,set()).add(str(path.relative_to(source)))
packages = {}
modules = {'gin-config':'gin','fbgemm-gpu':'fbgemm_gpu','apex':'apex','iopath':'iopath','numpy':'numpy','pandas':'pandas','scipy':'scipy','einops':'einops','tqdm':'tqdm','tensorboard':'tensorboard','mamba-ssm':'mamba_ssm','transformers':'transformers','matplotlib':'matplotlib','seaborn':'seaborn','absl-py':'absl','torch':'torch'}
for dist, module in modules.items():
    try: version = md.version(dist)
    except md.PackageNotFoundError: version = None
    try:
        importlib.import_module(module)
        status = 'PASS'
    except Exception as exc: status = f'{type(exc).__name__}: {exc}'
    packages[dist] = {'version':version,'actual_import':status}
result = {'os':platform.platform(),'macos':platform.mac_ver()[0],'machine':platform.machine(),'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'cuda_device_count':torch.cuda.device_count(),'compute_capability':torch.cuda.get_device_capability() if torch.cuda.is_available() else None,'mps_built':torch.backends.mps.is_built(),'mps_available':torch.backends.mps.is_available(),'rms_norm_available':hasattr(torch.nn.functional,'rms_norm'),'packages':packages,'source_external_imports':{k:sorted(v) for k,v in sorted(imports.items())}}
print(json.dumps(result,indent=2))
