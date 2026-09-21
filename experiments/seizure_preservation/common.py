"""Small experiment-specific I/O and reproducibility helpers."""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import os
import random
from pathlib import Path
import numpy as np


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows, fields=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def derived_seed(seed, *parts):
    digest = hashlib.sha256('|'.join(map(str, (seed, *parts))).encode()).digest()
    return int.from_bytes(digest[:4], 'little')


def seed_everything(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def import_source(path, name):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Reused source not found: {path}; configure its path explicitly')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
