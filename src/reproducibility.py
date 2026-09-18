"""Seed configuration and aggregate-only runtime provenance."""

import os
import platform
import random

import numpy as np
import sklearn
import torch


def seed_everything(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Use the DataLoader's seeded generator for each worker process."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are auto, cpu, or cuda[:index].")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; set device='cpu'.")
    return resolved


def runtime_metadata() -> dict:
    return {
        "python_version": platform.python_version(),
        "numpy_version": str(np.__version__),
        "sklearn_version": str(sklearn.__version__),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "deterministic_algorithms_requested": torch.are_deterministic_algorithms_enabled(),
        "determinism_scope": "Best effort within a fixed device and software environment; unsupported operations warn.",
    }
