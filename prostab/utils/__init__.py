# Adapted from SPURS and ByProt (see README acknowledgements).
"""Lightweight utilities; training helpers are loaded only when requested."""
import glob
import importlib
import logging
import os
import random

import numpy as np
import torch


def get_logger(name=__name__):
    return logging.getLogger(name)


log = get_logger(__name__)

def import_modules(models_dir, namespace, excludes=()):
    for path in glob.glob(models_dir + '/**', recursive=True)[1:]:
        if any(e in path for e in excludes):
            continue

        file = os.path.split(path)[1]
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            module_name = file[: file.find(".py")] if file.endswith(".py") else file

            _namespace = path.replace('/', '.')
            _namespace = _namespace[_namespace.find(namespace): _namespace.rfind('.' + module_name)]
            importlib.import_module(_namespace + "." + module_name)


def seed_everything(seed, verbose=False) -> int:
    """Seed Python, NumPy and Torch using the original inference convention."""
    if verbose:
        log.info(f"Random seed set to {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


def recursive_to(obj, device):
    """Move nested tensors without selecting an implicit CUDA device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: recursive_to(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [recursive_to(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(recursive_to(value, device) for value in obj)
    return obj


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(name)
    training = importlib.import_module("prostab.utils.training")
    try:
        return getattr(training, name)
    except AttributeError:
        raise AttributeError(name) from None
