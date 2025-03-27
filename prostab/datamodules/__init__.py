import importlib
from omegaconf import DictConfig
import os
import glob

from prostab.utils import import_modules

DATAMODULE_REGISTRY = {}


def register_datamodule(name):
    def decorator(cls):
        DATAMODULE_REGISTRY[name] = cls
        return cls
    return decorator


import_modules(os.path.dirname(__file__), "prostab.datamodules")
