import importlib
from omegaconf import DictConfig
import os
import glob

from prostab.utils import import_modules



MODEL_REGISTRY = {}


def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator



# automatically import any Python files in the models/ directory
import_modules(os.path.dirname(__file__), "prostab.models", excludes=['protein_structure_prediction'])
