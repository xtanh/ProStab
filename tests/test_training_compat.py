"""Import/config smoke tests only; these never load datasets or start training."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec('pytorch_lightning') is None,
    reason='Optional training dependencies are not installed')
ROOT = Path(__file__).resolve().parents[1]


def test_training_registry_and_helpers():
    from prostab.utils.registry import get_module
    from prostab import utils
    from omegaconf import OmegaConf
    assert get_module('model', 'esmfusion').__name__ == 'FusionModel'
    assert get_module('datamodule', 'megascale') is not None
    assert get_module('task', 'stability/megascale') is not None
    cfg = OmegaConf.create({'print_config': False, 'ignore_warnings': False})
    assert utils.extras(cfg) is cfg
    import prostab.training_pipeline
    import prostab.testing_pipeline


@pytest.mark.parametrize('script', ['train.py', 'test.py'])
def test_entrypoint_configuration(script):
    subprocess.run([sys.executable, str(ROOT / script), '--cfg', 'job'],
                   cwd=ROOT, check=True, stdout=subprocess.PIPE)
