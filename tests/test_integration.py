"""Opt-in real-weight regression; no weights are bundled with the test suite."""
import json
import csv
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    os.environ.get('PROSTAB_RUN_INTEGRATION') != '1', reason='Set PROSTAB_RUN_INTEGRATION=1 with prepared weights')]
ROOT = Path(__file__).resolve().parents[1]


def test_real_weights_api_and_cli(tmp_path):
    from prostab.inference import Predictor
    device = os.environ.get('PROSTAB_TEST_DEVICE', 'auto')
    model_dir = os.environ.get('PROSTAB_MODEL_DIR', str(ROOT / 'model_weight'))
    mpnn = os.environ.get('PROSTAB_MPNN_WEIGHTS', str(ROOT / 'data/checkpoints/ThermoMPNN/vanilla_model_weights/v_48_020.pt'))
    esm = os.environ.get('PROSTAB_ESM_WEIGHTS')
    pdb = ROOT / 'data/inference_example/1A0N.pdb'
    predictor = Predictor(model_dir, device, mpnn_weights=mpnn, esm_weights=esm)
    results = predictor.predict_many(pdb, 'A', ['V1A', 'P57A'])
    assert all(row['status'] == 'success' for row in results)
    values = [row['ddg'] for row in results]
    # Captured before refactoring with original weights, eval(), seed 45,
    # PyTorch 1.13.1 CUDA / RTX 4090. CPU comparison is cross-device only.
    if predictor.device.type == 'cuda':
        np.testing.assert_allclose(values, [0.12325843423604965, 0.6469135880470276], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(values[0], predictor.predict(pdb, 'A', 'V1A'), rtol=1e-5, atol=1e-6)
    del predictor
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    command = [sys.executable, '-m', 'prostab', 'predict', '--model-dir', model_dir,
               '--mpnn-weights', mpnn, '--device', device, '--pdb', str(pdb), '--chain', 'A', '--mutation', 'V1A']
    if esm:
        command += ['--esm-weights', esm]
    output = subprocess.check_output(command, cwd=tmp_path, text=True)
    np.testing.assert_allclose(json.loads(output)['ddg'], values[0], rtol=1e-5, atol=1e-6)

    source = tmp_path / 'mutations.csv'
    with source.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['pdb_path', 'chain', 'mutation'])
        for mutation in ['V1A', 'V999A', 'P57A']:
            writer.writerow([str(pdb), 'A', mutation])
    destination = tmp_path / 'predictions.csv'
    command = [sys.executable, '-m', 'prostab', 'predict-batch', '--model-dir', model_dir,
               '--mpnn-weights', mpnn, '--device', device, '--input', str(source),
               '--output', str(destination)]
    if esm:
        command += ['--esm-weights', esm]
    completed = subprocess.run(command, cwd=tmp_path)
    assert completed.returncode == 1
    with destination.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row['status'] for row in rows] == ['success', 'error', 'success']
    assert rows[1]['ddg'] == ''
    np.testing.assert_allclose([float(rows[0]['ddg']), float(rows[2]['ddg'])], values, rtol=1e-5, atol=1e-6)
