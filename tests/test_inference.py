import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from omegaconf import OmegaConf

from prostab.inference import Predictor, _load_state, get_prostab, parse_pdb, resolve_device
from prostab.cli import main

ROOT = Path(__file__).resolve().parents[1]
PDB = ROOT / 'data/inference_example/1A0N.pdb'
CFG = OmegaConf.load(ROOT / 'model_weight/.hydra/config.yaml')


def test_inference_does_not_import_training():
    subprocess.run([sys.executable, '-c',
                    'import sys; from prostab import Predictor; '
                    'from prostab.models.stability.prostab import FusionModel; '
                    'assert "pytorch_lightning" not in sys.modules; '
                    'assert "prostab.tasks" not in sys.modules'], check=True)


@pytest.mark.parametrize('chain,mutation,message', [
    ('Z', 'V1A', 'Chain'), ('AA', 'V1A', 'chain'),
    ('A', 'V999A', 'absent'), ('A', 'A1V', 'mismatch'),
    ('A', 'V1A:P57A', 'substitution'), ('A', 'V1X', 'substitution'),
    ('A', 'V1V', 'differ'), ('A', '', 'substitution'),
])
def test_invalid_input(chain, mutation, message):
    with pytest.raises(ValueError, match=message):
        parse_pdb(PDB, 'example', chain, CFG, mutation, 'cpu')


def test_legacy_batch_and_cpu():
    batch = parse_pdb(PDB, 'example', 'A', CFG, 'V1A', 'cpu')
    assert batch['mut_ids'].tolist() == [0]
    assert batch['mut_seq'][0][0] == 'A'
    assert batch['X'].device.type == 'cpu'
    assert parse_pdb(PDB, 'example', 'A', CFG, device='cpu')['mut_ids'].tolist() == [0]


def write_structure(path, residues, omit=None):
    lines = []
    serial = 0
    for number, insertion, amino in residues:
        for atom in ('N', 'CA', 'C', 'O'):
            if atom == omit:
                continue
            serial += 1
            lines.append('ATOM  {serial:5d} {atom:^4s} {amino:3s} A{number:4d}{ins:1s}   '
                         '{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n'.format(
                             serial=serial, atom=atom, amino=amino, number=number,
                             ins=insertion, x=serial, y=0., z=1.))
    path.write_text(''.join(lines))


def test_gaps_insertions_and_unsorted_records(tmp_path):
    path = tmp_path / 'gaps.pdb'
    write_structure(path, [(3, 'A', 'GLY'), (1, '', 'VAL'), (3, '', 'SER')])
    batch = parse_pdb(path, 'gaps', 'A', CFG, 'G3AA', 'cpu')
    assert batch['seq'] == 'V-SG'
    assert batch['mut_ids'].tolist() == [3]
    assert batch['mut_seq'] == ['V-SA']
    with pytest.raises(ValueError, match='absent'):
        parse_pdb(path, 'gaps', 'A', CFG, 'G2A', 'cpu')


def test_negative_residue_and_incomplete_backbone(tmp_path):
    path = tmp_path / 'negative.pdb'
    write_structure(path, [(-1, '', 'VAL')])
    assert parse_pdb(path, 'neg', 'A', CFG, 'V-1A', 'cpu')['mut_ids'].item() == 0
    write_structure(path, [(-1, '', 'VAL')], omit='O')
    with pytest.raises(ValueError, match='backbone'):
        parse_pdb(path, 'neg', 'A', CFG, 'V-1A', 'cpu')


def test_missing_files_and_devices(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match='configuration'):
        get_prostab(tmp_path, 'cpu')
    with pytest.raises(FileNotFoundError, match='PDB'):
        parse_pdb(tmp_path / 'missing.pdb', 'missing', 'A', CFG, 'V1A', 'cpu')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert resolve_device('auto').type == 'cpu'
    with pytest.raises(ValueError, match='unavailable'):
        resolve_device('cuda:0')


def test_strict_checkpoint_validation():
    model = torch.nn.Linear(2, 1)
    valid = {'model.' + k: v.clone() for k, v in model.state_dict().items()}
    _load_state(model, {'state_dict': dict(valid, metric=torch.tensor(1))})
    for state in ({}, dict(valid, **{'model.extra': torch.tensor(1)}),
                  dict(valid, **{'model.weight': torch.zeros(3, 3)})):
        with pytest.raises(RuntimeError):
            _load_state(model, {'state_dict': state})


class FakeModel(torch.nn.Module):
    def forward(self, batch):
        assert not self.training
        assert torch.is_inference_mode_enabled()
        return torch.tensor([0.25])


@pytest.fixture
def predictor(monkeypatch):
    monkeypatch.setattr('prostab.inference.get_prostab', lambda *a, **kw: (FakeModel(), CFG))
    return Predictor(device='cpu')


def test_api_batch(predictor):
    result = predictor.predict_many(PDB, 'A', ['V1A', 'V999A', 'P57A'])
    assert [r['status'] for r in result] == ['success', 'error', 'success']
    assert result[1]['ddg'] is None
    assert result[0]['ddg'] == predictor.predict(PDB, 'A', 'V1A')


def test_cli_single_and_mixed_csv(predictor, tmp_path, capsys):
    assert main(['predict', '--pdb', str(PDB), '--chain', 'A', '--mutation', 'V1A']) == 0
    assert json.loads(capsys.readouterr().out)['ddg'] == 0.25
    (tmp_path / 'structure.pdb').write_bytes(PDB.read_bytes())
    source = tmp_path / 'input.csv'
    source.write_text('pdb_path,chain,mutation\nstructure.pdb,A,V1A\nstructure.pdb,A,V999A\nstructure.pdb,A,\n')
    out = tmp_path / 'out.csv'
    assert main(['predict-batch', '--input', str(source), '--output', str(out)]) == 1
    with out.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [r['status'] for r in rows] == ['success', 'error', 'error']
    assert rows[0]['pdb_path'] == 'structure.pdb'
    assert rows[1]['ddg'] == ''
    before = out.read_bytes()
    assert main(['predict-batch', '--input', str(source), '--output', str(out)]) == 1
    assert out.read_bytes() == before


def test_bad_csv_before_loading_weights(tmp_path):
    source = tmp_path / 'input.csv'
    source.write_text('mutation\nV1A\n')
    assert main(['predict-batch', '--input', str(source), '--output', str(tmp_path / 'out.csv')]) == 1
    assert not (tmp_path / 'out.csv').exists()


def test_malformed_pdb_is_a_row_error(predictor, tmp_path):
    path = tmp_path / 'invalid.pdb'
    path.write_text('ATOM                  A\n')
    result = predictor.predict_many(path, 'A', ['V1A'])
    assert result[0]['status'] == 'error'
    assert result[0]['ddg'] is None


def test_missing_auxiliary_weights(tmp_path):
    (tmp_path / '.hydra').mkdir()
    (tmp_path / '.hydra/config.yaml').write_text('model: {}')
    (tmp_path / 'checkpoints').mkdir()
    (tmp_path / 'checkpoints/best.ckpt').touch()
    with pytest.raises(FileNotFoundError, match='ProteinMPNN'):
        get_prostab(tmp_path, 'cpu', mpnn_weights=tmp_path / 'missing.pt')
