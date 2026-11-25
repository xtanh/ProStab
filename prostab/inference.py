"""Prediction API, adapted from the SPURS inference workflow.

Mutations use PDB residue identifiers (including insertion codes), not sequence
indices. Each prediction is an independent single amino-acid substitution.
"""
from pathlib import Path
import math
import re

from omegaconf import OmegaConf
import torch

from prostab.utils import recursive_to, seed_everything
from prostab.datamodules.datasets.utils import alt_parse_PDB, get_pdb
from prostab.datamodules.datasets.data_utils import Alphabet

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
ALPHABET = AMINO_ACIDS + 'X'
MUTATION = re.compile(r'^([ACDEFGHIKLMNPQRSTVWY])(-?\d+[A-Za-z]?)([ACDEFGHIKLMNPQRSTVWY])$')


def resolve_device(device='auto') -> torch.device:
    """Validate the device before loading large model weights."""
    if str(device) == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices: auto, cpu, cuda:N')
    if device.type == 'cuda':
        index = device.index if device.index is not None else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError('CUDA device {} is unavailable; use --device cpu.'.format(device))
        device = torch.device('cuda', index)
    return device


def _require_file(path, description):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError('{} not found: {}. See README: Weights.'.format(description, path))
    return path


def _load_state(model, checkpoint):
    """Strip only the Lightning model prefix and require a complete model."""
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        raise ValueError('Expected a ProStab Lightning checkpoint containing state_dict.')
    state = checkpoint['state_dict']
    state = {key[len('model.'):]: value for key, value in state.items()
             if key.startswith('model.')}
    # strict=True checks missing/unexpected model keys and tensor shapes. Task
    # metrics outside model.* are intentionally not part of the inference model.
    model.load_state_dict(state, strict=True)


def get_prostab(ckpt_path, device='auto', *, mpnn_weights=None, esm_weights=None):
    """Return (eval-mode model, config) from an existing experiment directory.

    Explicit auxiliary weight paths override the historical ProteinMPNN location
    and ESM's Torch Hub cache. Only load checkpoints from sources you trust.
    """
    device = resolve_device(device)
    root = Path(ckpt_path).expanduser().resolve()
    config_path = _require_file(root / '.hydra/config.yaml', 'ProStab configuration')
    weight_path = _require_file(root / 'checkpoints/best.ckpt', 'ProStab checkpoint')
    if mpnn_weights is None:
        mpnn_weights = Path(__file__).resolve().parents[1] / 'data/checkpoints/ThermoMPNN/vanilla_model_weights/v_48_020.pt'
    mpnn_weights = _require_file(mpnn_weights, 'ProteinMPNN checkpoint (set mpnn_weights / --mpnn-weights)')
    if esm_weights is not None:
        esm_weights = str(_require_file(esm_weights, 'ESM checkpoint'))
        regression = Path(esm_weights).with_name(Path(esm_weights).stem + '-contact-regression.pt')
        _require_file(regression, 'ESM contact regression checkpoint')
    cfg = OmegaConf.load(config_path)
    cfg.model.pop('_target_', None)
    seed_everything(cfg.train.seed)
    from prostab.models.stability.prostab import FusionModel
    model = FusionModel(cfg.model, mpnn_weights=str(mpnn_weights), esm_weights=esm_weights)
    checkpoint = torch.load(str(weight_path), map_location='cpu')
    _load_state(model, checkpoint)
    del checkpoint
    return model.to(device).eval(), cfg


def parse_pdb(pdb_path, pdb_name, chain, cfg, mutation=None, device='auto'):
    """Build a model batch. Invalid substitutions raise ValueError.

    The legacy mutation=None mode is retained for low-level callers. The public
    Predictor API always requires a substitution. Blank PDB chain IDs use ' '.
    """
    device = resolve_device(device)
    path = Path(pdb_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError('PDB file not found: {}'.format(path))
    if not isinstance(chain, str) or len(chain) != 1:
        raise ValueError('chain must be a single PDB chain identifier.')
    match = None
    if mutation is not None:
        match = MUTATION.fullmatch(mutation) if isinstance(mutation, str) else None
        if match is None:
            raise ValueError('Expected one substitution such as V1A or G42AS (PDB residue 42A).')
        if match[1] == match[3]:
            raise ValueError('Wild-type and mutant amino acids must differ.')
    try:
        parsed = alt_parse_PDB(str(path), [chain])[0]
    except (ValueError, IndexError, KeyError, TypeError) as exc:
        raise ValueError('Cannot parse PDB {}: {}'.format(path, exc)) from exc
    if not parsed['seq'] or 'seq_chain_' + chain not in parsed:
        raise ValueError('Chain {!r} has no protein residues in {}.'.format(chain, path))
    sequence = parsed['seq']
    index = None
    if match:
        wt, position, mt = match.groups()
        try:
            index = parsed['resn_list'].index(position)
        except ValueError:
            raise ValueError('Residue {} is absent from chain {!r}.'.format(position, chain)) from None
        if sequence[index] != wt:
            raise ValueError('Wild-type mismatch at {}: expected {}, PDB contains {}.'.format(position, wt, sequence[index]))
        coordinates = parsed['coords_chain_' + chain]
        if any(not all(math.isfinite(v) for v in coordinates[atom + '_chain_' + chain][index])
               for atom in ('N', 'CA', 'C', 'O')):
            raise ValueError('Mutation residue {} lacks complete backbone coordinates.'.format(position))
    pdb = get_pdb(parsed, pdb_name, pdb_name, check_assert=False)
    if match:
        mutated = sequence[:index] + mt + sequence[index + 1:]
        encoding = torch.zeros(42)
        encoding[ALPHABET.index(wt)] = 1
        encoding[21 + ALPHABET.index(mt)] = 1
        pdb['mut_ids'] = torch.tensor([index])
        pdb['append_tensors'] = encoding.unsqueeze(0)
        pdb['mut_seq'] = [mutated]
    else:
        pdb['mut_ids'] = torch.tensor([0])
        pdb['append_tensors'] = torch.zeros(1, 42)
        pdb['mut_seq'] = [sequence]
    pdb['ddG'] = torch.tensor([[0.0]])
    pdb['dataset'] = [pdb_name]
    batch = Alphabet(**cfg.datamodule.alphabet).featurize([pdb])
    return recursive_to(batch, device)


class Predictor:
    """Load once, then predict single substitutions (ΔΔG in kcal/mol)."""

    def __init__(self, model_dir='model_weight', device='auto', *, mpnn_weights=None, esm_weights=None):
        self.device = resolve_device(device)
        self.model, self.cfg = get_prostab(model_dir, self.device,
                                          mpnn_weights=mpnn_weights, esm_weights=esm_weights)

    def predict(self, pdb_path, chain: str, mutation: str) -> float:
        """Return a float; negative ΔΔG predicts increased stability."""
        if mutation is None:
            raise ValueError('A single substitution is required.')
        batch = parse_pdb(pdb_path, Path(pdb_path).stem, chain, self.cfg,
                          mutation=mutation, device=self.device)
        self.model.eval()
        with torch.inference_mode():
            value = float(self.model(batch).item())
        if not math.isfinite(value):
            raise RuntimeError('Model returned a non-finite prediction.')
        return value

    def predict_many(self, pdb_path, chain: str, mutations) -> list:
        """Return ordered result dictionaries; failed rows have ddg=None."""
        if isinstance(mutations, str):
            raise ValueError("mutations must be an iterable of substitutions, not one string.")
        return [self._predict_record(pdb_path, chain, mutation) for mutation in mutations]

    def _predict_record(self, pdb_path, chain, mutation):
        record = dict(pdb_path=str(pdb_path), chain=chain, mutation=mutation,
                      ddg=None, status='error', error='')
        try:
            record['ddg'] = self.predict(pdb_path, chain, mutation)
            record['status'] = 'success'
        except (ValueError, OSError, RuntimeError) as exc:
            record['error'] = str(exc)
        return record
