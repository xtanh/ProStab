# ProStab

Predict protein stability changes (ΔΔG) for **single amino-acid substitutions**
using a wild-type PDB structure, ProteinMPNN and ESM2 representations.

**[Use the online server](https://ails.sjtu.edu.cn/prostab/)** — no local installation required.
For multiple substitutions and insertions/deletions, see [UniStab](https://github.com/xlab-BioAI/UniStab).
Batch prediction here means independent single substitutions, not a combined mutant.

![Model architecture](assets/model.png)

## Quick start

The supported reference environment is Linux, Python 3.7.12 and PyTorch 1.13.1.
This release retains the original model/runtime versions where available.
The historical `fair-esm==2.0.1` export is replaced by the installable
`fair-esm==2.0.0` release, with real-weight regression checks. CUDA is
optional: CPU inference works but is slower. The model has substantial memory
requirements; GPU memory depends on sequence length. No training datasets are
needed for prediction.

```bash
git clone https://github.com/xtanh/ProStab.git
cd ProStab
conda env create -f environment.yml
conda activate ProStab
```

Run environment creation from the repository root: it installs the package with
`pip install -e .`. For an existing compatible Python environment:

```bash
python -m pip install 'pip==23.3.2'
python -m pip install -e .
prostab --help
```

Prepare the weights below, then run:

```bash
prostab predict --model-dir model_weight \
  --pdb data/inference_example/1A0N.pdb --chain A --mutation V1A
```

The command prints a JSON record with `ddg`, `status`, and the input identifiers.
With the supplied checkpoint, the reference CUDA result for V1A is approximately
`0.1232584`. Small device-dependent numerical differences are possible.

## Weights

1. Download the [ProStab checkpoint](https://drive.google.com/file/d/1xZOG3wkn6UGJS_j533laRbZLWV5DU13T/view).
   Extract it if needed and put `best.ckpt` in `model_weight/checkpoints/`.
   Keep the matching configuration already included at `model_weight/.hydra/config.yaml`.
2. Download [ProteinMPNN v_48_020.pt](https://github.com/dauparas/ProteinMPNN/blob/main/vanilla_model_weights/v_48_020.pt)
   and put it in `data/checkpoints/ThermoMPNN/vanilla_model_weights/`.
3. ESM automatically downloads `esm2_t33_650M_UR50D.pt` and its contact regression
   companion on the first model load. Torch Hub normally caches these under
   `~/.cache/torch/hub/checkpoints/`; set `TORCH_HOME` to choose another cache location.

```text
model_weight/
  .hydra/config.yaml
  checkpoints/best.ckpt
data/checkpoints/ThermoMPNN/vanilla_model_weights/v_48_020.pt
```

For offline use, prepare ESM's cached files beforehand or pass `--esm-weights`
with the local model `.pt` path. Keep `esm2_t33_650M_UR50D-contact-regression.pt`
in the same directory. `--mpnn-weights` overrides the ProteinMPNN location.
`--model-dir` points to the ProStab directory above, not directly to the checkpoint.
Only load trusted checkpoint files.

For example, from any working directory after installation:

```bash
prostab predict --model-dir /path/to/ProStab/model_weight \
  --mpnn-weights /path/to/v_48_020.pt \
  --esm-weights /path/to/esm2_t33_650M_UR50D.pt \
  --pdb /path/to/structure.pdb --chain A --mutation V1A --device cuda:0
```

`--device auto` (default) uses CUDA when available, otherwise CPU. Use `--device cpu`
to force CPU or `--device cuda:N` for a specific visible GPU. An unavailable requested
device is an error, rather than a silent fallback.

## Batch predictions

```bash
prostab predict-batch --model-dir model_weight \
  --input examples/mutations.csv --output predictions.csv
```

Input columns are `pdb_path,chain,mutation`; each row may refer to a different PDB
or chain. Relative PDB paths are resolved against the **input CSV's directory**.
The model is loaded once and rows are evaluated sequentially. Results retain input
order and these three columns, plus `ddg,status,error`. Invalid rows have an empty `ddg` and an
error message; other rows continue. Exit status is 0 for all-success and 1 if any
row fails. Existing output files are never overwritten. Fatal setup errors, such
as missing weights, stop the command before prediction.

## Python API

```python
from prostab import Predictor

predictor = Predictor(model_dir="model_weight", device="auto")
value = predictor.predict("data/inference_example/1A0N.pdb", "A", "V1A")
print(value)

results = predictor.predict_many(
    "data/inference_example/1A0N.pdb", "A", ["V1A", "P57A"]
)
# Ordered dictionaries: pdb_path, chain, mutation, ddg, status, error.
```

`Predictor` also accepts `mpnn_weights=` and `esm_weights=`. Single prediction
raises an exception on invalid input; `predict_many` records individual failures
with `ddg=None`. See [inference.ipynb](inference.ipynb) for an executable example.
The original `get_prostab(...) -> (model, cfg)` and `parse_pdb(...)` calls remain
available in `prostab.inference`.

## Inputs and interpretation

- Use the **wild-type structure**, one PDB chain, and a substitution such as `V1A`.
- Positions are original PDB residue identifiers, not sequence offsets. `G42AS`
  means glycine at residue `42A` (insertion code A) becomes serine. Negative residue
  numbers are supported. Chain IDs are case-sensitive; use a single space for a
  blank PDB chain ID.
- The wild-type amino acid must match the structure. Missing positions, incomplete
  backbone coordinates at the mutation site, unsupported amino acids and combined
  mutations are rejected. Numbering gaps are retained when mapping residues.
- ΔΔG is reported in **kcal/mol**: negative predicts increased stability; positive
  predicts decreased stability, following the existing ProStab server convention.

**Behavior corrections:** model loading now selects evaluation mode and the
high-level API disables autograd. Earlier notebook/README examples could leave
Dropout active, so their outputs may differ. Missing residues now raise an error
instead of silently substituting position zero. Core checkpoint parameters must
all match; incompatible checkpoints are rejected.

## Troubleshooting

- **Missing checkpoint/config:** check the directory layout above and explicitly
  supply paths when working outside the checkout.
- **ESM download fails:** prepare both ESM files on a connected machine, then use
  `--esm-weights` or a populated `TORCH_HOME` cache.
- **CUDA unavailable/out of memory:** select another visible GPU or use CPU.
  Independent batch rows do not require loading multiple models.
- **Wild-type mismatch or missing position:** inspect the PDB chain, original
  residue number and insertion code; do not renumber mutations as sequence offsets.
- **Dependency resolution:** use the pinned environment rather than upgrading
  PyTorch/Lightning in place. `environment-legacy.yml` preserves the historical full
  environment export (including its old machine-specific prefix) for reference only.

## Development and existing training code

```bash
python -m pip install -e '.[dev]'
python -m pytest -m 'not integration'
# Optional real-weight tests; ESM must already be cached or PROSTAB_ESM_WEIGHTS set.
PROSTAB_RUN_INTEGRATION=1 python -m pytest -m integration
```

Training and evaluation entrypoints remain `train.py` and `test.py`; install
`pip install -e '.[train]'` with pip 23.3.2 to use the original training dependencies.
This update does not reproduce the paper's datasets or training runs. Training
requires the original data and configuration. Web deployment is maintained
separately; the local CLI/API do not call or modify the online service.

## 📄 License

This project builds heavily upon **SPURS**. Please refer to their [original license](https://github.com/luo-group/SPURS) for more details.

##  Acknowledgments

We gratefully acknowledge the following projects and contributions that made ProStab possible:

### Foundation Framework
**Important Note**: This project is significantly based on the **SPURS** framework. We gratefully acknowledge that:

- **Benchmark Datasets**: All evaluation datasets originate from SPURS
- **Training Strategy**: Our training methodology follows SPURS protocols  
- **Evaluation Framework**: Assessment metrics and procedures are consistent with SPURS
- **Code Foundation**: The training and evaluation framework is built upon SPURS

We sincerely thank the SPURS team for providing an extensible training, evaluation, and modeling framework that enabled this research.

### Key Components
- **SPURS**: Foundational framework from [Luo Group](https://github.com/luo-group/SPURS/tree/main)
- **ProMEP**: Transformer encoder fuse strutural information and sequence information [Cheng et al](https://github.com/wenjiegroup/ProMEP)
- **ESM**: Protein language models from [Meta AI](https://github.com/facebookresearch/esm)
- **ProteinMPNN**: Inverse folding model from [Dauparas et al.](https://github.com/dauparas/ProteinMPNN)

## 📚 Citation

If you use ProStab in your research, please cite our work and the foundational papers:

```bibtex
@article{prostab2025,
  title={ProStab: Prediction of protein stability change upon mutations by inverse folding and protein language models},
  author={Tan, Hong and Wei, Xiaowei and Lin, Shenggeng and Mao, Xueying and Chen, Junwei and Sun, Heqi and Zhang, Yufang and Zhou, Zhenghong and Wei, Dong-Qing and Lin, Shuangjun and Xiong, Yi},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.08.11.669595},
  url={https://doi.org/10.1101/2025.08.11.669595}
}

@article{li2025generalizable,
  title={Generalizable and scalable protein stability prediction with rewired protein generative models},
  author={Li, Ziang and Luo, Yunan},
  journal={Nature Communications},
  year={2025},
  publisher={Nature Publishing Group UK London}
}

@article{thermompnn2024,
  title={Transfer learning to leverage larger datasets for improved prediction of protein stability changes},
  author={Dieckhaus, Henry and Brocidiacono, Michael and Randolph, Nicholas Z. and Kuhlman, Brian},
  journal={Proceedings of the National Academy of Sciences},
  volume={121},
  number={6},
  pages={e2314853121},
  year={2024},
  doi={10.1073/pnas.2314853121},
  url={https://www.pnas.org/doi/abs/10.1073/pnas.2314853121}
}
```

## Reference
Our work is based on the following papers.

```bibtex


@article{li2025generalizable,
  title={Generalizable and scalable protein stability prediction with rewired protein generative models},
  author={Li, Ziang and Luo, Yunan},
  journal={Nature Communications},
  year={2025},
  publisher={Nature Publishing Group UK London}
}

@article{thermompnn2024,
  title={Transfer learning to leverage larger datasets for improved prediction of protein stability changes},
  author={Dieckhaus, Henry and Brocidiacono, Michael and Randolph, Nicholas Z. and Kuhlman, Brian},
  journal={Proceedings of the National Academy of Sciences},
  volume={121},
  number={6},
  pages={e2314853121},
  year={2024},
  doi={10.1073/pnas.2314853121},
  url={https://www.pnas.org/doi/abs/10.1073/pnas.2314853121}
}


@article{cheng2024zero,
  title={Zero-shot prediction of mutation effects with multimodal deep representation learning guides protein engineering},
  author={Cheng, Peng and Mao, Cong and Tang, Jin and Yang, Sen and Cheng, Yu and Wang, Wuke and Gu, Qiuxi and Han, Wei and Chen, Hao and Li, Sihan and others},
  journal={Cell Research},
  volume={34},
  number={9},
  pages={630--647},
  year={2024},
  publisher={Springer Nature Singapore Singapore}
}


@article{rives2021biological,
  title={Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences},
  author={Rives, Alexander and Meier, Joshua and Sercu, Tom and Goyal, Siddharth and Lin, Zeming and Liu, Jason and Guo, Demi and Ott, Myle and Zitnick, C Lawrence and Ma, Jerry and others},
  journal={Proceedings of the National Academy of Sciences},
  volume={118},
  number={15},
  pages={e2016239118},
  year={2021},
  publisher={National Acad Sciences},
  note={bioRxiv 10.1101/622803},
  doi={10.1073/pnas.2016239118},
  url={https://www.pnas.org/doi/full/10.1073/pnas.2016239118},
}

@inproceedings{zheng2023lm_design,
    title={Structure-informed Language Models Are Protein Designers},
    author={Zheng, Zaixiang and Deng, Yifan and Xue, Dongyu and Zhou, Yi and YE, Fei and Gu, Quanquan},
    booktitle={International Conference on Machine Learning},
    year={2023}
}

@article{dauparas2022robust,
  title={Robust deep learning--based protein sequence design using ProteinMPNN},
  author={Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and Bai, Hua and Ragotte, Robert J and Milles, Lukas F and Wicky, Basile IM and Courbet, Alexis and de Haas, Rob J and Bethel, Neville and others},
  journal={Science},
  volume={378},
  number={6615},  
  pages={49--56},
  year={2022},
  publisher={American Association for the Advancement of Science}
}
```


---


