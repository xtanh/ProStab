<h1 align="center">ProStab</h1>

<p align="center">
  <strong>Protein stability prediction with inverse folding and protein language models</strong>
</p>

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&amp;logo=pytorch&amp;logoColor=white" alt="PyTorch"></a>
  <a href="https://lightning.ai/"><img src="https://img.shields.io/badge/Lightning-792EE5?style=flat-square&amp;logo=lightning&amp;logoColor=white" alt="Lightning"></a>
  <a href="https://www.biorxiv.org/content/10.1101/2025.08.11.669595v1"><img src="https://img.shields.io/badge/bioRxiv-2025-B31B1B?style=flat-square" alt="bioRxiv paper"></a>
  <a href="https://github.com/facebookresearch/hydra"><img src="https://img.shields.io/badge/Hydra-1e90ff?style=flat-square" alt="Hydra"></a>
</p>

<p align="center">
  <a href="https://ails.sjtu.edu.cn/prostab/"><strong>Web server</strong></a> ·
  <a href="https://www.biorxiv.org/content/10.1101/2025.08.11.669595v1">Paper</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#python-api">Python API</a> ·
  <a href="#citation">Citation</a>
</p>

Predict stability changes (ΔΔG) for **single amino-acid substitutions** from a
wild-type PDB structure. ProStab combines ProteinMPNN structural representations
with ESM2 sequence representations.

<p align="center">
  <img src="assets/model.png" alt="ProStab model architecture: integration of sequence and structural representations" width="900">
</p>

### Choose how to use ProStab

| Interface | Get started |
| --- | --- |
| 🌐 **Web server** | [Upload a structure and predict online](https://ails.sjtu.edu.cn/prostab/) |
| 💻 **Command line** | [Run a single prediction](#quick-start) or [process a CSV](#batch-predictions) |
| 🐍 **Python** | [Use ProStab in your own scripts](#python-api) |

> **Explore UniStab:** for multiple substitutions and insertions/deletions,
> see [our related project](https://github.com/xlab-BioAI/UniStab).

---

## Quick start

Use Linux with Python 3.7.12 and PyTorch 1.13.1. Prediction runs on CUDA or CPU
and requires the model weights below. GPU memory usage depends on sequence length.

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
`0.1232584`.

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
`--model-dir` points to the ProStab directory shown above.

For example, from any working directory after installation:

```bash
prostab predict --model-dir /path/to/ProStab/model_weight \
  --mpnn-weights /path/to/v_48_020.pt \
  --esm-weights /path/to/esm2_t33_650M_UR50D.pt \
  --pdb /path/to/structure.pdb --chain A --mutation V1A --device cuda:0
```

`--device auto` (default) uses CUDA when available, otherwise CPU. Use `--device cpu`
to select CPU or `--device cuda:N` for a specific visible GPU.

## Batch predictions

```bash
prostab predict-batch --model-dir model_weight \
  --input examples/mutations.csv --output predictions.csv
```

Each row describes an independent single substitution, with columns
`pdb_path,chain,mutation`. Rows may refer to different PDB files or chains.
Relative PDB paths are resolved against the **input CSV's directory**.
The model is loaded once and rows are evaluated sequentially. Results retain input
order and these three columns, plus `ddg,status,error`. Invalid rows have an empty
`ddg` and an error message; other rows continue. Exit status is 0 for all-success and 1 if any
row fails. Choose a new filename for each output CSV.

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
- Use original PDB residue identifiers for positions. `G42AS`
  means glycine at residue `42A` (insertion code A) becomes serine. Negative residue
  numbers are supported. Chain IDs are case-sensitive; use a single space for a
  blank PDB chain ID.
- The mutation site needs a matching wild-type amino acid and complete backbone
  coordinates (N, CA, C, O). Use standard one-letter amino-acid codes.
- ΔΔG is reported in **kcal/mol**: negative predicts increased stability; positive
  predicts decreased stability.

## Troubleshooting

- **Missing checkpoint/config:** check the directory layout above and explicitly
  supply paths when working outside the checkout.
- **ESM download fails:** prepare both ESM files on a connected machine, then use
  `--esm-weights` or a populated `TORCH_HOME` cache.
- **CUDA unavailable/out of memory:** select another visible GPU or use CPU.
- **Wild-type mismatch or missing position:** inspect the PDB chain, original
  residue number and insertion code.
- **Dependency resolution:** install with `environment.yml`. The full historical
  environment is saved in `environment-legacy.yml`.

## Development and existing training code

```bash
python -m pip install -e '.[dev]'
python -m pytest -m 'not integration'
# Optional real-weight tests; ESM must already be cached or PROSTAB_ESM_WEIGHTS set.
PROSTAB_RUN_INTEGRATION=1 python -m pytest -m integration
```

Training and evaluation entrypoints remain `train.py` and `test.py`; install
`pip install -e '.[train]'` with pip 23.3.2 to use the original training dependencies.
Training requires the original data and configuration.

## 📄 License

This project builds heavily upon **SPURS**. Please refer to their [original license](https://github.com/luo-group/SPURS) for more details.

## Acknowledgments

We gratefully acknowledge the following projects and contributions that made ProStab possible:

The training and evaluation pipelines, datasets, configurations, baselines and
metrics are adapted from [SPURS](https://github.com/luo-group/SPURS).
We thank the SPURS team for their code, data, feedback and guidance.

### Key Components

- **SPURS**: Foundational framework from [Luo Group](https://github.com/luo-group/SPURS/tree/main)
- **ProMEP**: Transformer-based fusion of structural and sequence information [Cheng et al](https://github.com/wenjiegroup/ProMEP)
- **ESM**: Protein language models from [Meta AI](https://github.com/facebookresearch/esm)
- **ProteinMPNN**: Inverse folding model from [Dauparas et al.](https://github.com/dauparas/ProteinMPNN)

<a id="citation"></a>

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


