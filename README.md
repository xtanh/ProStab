# ProStab: Prediction of protein stability change upon mutations by inverse folding and protein language models
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Lightning](https://img.shields.io/badge/Lightning-792EE5?logo=lightning&logoColor=white)
[![arXiv](https://img.shields.io/badge/arXiv-2503.19821-B31B1B)](https://arxiv.org/abs/2503.19821)
[![Hydra](https://img.shields.io/badge/Hydra-1e90ff?logo=dropbox&logoColor=white)](https://github.com/facebookresearch/hydra)

## Overview

ProStab, a deep learning framework that integrates sequence-derived and structure-informed features for accurate prediction of ∆∆G for protein point mutations given an initial structure. ProStab combines representations from a protein language model applied to both wild-type and mutant sequences, and from the inverse folding model ProteinMPNN applied to the wild-type structure. It jointly models two sources of information: mutation-specific effects, captured as embedding differences at the substitution site between wild-type and mutant sequences; and site-specific priors, derived from the wild-type sequence and structure, which reflect the local context and substitutional tolerance.

![Model Architecture](./assets/model.png)


## Requirements

```bash
torch>=1.12.0
pytorch-lightning
hydra-core
esm
```

## Installation

```bash
git clone https://github.com/xtanh/ProStab.git
cd ProStab
# Add installation steps if needed
```

## Usage

### Training

```bash
python train.py experiment_path=logs/proteinStability
```

### Testing

```bash
python test.py experiment_path=logs/proteinStability datamodule._target_=megascale data_split=test ckpt_path=logs/proteinStability/checkpoints/best.ckpt mode=predict
```

## Model Architecture

The model consists of three main components:
1. ESM2 encoder for sequence features
2. ProteinMPNN encoder for structural features
3. Fusion module 

## Data

Required data files (not included in repository):
<!-- - `data/dataset/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv`
- `data/dataset/Domainome/Supplementary_Table_5_aPCA_vs_variant_effect_predictors.csv`
- `data/dataset/geostab_data/dms/dms.csv` -->



## Results

[Add your model's performance metrics and comparisons]

## Citation

If you use this code, please cite:

