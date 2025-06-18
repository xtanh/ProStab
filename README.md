# ProStab: Protein Stability Prediction
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Lightning](https://img.shields.io/badge/Lightning-792EE5?logo=lightning&logoColor=white)
[![arXiv](https://img.shields.io/badge/arXiv-2503.19821-B31B1B)]()
[![Hydra](https://img.shields.io/badge/Hydra-1e90ff?logo=dropbox&logoColor=white)](https://github.com/facebookresearch/hydra)

A deep learning model for predicting protein stability changes upon mutations, combining ESM2 and ProteinMPNN.

## Overview

ProStab is a fusion model that leverages both ESM2 and ProteinMPNN to predict protein stability changes (ΔΔG) upon mutations. The model architecture combines sequence-based features from ESM2 and structure-based features from ProteinMPNN through a Transformer module.



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

