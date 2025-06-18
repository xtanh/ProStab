# ProStab: Prediction of protein stability change upon mutations by inverse folding and protein language models
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Lightning](https://img.shields.io/badge/Lightning-792EE5?logo=lightning&logoColor=white)
[![arXiv](https://img.shields.io/badge/arXiv-2503.19821-B31B1B)](https://arxiv.org/abs/2503.19821)
[![Hydra](https://img.shields.io/badge/Hydra-1e90ff?logo=dropbox&logoColor=white)](https://github.com/facebookresearch/hydra)

## Overview

ProStab, a deep learning framework that integrates sequence-derived and structure-informed features for accurate prediction of ∆∆G for protein point mutations given an initial structure. ProStab combines representations from a protein language model applied to both wild-type and mutant sequences, and from the inverse folding model ProteinMPNN applied to the wild-type structure. It jointly models two sources of information: mutation-specific effects, captured as embedding differences at the substitution site between wild-type and mutant sequences; and site-specific priors, derived from the wild-type sequence and structure, which reflect the local context and substitutional tolerance.

![Model Architecture](./assets/model.png)


## Installation

```bash
git clone https://github.com/xtanh/ProStab.git
cd ProStab
# Add installation steps if needed
```

## Requirements

```bash
conda env create -f environment.yml
conda activate ProStab
```

## Downloading weights and data

1. Download pre-trained weights from: [https://drive.google.com/file/d/1xZOG3wkn6UGJS_j533laRbZLWV5DU13T/view?usp=share_link]
2. Extract and place in `model_weight/checkpoints/`

```bash
mkdir -p model_weight/checkpoints
# Place downloaded best.ckpt in model_weight/checkpoints/
```

### Testing

```bash
python test.py experiment_path=model_weight  datamodule._target_=megascale data_split=test ckpt_path=model_weight/checkpoints/best.ckpt  mode=predict
```


### Training

```bash
python train.py experiment_path=logs/proteinStability
```

### Inference

```bash
from prostab.inference import  parse_pdb, get_prostab
model, cfg = get_prostab('./logs/proteinStability')
pdb_name = 'example'
pdb_path = '/home/xy_th/PROSTAB/data/inference_example/1A0N.pdb'
chain = 'A'
mutation = "V1A"  
pdb_mut = parse_pdb(pdb_path, pdb_name, chain, cfg, mutation=mutation)
result_mutant = model(pdb_mut, return_logist=True)
print(f"mutation {mutation} score: {result_mutant.item()}")
```

## Data

Required data files (not included in repository):
<!-- - `data/dataset/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv`
- `data/dataset/Domainome/Supplementary_Table_5_aPCA_vs_variant_effect_predictors.csv`
- `data/dataset/geostab_data/dms/dms.csv` -->



## Results

[Add your model's performance metrics and comparisons]

## Citation

If you use this code, please cite:

