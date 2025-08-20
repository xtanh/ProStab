# ProStab: Prediction of protein stability change upon mutations by inverse folding and protein language models
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Lightning](https://img.shields.io/badge/Lightning-792EE5?logo=lightning&logoColor=white)
[![arXiv](https://img.shields.io/badge/arXiv-25-B31B1B)](https://www.biorxiv.org/content/10.1101/2025.08.11.669595v1)
[![Hydra](https://img.shields.io/badge/Hydra-1e90ff?logo=dropbox&logoColor=white)](https://github.com/facebookresearch/hydra)

## Overview

ProStab, a deep learning framework that integrates sequence-derived and structure-informed features for accurate prediction of ∆∆G for protein point mutations given an initial structure. ProStab combines representations from a protein language model applied to both wild-type and mutant sequences, and from the inverse folding model ProteinMPNN applied to the wild-type structure. It jointly models two sources of information: mutation-specific effects, captured as embedding differences at the substitution site between wild-type and mutant sequences; and site-specific priors, derived from the wild-type sequence and structure, which reflect the local context and substitutional tolerance.

![Model Architecture](assets/model.png)



## Web server
[ProStab web server](https://ails.sjtu.edu.cn/prostab/)
⭐ **Ongoing updates will keep making this web server easier and more convenient to use.** ⭐
⭐ **Coming soon: Thermal stability prediction for multiple mutations and indels.** ⭐

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

**ProteinMPNN Weights**
1. Download ProteinMPNN weights: `v_48_020.pt` from [ProteinMPNN GitHub](https://github.com/dauparas/ProteinMPNN)
2. Create the directory and place the file:
```bash
mkdir -p ./data/checkpoints/ThermoMPNN/vanilla_model_weights
# Place v_48_020.pt in ./data/checkpoints/ThermoMPNN/vanilla_model_weights/
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
model, cfg = get_prostab('./model_weight')
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

## 📄 License

This project is licensed under the MIT License 

```bash
Copyright (c) 2025 Hong Tan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Acknowledgments

We gratefully acknowledge the following projects and contributions that made ProStab possible:

- **SPURS**: This project builds upon code from [SPURS](https://github.com/luo-group/SPURS/tree/main) by the Luo Group for foundational implementations
- **ESM**: Protein language models from Meta AI's [ESM repository](https://github.com/facebookresearch/esm)
- **ProteinMPNN**: Inverse folding model from [Dauparas et al.](https://github.com/dauparas/ProteinMPNN)

## Citation

If you use this code, please cite:

Hong Tan, Xiaowei Wei, Shenggeng Lin, Xueying Mao, Junwei Chen, Heqi Sun, Yufang Zhang, Zhenghong Zhou, Dong-Qing Wei, Shuangjun Lin, Yi Xiong
bioRxiv 2025.08.11.669595; doi: https://doi.org/10.1101/2025.08.11.669595

⭐ **If you find ProStab useful, please star this repository!** ⭐

