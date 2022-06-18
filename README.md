# Localized Adversarial Domain Generalization

## Installation
Our code is developed based on the [wilds benchmark](https://wilds.stanford.edu/). Please follow the instruction to install wilds.

Requirements: Pytorch, torchvision, tqdm, wandb, wilds=1.2.2. 

You need a GPU to run the code, and results will be logged with wandb.

## Data Preparation

Please follow wilds benchmark to download the required datasets to `../../data/wild/`, e.g. `../../data/wild/camelyon17_v1.0`. 

## Experiments
We provide the script for Camelyon17 and Povertymap.

1. Conduct experiments on Camelyon17 by running
```
 python examples/run_expt_camelyon17.py
```
Please change the random_seed to reproduce our results.

2. Conduct experiments on Poverty by running 
```
python examples/run_exp_poverty.py  
```
Please change fold to reproduce our results.

Thanks for your interests.

## Citation
```
@inproceedings{zhu2022localized,
  title={Localized Adversarial Domain Generalization},
  author={Zhu, Wei and Lu, Le and Xiao, Jing and Han, Mei and Luo, Jiebo and Harrison, Adam P},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7108--7118},
  year={2022}
}
```