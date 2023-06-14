<div align="center">

# Superpoint Transformer

[![python](https://img.shields.io/badge/-Python_3.8+-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.12+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_1.6+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.2-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

[//]: # ([![Paper]&#40;http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg&#41;]&#40;https://www.nature.com/articles/nature14539&#41;)
[//]: # ([![Conference]&#40;http://img.shields.io/badge/AnyConference-year-4b44ce.svg&#41;]&#40;https://papers.nips.cc/paper/2020&#41;)


Official implementation for
<br>
_Efficient 3D Semantic Segmentation with Superpoint Transformer_
<br>
🚀⚡🔥
<br>


</div>

<p align="center">
    <img width="90%" src="./media/teaser.jpg">
</p>

## 📌  Description

SPT is a superpoint-based transformer 🤖 architecture that efficiently ⚡ 
performs semantic segmentation on large-scale 3D scenes. This method includes a 
fast algorithm that partitions 🧩 point clouds into a hierarchical superpoint 
structure, as well as a self-attention mechanism to exploit the relationships 
between superpoints at multiple scales. 

## 📰  Updates

- **15.06.2023 Official release** 🌱

## 🏗  Installation
Simply run `install.sh` to install all dependencies in a new conda environment 
named `spt`. 
```bash
# Creates a conda env named 'spt' env and installs dependencies
./install.sh
```

## 🚀  Reproducing our results
See the [Datasets page](docs/dataset.md) for further details on supported 
datasets and the data structure. 

### Training SPT
Use the following commands to train SPT:
```bash
# Train SPT on S3DIS Fold 5
python src/train.py trainer=gpu model=spt_s3dis datamodule=s3dis datamodule.fold=5 trainer.max_epochs=2000

# Train SPT on KITTI-360 Val
# ⚠️ KITTI-360 does not support automatic download, follow prompted instructions
python src/train.py trainer=gpu model=spt_kitti360 datamodule=kitti360 trainer.max_epochs=200 

# Train SPT on DALES
python src/train.py trainer=gpu model=spt_dales datamodule=dales trainer.max_epochs=400
```

You may use [Weights and Biases](https://wandb.ai) to track your experiments, 
by adding the adequate argument:

```bash
# Log S3DIS experiments to W&B
python src/train.py logger=wandb_s3dis ...

# Log KITTI-360 experiments to W&B
python src/train.py logger=wandb_kitti360 ...

# Log DALES experiments to W&B
python src/train.py logger=wandb_dales ...
```

### Evaluating SPT
Use the following commands to evaluate SPT from a checkpoint file 
`checkpoint.ckpt`. 
```bash
# Train SPT on S3DIS Fold 5
python src/eval.py ckpt_path=checkpoint.ckpt trainer=gpu model=spt_s3dis datamodule=s3dis datamodule.fold=5 trainer.max_epochs=2000

# Train SPT on KITTI-360 Val
# ⚠️ KITTI-360 does not support automatic download, follow prompted instructions
python src/eval.py ckpt_path=checkpoint.ckpt trainer=gpu model=spt_kitti360 datamodule=kitti360 trainer.max_epochs=200 

# Train SPT on DALES
python src/eval.py ckpt_path=checkpoint.ckpt trainer=gpu model=spt_dales datamodule=dales trainer.max_epochs=400
```

### 🔩  Project structure
```
└── superpoint_transformer
    │
    ├── configs                   # Hydra configs
    │   ├── callbacks                 # Callbacks configs
    │   ├── data                      # Data configs
    │   ├── debug                     # Debugging configs
    │   ├── experiment                # Experiment configs
    │   ├── extras                    # Extra utilities configs
    │   ├── hparams_search            # Hyperparameter search configs
    │   ├── hydra                     # Hydra configs
    │   ├── local                     # Local configs
    │   ├── logger                    # Logger configs
    │   ├── model                     # Model configs
    │   ├── paths                     # Project paths configs
    │   ├── trainer                   # Trainer configs
    │   │
    │   ├── eval.yaml                 # Main config for evaluation
    │   └── train.yaml                # Main config for training
    │
    ├── data                      # Project data
    │
    ├── docs                      # Documentation
    │
    ├── logs                      # Logs generated by hydra and lightning loggers
    │
    ├── media                     # Media illustrating the project
    │
    ├── notebooks                 # Jupyter notebooks
    │
    ├── scripts                   # Shell scripts
    │
    ├── src                       # Source code
    │   ├── data                      # Data structure for hierarchical partitions
    │   ├── datamodules               # Lightning DataModules
    │   ├── datasets                  # Datasets
    │   ├── dependencies              # Compiled dependencies
    │   ├── loader                    # DataLoader
    │   ├── loss                      # Loss
    │   ├── metrics                   # Metrics
    │   ├── models                    # Model architecture
    │   ├── nn                        # Model building blocks
    │   ├── optim                     # Optimization 
    │   ├── transforms                # Functions for transforms, pre-transforms, etc
    │   ├── utils                     # Utilities
    │   ├── visualization             # Interactive visualization tool
    │   │
    │   ├── eval.py                   # Run evaluation
    │   └── train.py                  # Run training
    │
    ├── tests                     # Tests of any kind
    │
    ├── .env.example              # Example of file for storing private environment variables
    ├── .gitignore                # List of files ignored by git
    ├── .pre-commit-config.yaml   # Configuration of pre-commit hooks for code formatting
    ├── install.sh                # Installation script
    ├── LICENSE                   # Project license
    └── README.md

```

### Setting up `data/` and `logs/`
The `data/` and `logs/` directories will store all your datasets and training 
logs. By default, these are placed in the repository directory. 

Since this may take some space, or your heavy data may be stored elsewhere, you 
may specify other paths for these directories by creating a 
`configs/local/defaults.yaml` file containing the following:

```yaml
# @package paths

# path to data directory
data_dir: /path/to/your/data/

# path to logging directory
log_dir: /path/to/your/logs/
```

### Structure of `data/` 
See the [Datasets page](docs/dataset.md) for further details. 

### Structure of `logs/` 
See the [Logs page](docs/logs.md) for further details.

## 💳  Credits
- This project was built using [Lightning-Hydra template](https://github.com/ashleve/lightning-hydra-template).
- The main data structures of this work rely on [PyToch Geometric](https://github.com/pyg-team/pytorch_geometric)
- Some point cloud operations were inspired from the [Torch-Points3D framework](https://github.com/nicolas-chaulet/torch-points3d), although not merged with the official project at this point. 
- For the KITTI-360 dataset, some code from the official [KITTI-360](https://github.com/autonomousvision/kitti360Scripts) was used.
- Some superpoint-graph-related operations were inspired from [Superpoint Graph](https://github.com/loicland/superpoint_graph)
- The hierarchical superpoint partition is computed using [Parallel Cut-Pursuit](https://gitlab.com/1a7r0ch3/parallel-cut-pursuit)

## Citing our work
If your work uses all or part of the present code, please include the following a citation:

```
@inproceedings{robert2023spt,
  title={Efficient 3D Semantic Segmentation with Superpoint Transformer},
  author={Robert, Damien and Raguet, Hugo and Landrieu, Loic},
  booktitle={arxiv},
  year={2023}
}
```