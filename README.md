# AC/DC Code

This code allows replicating the image experiments of [**AC/DC: Alternating Compressed/DeCompressed Training of Deep Neural Networks**](https://arxiv.org/abs/2106.12379). This code allows for training compressed and decompressed WideResNet models for CIFAR-100, ResNet50 and MobileNet models for Imagenet (also for 2:4 sparsity), and also memorization experiments on CIFAR-10. With small modifications this code can also be used for other datasets and architectures, but these are not provided.

The code is based on the open-source code for the WoodFisher neural network compression paper, which can be found on Github [here](http://github.com/IST-DASLab/WoodFisher "here").

## To Run:
We recommend access to at least one GPU for running CIFAR-10 and CIFAR-100 training and at least 2-4 GPUs for running Imagenet training. The ImageNet experiments were run using PyTorch 1.7.1, but the code was also tested using PyTorch 1.8. We recommend, if possible, using WandB for experiments tracking. Otherwise, all code should be run with the `--use_wandb` flag **disabled**.

All requirements are listed in the `requirements.txt` file. Once they are installed, training for the appropriate dataset, architecture, and sparsity can be started by running the appropriate `.sh` file, which can be adjusted to give the correct GPU ids, output directories, dataset path and `--use_wandb` flag.

Please note that each shell script points to a configuration file, found in configs/neurips. These configuration files set the pruning and recycling schedule, as well as the learning rate and its scheduler.

## Checkpoints

We provide AC/DC checkpoints trained on ImageNet using ResNet50 and MobileNetV1. To evaluate the checkpoints, we recommend using the validation scripts `run_imagenet_acdc_resnet50_validate.sh` or `run_imagenet_acdc_mobilenet_validate.sh`, by changing the ImageNet path `--dset-path` and providing the appropriate checkpoints with `--from_checkpoint_path`. All accuracies are obtained after evaluating with the scripts provided above, using PyTorch 1.7.1. We note that small differences in accuracy could arise from using other PyTorch versions, such as 1.8. The sparsity level indicated is computed with respect to all prunable parameters, excluding biases and Batch Normalization parameters.

### ResNet50 checkpoints

Models pruned using *global magnitude*

| Sparsity | Checkpoint | Top-1 Acc. (%) |
| -------- | ---------- | ---------------|
| 80%      | [AC/DC](https://seafile.ist.ac.at/f/081f90d21c9e4236bd14/?dl=1 "AC/DC") | 76.45% | 
| 90%      | [AC/DC](https://seafile.ist.ac.at/f/1b4f578130364e5bb929/?dl=1 "AC/DC") | 75.03% |
| 95%      | [AC/DC](https://seafile.ist.ac.at/f/e51ab6ee47bb490cb119/?dl=1 "AC/DC") | 73.14% |
| 98%      | [AC/DC](https://seafile.ist.ac.at/f/9ec9f8f4eda04c5f9d74/?dl=1 "AC/DC") | 68.48% |

---
Models with *uniform sparsity* (we also provide models with extended training time, namely 2x the original number of training iterations). The first and last layers are kept dense and the sparsity level is calculated with respect to only the pruned layers.

| Sparsity | Checkpoint | Top-1 Acc. (%) |
| -------- | ---------- | ----------     |
| 90%      | [AC/DC](https://seafile.ist.ac.at/f/62d9f99ffb914a12afbe/?dl=1 "AC/DC") | 75.04% |
| 95%      | [AC/DC](https://seafile.ist.ac.at/f/410aeec2a7fa40048e24/?dl=1 "AC/DC") | 73.28% |
| 90%      | [AC/DC 2x](https://seafile.ist.ac.at/f/2adc387b16be471fafdc/?dl=1 "AC/DC") | 76.09% |
| 95%      | [AC/DC 2x](https://seafile.ist.ac.at/f/152b58a5bb33448599ee/?dl=1 "AC/DC") | 74.29% |

### MobileNetV1 checkpoints

Models pruning using *global magnitude*
| Sparsity | Checkpoint | Top-1 Acc. (%) |
| -------- | ---------- | ----------     |
| 75%      | [AC/DC](https://seafile.ist.ac.at/f/d1c1c0dbb42e4b32adc5/?dl=1 "AC/DC") | 70.41% |
| 90%      | [AC/DC](https://seafile.ist.ac.at/f/576714b3f3924a4ea68a/?dl=1 "AC/DC") | 66.24% |



## Structure of the repo

* `main.py` is the module to run training/pruning from. You will need to provide data and config paths and specify dataset and architecture names.
* `configs/` contains yaml config files we use for specifying training and pruning schedules.
* `models/` directory contains currently available models. To add a new model, it must be served by `get_model` function in `models/__init__.py`.
* `policies/` contains `pruner`-type policies who prune specific layers, `trainer`-type policies who train an entire model or submodules, and a `Manager` class which executes these policies as specified in a given config.
* `utils/` contains utilities for loading datasets, masking layers for pruning, and performing helper computations.
