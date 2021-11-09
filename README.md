# AC/DC Code

This code allows replicating the image experiments of [**AC/DC: Alternating Compressed/DeCompressed Training of Deep Neural Networks**](https://arxiv.org/abs/2106.12379). This code allows for training compressed and decompressed WideResNet models for CIFAR-100, ResNet50 and MobileNet models for Imagenet (also for 2:4 sparsity), and also memorization experiments on CIFAR-10. With small modifications this code can also be used for other datasets and architectures, but these are not provided.

The code is based on the open-source code for the WoodFisher neural network compression paper, which can be found on Github [here](http://github.com/IST-DASLab/WoodFisher "here").

### To Run:
We recommend access to at least one GPU for running CIFAR-10 and CIFAR-100 training and at least 2-4 GPUs for running Imagenet training. The ImageNet experiments were run using PyTorch 1.7.1, but the code was also tested using PyTorch 1.8. We recommend, if possible, using WandB for experiments tracking. Otherwise, all code should be run with the `--use_wandb` flag **disabled**.

All requirements are listed in the `requirements.txt` file. Once they are installed, training for the appropriate dataset, architecture, and sparsity can be started by running the appropriate `.sh` file, which can be adjusted to give the correct GPU ids, output directories, dataset path and `--use_wandb` flag.

Please note that each shell script points to a configuration file, found in configs/neurips. These configuration files set the pruning and recycling schedule, as well as the learning rate and its scheduler.


## Structure of the repo

* `main.py` is the module to run training/pruning from. You will need to provide data and config paths and specify dataset and architecture names.
* `configs/` contains yaml config files we use for specifying training and pruning schedules.
* `models/` directory contains currently available models. To add a new model, it must be served by `get_model` function in `models/__init__.py`.
* `policies/` contains `pruner`-type policies who prune specific layers, `trainer`-type policies who train an entire model or submodules, and a `Manager` class which executes these policies as specified in a given config.
* `utils/` contains utilities for loading datasets, masking layers for pruning, and performing helper computations.
