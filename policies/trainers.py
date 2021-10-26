"""
This module implements training policies.
For most usecases, only one trainer instance is needed for training and pruning
with a single model. Several trainers can be used for training with knowledge distillation.
"""

import numpy as np
import torch
import torch.nn as nn
from optimization.sgd import SGD
# from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.cuda.amp import autocast
import torch.nn.functional as F
import logging
# import torchcontrib

from policies.policy import PolicyBase
from optimization.gradual_norm_reduction_pruner import (
    _preprocess_params_for_pruner_optim, 
    GradualNormPrunerSGD
)
from optimization.lr_schedulers import StageExponentialLR, CosineLR
from utils.jsd_loss import JsdCrossEntropy
from utils.masking_utils import WrappedLayer

SPECIAL_OPTIMIZERS = ['GradualNormPrunerSGD']


def build_optimizer_from_config(model, optimizer_config):
    optimizer_class = optimizer_config['class']
    restricted_keys = ['class', 'swa_start', 'swa_freq', 'swa_lr', 'modules']
    optimizer_args = {k: v for k, v in optimizer_config.items() if k not in restricted_keys}
    if optimizer_class in SPECIAL_OPTIMIZERS:
        params = _preprocess_params_for_pruner_optim(model, optimizer_config['modules'])
        optimizer_args['params'] = params
    else:
       optimizer_args['params'] = model.parameters()
    optimizer = globals()[optimizer_class](**optimizer_args)

    if 'swa_start' in optimizer_config.keys():
        optimizer = torchcontrib.optim.SWA(optimizer, swa_start=optimizer_config['swa_start'],
            swa_freq=optimizer_config['swa_freq'], swa_lr=optimizer_config['swa_lr'])
    return optimizer


def build_lr_scheduler_from_config(optimizer, lr_scheduler_config):
    lr_scheduler_class = lr_scheduler_config['class']
    lr_scheduler_args = {k: v for k, v in lr_scheduler_config.items() if k != 'class'}
    lr_scheduler_args['optimizer'] = optimizer
    epochs = lr_scheduler_args['epochs']
    lr_scheduler_args.pop('epochs')
    lr_scheduler = globals()[lr_scheduler_class](**lr_scheduler_args)
    return lr_scheduler, epochs


def build_training_policy_from_config(model, scheduler_dict, trainer_name, use_lr_rewind=False,
                                      use_jsd=False, num_splits=None, fp16_scaler=None):
    trainer_dict = scheduler_dict['trainers'][trainer_name]
    optimizer = build_optimizer_from_config(model, trainer_dict['optimizer'])
    lr_scheduler, epochs = build_lr_scheduler_from_config(optimizer, trainer_dict['lr_scheduler'])
    return TrainingPolicy(model, optimizer, lr_scheduler, epochs,
        use_jsd=use_jsd, num_splits=num_splits, fp16_scaler=fp16_scaler)


class TrainingPolicy(PolicyBase):
    def __init__(self, model, optimizer, lr_scheduler, epochs,
                 use_jsd=False, num_splits=None, fp16_scaler=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.model = model
        self.fp16_scaler = fp16_scaler
        self.enable_autocast = False
        if fp16_scaler is not None:
            self.enable_autocast = True

        print("initial optim lr", self.optim_lr)

        self.use_jsd = use_jsd
        self.num_splits = num_splits

        if self.use_jsd:
            if self.num_splits == 0: raise ValueError('num_splits > 0! if use_jsd == True')
            self.jsd_loss = JsdCrossEntropy(num_splits=self.num_splits)

    def eval_model(self, loader, device, epoch_num):
        self.model.eval()
        eval_loss = 0
        correct = 0
        with torch.no_grad():
            for in_tensor, target in loader:
                in_tensor, target = in_tensor.to(device), target.to(device)
                with autocast(enabled=self.enable_autocast):
                    output = self.model(in_tensor)
                    eval_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        eval_loss /= len(loader.dataset)
        return eval_loss, correct

    @property
    def optim_lr(self):
        return list(self.optimizer.param_groups)[0]['lr']


    def on_minibatch_begin(self, minibatch, device, loss, **kwargs):
        """
        Loss can be composite, e.g., if we want to add some KD or
        regularization in future
        """
        self.model.train()
        self.optimizer.zero_grad()
        in_tensor, target = minibatch

        if hasattr(self, 'jsd_loss'):
            in_tensor = torch.cat(in_tensor)
            target = torch.cat(self.num_splits*[target])
        in_tensor, target = in_tensor.to(device), target.to(device)
        
        with autocast(enabled=self.enable_autocast):
            output = self.model(in_tensor)
            if hasattr(self, 'jsd_loss'):
                loss += self.jsd_loss(output, target)
            else:
                loss += F.cross_entropy(output, target)

        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        acc = 1.0 * correct / target.size(0)
        loss = torch.sum(loss)
        acc = np.sum(acc)
        return loss, acc

    def on_parameter_optimization(self, loss, epoch_num, reset_momentum, **kwargs):
        if reset_momentum:
            print("resetting momentum")
            self.optimizer.reset_momentum_buffer()
        if self.enable_autocast:
            self.fp16_scaler.scale(loss).backward()
            self.fp16_scaler.step(self.optimizer)
            self.fp16_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()



    def on_epoch_end(self, bn_loader, swap_back, device, epoch_num, **kwargs):
        start, freq, end = self.epochs
        if (epoch_num - start) % freq == 0 and epoch_num < end + 1 and start - 1 < epoch_num:
            self.lr_scheduler.step()
        if hasattr(self.lr_scheduler, 'change_mode') and epoch_num > end:
            self.lr_scheduler.change_mode()
            self.lr_scheduler.step()
        
        if hasattr(self.optimizer, 'on_epoch_begin'):
            self.optimizer.on_epoch_begin()

        if bn_loader is not None:
            print('Averaged SWA model:')
            self.optimizer.swap_swa_sgd()
            self.optimizer.bn_update(bn_loader, self.model, device)

        if swap_back:
            self.optimizer.swap_swa_sgd()

if __name__ == '__main__':
    """
    TODO: remove after debug
    """
    from efficientnet_pytorch import EfficientNet
    from masking_utils import get_wrapped_model

    from utils import read_config
    path = "./configs/test_config.yaml"
    sched_dict = read_config(stream)

    model = get_wrapped_model(EfficientNet.from_pretrained('efficientnet-b1'))
    optimizer = build_optimizer_from_config(model, sched_dict['optimizer'])
    lr_scheduler,_ = build_lr_scheduler_from_config(optimizer, sched_dict['lr_scheduler'])
    training_policy = build_training_policy_from_config(model, sched_dict)

