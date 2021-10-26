"""
Implement Pruners here.

"""
import numpy as np

from policies.policy import PolicyBase
from utils import (get_total_sparsity,
                    recompute_bn_stats,
                    percentile,
                    get_prunable_children)


import torch
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

import logging
from typing import List, Dict
from copy import deepcopy

from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pdb

def build_pruner_from_config(model, pruner_config):
    """
    This function takes the takes pruner config and model that are provided by function
    build_pruners_from_config. We assume that each pruner have one parameter
    group, i.e., which shares sparsity levels and pruning schedules.

    The *suggested!* .yaml file structure is defined in build_pruners_from_config.
    """
    pruner_class = pruner_config['class']
    pruner_args = {k: v for k, v in pruner_config.items() if k != 'class'}
    pruner = globals()[pruner_class](model, **pruner_args)
    return pruner

def build_pruners_from_config(model, config):
    """
    This function takes *general* config file for current run and model 
    and returns a list of pruners which are build by build_pruner_from_config.

    Example config.yaml for pruner instances:

    >>> pruners:
    >>>   pruner_1:
    >>>     class: MagnitudePruner
    >>>     epochs: [0,2,4] # [start, freq, end] for now (TODO: but can extend functionality?)
    >>>     weight_only: True # if True prunes only *.weight parameters in specified layers
    >>>                       # if *.bias is None thif flag is just ignored
    >>>     initial_sparsity: 0.05 # initial sparsity level for parameters
    >>>     target_sparsity: 0.7 # desired sparsity level at the end of pruning
    >>>     modules: [net.0] # modules of type (nn.Conv2d, nn.Linear, effnet.StaticPaddingConv2d) (or
    >>>                      # any instance containing as parameters *.weight and *.bias? TODO: maybe useful?)
    >>>     keep_pruned: False # Optional from now on
    >>>     degree: 3 # Optional degree to use for polynomial schedule
    >>>   pruner_2:
    >>>     class: MagnitudePruner
    >>>     epochs: [0,2,4]
    >>>     weight_only: True
    >>>     initial_sparsity: 0.05
    >>>     target_sparsity: 0.8
    >>>     modules: [net.2]
    >>>     keep_pruned: False


    There is an optional arguments:
        keep_pruned: whether pruned weights values shoud be store, recommended values is false 
                     unless you want to use reintroduction with previous magnitudes
    """
    if 'pruners' not in config: return []
    pruners_config = config['pruners']
    pruners = [build_pruner_from_config(model, pruner_config) 
               for pruner_config in pruners_config.values()]
    return pruners


class Pruner(PolicyBase):
    def __init__(self, *args, **kwargs):
        # TODO: figure out a better initialization strategy so that we make sure these attributes are present in all descendant objects,
        # as well as a method to check that the supplied modules comply with our assumptions. Maybe it is fine this way, too.
        # the following asserts are needed because the base class relies on these attributes:
        assert hasattr(self, '_modules'), "@Pruner: make sure any Pruner has 'modules' and 'module_names' attribute"
        assert hasattr(self, '_module_names'), "@Pruner: make sure any Pruner has 'modules' and 'module_names' attribute"
        # this is needed because after_parameter_optimization method assumes this:
        assert all([is_wrapped_layer(_module) for _module in self._modules]), \
            "@Pruner: currently the code assumes that you supply prunable layers' names directly in the config"

    def on_epoch_end(self, **kwargs):
        # Included for completeness, but there is nothing to close out here.
        pass

    def measure_sparsity(self, **kwargs):
        sparsity_dict = {}
        for _name, _module in zip(self._module_names, self._modules):
            num_zeros, num_params = get_total_sparsity(_module)
            sparsity_dict[_name] = (num_zeros, num_params)
        return sparsity_dict

    def after_parameter_optimization(self, model, **kwargs):
        """
        Currently this stage is used to mask pruned neurons within the layer's data.
        TODO: think if this is general enough to be all Pruners' method, or
        it is GradualPruners' method only.
        """
        for _module in self._modules:
            _module.apply_masks_to_data()


class GradualPruner(Pruner):
    def __init__(self, model, **kwargs):
        """
        Arguments:
            model {nn.Module}: network with wrapped modules to bound pruner
        Key arguments:
            kwargs['initial_sparsity']: initial_sparsity layer sparsity
            kwargs['target_sparsity']: target sparsity for pruning end
            kwargs['weight_only']: bool, if only weights are pruned
            kwargs['epochs']: list, [start_epoch, pruning_freq, end_epoch]
            kwargs['modules']: list of module names to be pruned
            kwargs['degree']: float/int, degree to use in polinomial schedule, 
                              degree == 1 stands for uniform schedule
        """
        self._start, self._freq, self._end = kwargs['epochs']
        self._weight_only = kwargs['weight_only']
        self._initial_sparsity = kwargs['initial_sparsity']
        self._target_sparsity = kwargs['target_sparsity']

        self._keep_pruned = kwargs['keep_pruned'] if 'keep_pruned' in kwargs else False
        self._degree = kwargs['degree'] if 'degree' in kwargs else 3

        self._model = model
        modules_dict = dict(self._model.named_modules())

        prefix = ''
        if isinstance(self._model, torch.nn.DataParallel):
            prefix = 'module.'
        # Unwrap user-specified modules to prune into lowest-level prunables:
        self._module_names = [prefix + _name for _name in kwargs['modules']]
        # self._module_names = [prefix + _name for _name in get_prunable_children(self._model, kwargs['modules'])]

        self._modules = [
            modules_dict[module_name] for module_name in self._module_names
        ]

        if self._keep_pruned:
            for module in self._modules:
                module.copy_pruned(True)

        logging.debug(f'Constructed {self.__class__.__name__} with config:')
        logging.debug('\n'.join([f'    -{k}:{v}' for k,v in kwargs.items()]) + '\n')

    def update_initial_sparsity(self):
        parameter_sparsities = []
        for module in self._modules:
            w_sparsity, b_sparsity = module.weight_sparsity, module.bias_sparsity
            parameter_sparsities.append(w_sparsity)
            if b_sparsity is not None: parameter_sparsities.append(b_sparsity)
        self._initial_sparsity = np.mean(parameter_sparsities)

    @staticmethod
    def _get_param_stat(param):
        raise NotImplementedError("Implement in child class.")

    def _polynomial_schedule(self, curr_epoch):
        scale = self._target_sparsity - self._initial_sparsity
        progress = min(float(curr_epoch - self._start) / (self._end - self._start), 1.0)
        remaining_progress = (1.0 - progress) ** self._degree
        return self._target_sparsity - scale * remaining_progress

    def _required_sparsity(self, curr_epoch):
        return self._polynomial_schedule(curr_epoch)

    def _pruner_not_active(self, epoch_num):
        return ((epoch_num - self._start) % self._freq != 0 or epoch_num > self._end or epoch_num < self._start)


    @staticmethod
    def _get_pruning_mask(param_stats, sparsity=None, threshold=None):
        if param_stats is None: return None
        if sparsity is None and threshold is None: return None
        if threshold is None:
            threshold = percentile(param_stats, sparsity)
        return (param_stats > threshold).float()


class MagnitudePruner(GradualPruner):
    def __init__(self, model, **kwargs):
        super(MagnitudePruner, self).__init__(model, **kwargs)

    @staticmethod
    def _get_param_stat(param, param_mask):
        if param is None or param_mask is None: return None
        return (param.abs() + 1e-4) * param_mask

    def on_epoch_begin(self, epoch_num, **kwargs):
        if self._pruner_not_active(epoch_num):
            return False, {}
        for module in self._modules:
            level = self._required_sparsity(epoch_num)
            w_stat, b_stat = self._get_param_stat(module.weight, module.weight_mask),\
                             self._get_param_stat(module.bias, module.bias_mask)
            module.weight_mask, module.bias_mask = self._get_pruning_mask(w_stat, level),\
                                                   self._get_pruning_mask(None if self._weight_only else b_stat, level)
        return True, {"level": level}



class UnstructuredMagnitudePruner(GradualPruner):
    def __init__(self, model, **kwargs):
        super(UnstructuredMagnitudePruner, self).__init__(model, **kwargs)

    @staticmethod
    def _get_param_stat(param, param_mask):
        if param is None or param_mask is None: return None
        return ((param.abs() + 1e-4) * param_mask)

    def on_epoch_begin(self, epoch_num, device, **kwargs):
        if self._pruner_not_active(epoch_num):
            return False, {}

        level = self._required_sparsity(epoch_num)
        logging.debug("Desired sparsity level is ", level)
        if level == 0:
            return False, {}
        weights = torch.zeros(0)
        if device.type == 'cuda':
            weights = weights.cuda()
        total_params = 0
        for module in self._modules:
            weights = torch.cat((weights, self._get_param_stat(module.weight, module.weight_mask).view(-1)))
            if not self._weight_only:
                if module.bias is not None:
                    weights = torch.cat((weights, self._get_param_stat(module.bias, module.bias_mask).view(-1)))
        threshold = percentile(weights, level)

        for module in self._modules:
            w_stat, b_stat = self._get_param_stat(module.weight, module.weight_mask),\
                             self._get_param_stat(module.bias, module.bias_mask)
            module.weight_mask, module.bias_mask = self._get_pruning_mask(w_stat, threshold=threshold),\
                                                   self._get_pruning_mask(None if self._weight_only else b_stat,
                                                                          threshold=threshold)
        return True, {"level": level}



# Implements N:M pruner for structured sparsity, as in here: https://github.com/NM-sparsity/NM-sparsity
# Paper: https://openreview.net/pdf?id=K9bw7vqp_s
class MagnitudeNMPruner(Pruner):
    def __init__(self, model, **kwargs):

        self._start, self._freq, self._end = kwargs['epochs']
        self._weight_only = kwargs['weight_only']
        self._N = kwargs['N']
        self._M = kwargs['M']

        self._model = model
        modules_dict = dict(self._model.named_modules())

        prefix = ''
        if isinstance(self._model, torch.nn.DataParallel):
            prefix = 'module.'
        # Unwrap user-specified modules to prune into lowest-level prunables:
        self._module_names = [prefix + _name for _name in kwargs['modules']]
        # self._module_names = [prefix + _name for _name in get_prunable_children(self._model, kwargs['modules'])]

        self._modules = [
            modules_dict[module_name] for module_name in self._module_names
        ]
        logging.debug(f'Constructed {self.__class__.__name__} with config:')
        logging.debug('\n'.join([f'    -{k}:{v}' for k, v in kwargs.items()]) + '\n')


    def _pruner_not_active(self, epoch_num):
        return ((epoch_num - self._start) % self._freq != 0 or epoch_num > self._end or epoch_num < self._start)


    def on_epoch_begin(self, epoch_num, device, **kwargs):
        if self._pruner_not_active(epoch_num):
            return False, {}

        level = self._N / self._M

        for module in self._modules:
            module.bias_mask = None

            cloned_weight = module.weight.clone()
            elem_w = module.weight.numel()
            group_w = int(elem_w / self._M)

            if len(module.weight.shape)==4:
                # N:M sparsity for convolutional layers
                weight_temp = module.weight.detach().abs().permute(0, 2, 3, 1).reshape(group_w, self._M)
                idxs = torch.argsort(weight_temp, dim=1)[:, :int(self._M - self._N)]
                w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
                w_b = w_b.scatter_(dim=1, index=idxs, value=0).reshape(cloned_weight.permute(0, 2, 3, 1).shape)
                module.weight_mask = w_b.permute(0, 3, 1, 2)
            elif len(module.weight.shape)==2:
                # N:M sparsity for linear layers
                weight_temp = module.weight.detach().abs().reshape(group_w, self._M)
                idxs = torch.argsort(weight_temp, dim=1)[:, :int(self._M - self._N)]
                w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
                module.weight_mask = w_b.scatter_(dim=1, index=idxs, value=0).reshape(module.weight.shape)

            else:
                raise NotImplementedError("Only support layers of dimension 2 or 4")

        return True, {"level": level}


class TrustRegionMagnitudePruner(GradualPruner):
    def __init__(self, model, **kwargs):
        super(TrustRegionMagnitudePruner, self).__init__(model, **kwargs)

    @staticmethod
    def _get_param_stat(param, param_mask):
        if param is None or param_mask is None: return None
        return (param.abs() + 1e-4) * param_mask

    def _get_meta(self):
        meta = {'bottom magnitudes': {}, 'weights': {}}
        for idx, module in enumerate(self._modules):
            weight = module.weight[module.weight_mask.byte()].abs()
            for sp in [0.05,0.1,0.2,0.3,0.4,0.5]:
                threshold = percentile(weight, sp)
                val = (weight * (weight <= threshold).float()).norm()
                meta['bottom magnitudes'][self._module_names[idx] + f'_{sp}'] = val
            meta['weights'][self._module_names[idx]] = module.weight * module.weight_mask
        return meta

    def on_epoch_begin(self, epoch_num, **kwargs):
        meta = self._get_meta()
        level = self._required_sparsity(epoch_num)
        if self._pruner_not_active(epoch_num):
            return False, meta
        for idx, module in enumerate(self._modules):
            w_stat, b_stat = self._get_param_stat(module.weight, module.weight_mask),\
                             self._get_param_stat(module.bias, module.bias_mask)
            module.weight_mask, module.bias_mask = self._get_pruning_mask(w_stat, level),\
                                                   self._get_pruning_mask(None if self._weight_only else b_stat, level)
        return True, meta


class FisherPruner(GradualPruner):
    def __init__(self, model, **kwargs):
        super(FisherPruner, self).__init__(model, **kwargs)

    @staticmethod
    def _get_param_stat(param, param_mask):
        if param is None or param_mask is None: return None
        return (param.grad * param ** 2 + 1e-4) * param_mask

    def _release_grads(self):
        optim.SGD(self._model.parameters(), lr=1e-10).zero_grad() # Yeah, I know but don't want to do it manually

    def _compute_avg_sum_grad_squared(self, dset, subset_inds, device, num_workers):
        self._release_grads() 

        tmp_hooks, N = [], len(subset_inds)  #len(dset)
        for module in self._modules:
            tmp_hooks.append(module.weight.register_hook(lambda grad: grad ** 2 / (2 * N)))
            if module.bias is not None:
                tmp_hooks.append(module.bias.register_hook(lambda grad: grad ** 2 / (2 * N)))

        dummy_loader = torch.utils.data.DataLoader(dset, batch_size=1, num_workers=num_workers, 
                                                   sampler=SubsetRandomSampler(subset_inds))
        for in_tensor, target in dummy_loader:
            in_tensor, target = in_tensor.to(device), target.to(device)
            output = self._model(in_tensor)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()

        for hook in tmp_hooks:
            hook.remove()

    def on_epoch_begin(self, dset, subset_inds, device, num_workers, epoch_num, **kwargs):
        meta = {}
        if self._pruner_not_active(epoch_num):
            return False, {}
        self._compute_avg_sum_grad_squared(dset, subset_inds, device, num_workers)
        for module in self._modules:
            level = self._required_sparsity(epoch_num)
            w_stat, b_stat = self._get_param_stat(module.weight, module.weight_mask),\
                             self._get_param_stat(module.bias, module.bias_mask)
            module.weight_mask, module.bias_mask = self._get_pruning_mask(w_stat, level),\
                                                   self._get_pruning_mask(None if self._weight_only else b_stat, level)
        self._release_grads()
        return True, meta


class SNIPPruner(GradualPruner):
    def __init__(self, model, **kwargs):
        super(SNIPPruner, self).__init__(model, **kwargs)

    @staticmethod
    def _get_param_stat(param, param_mask):
        if param is None and param_mask is None: return None
        return (param.abs() / param.abs().sum() + 1e-4) * param_mask

    def _release_grads(self):
        optim.SGD(self._model.parameters(), lr=1e-10).zero_grad()

    def _compute_mask_grads(self, dset, subset_inds, device, num_workers, batch_size):
        self._release_grads() 

        dummy_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, num_workers=num_workers,
                                                   sampler=SubsetRandomSampler(subset_inds))
        for in_tensor, target in dummy_loader:
            in_tensor, target = in_tensor.to(device), target.to(device)
            output = self._model(in_tensor)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()

    def on_epoch_begin(self, dset, subset_inds, device, num_workers, batch_size, epoch_num, **kwargs):
        meta = {}
        if self._pruner_not_active(epoch_num):
            return False, {}
        self._compute_mask_grads(dset, subset_inds, device, num_workers, batch_size)
        for module in self._modules:
            level = self._required_sparsity(epoch_num)
            w_stat, b_stat = self._get_param_stat(module.weight_mask_grad, module.weight_mask),\
                             self._get_param_stat(module.bias_mask_grad, module.bias_mask)
            module.weight_mask, module.bias_mask = self._get_pruning_mask(w_stat, level),\
                                                   self._get_pruning_mask(None if self._weight_only else b_stat, level)
        self._release_grads()
        return True, meta


class NaiveHessianPruner(GradualPruner):
    def __init__(self, model, **kwargs):
        super(NaiveHessianPruner, self).__init__(model, **kwargs)

    @staticmethod
    def _get_param_stat(param, param_mask):
        if param is None or param_mask is None: return None
        #statistic can be negative so zeros breaking sparsity level
        #can substract (minimal + eps) and then zero out pruned stats
        param_stat = param.pow(2).mul(param.hess_diag.view_as(param))
        return (param_stat - param_stat.min() + 1e-8) * param_mask
        # param_stat = param.pow(2).mul(param.hess_diag).abs()
        # return (param_stat + 1e-4) * param_mask

    def _release_grads(self):
        optim.SGD(self._model.parameters(), lr=1e-10).zero_grad()

    def _add_hess_attr(self):
        self._release_grads()
        for param in self._model.parameters():
            setattr(param, 'hess_diag', torch.zeros(param.numel()))

    def _del_hess_attr(self):
        self._release_grads()
        for param in self._model.parameters():
            delattr(param, 'hess_diag')

    def _compute_second_derivatives(self):
        for module in self._modules:
            for param in module.parameters():
                for i in tqdm(range(param.grad.numel())):
                    param.hess_diag[i] += torch.autograd.grad(param.grad.view(-1)[i], param, 
                       retain_graph=True)[0].view(-1)[i]

    def _compute_diag_hessian(self, dset, subset_inds, device, num_workers, batch_size):
        dummy_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, num_workers=num_workers,
                                                   sampler=SubsetRandomSampler(subset_inds))
        loss = 0.
        for in_tensor, target in tqdm(dummy_loader):
            in_tensor, target = in_tensor.to(device), target.to(device)
            output = self._model(in_tensor)
            loss += torch.nn.functional.cross_entropy(output, target, reduction='sum') / len(dummy_loader.dataset)
        loss.backward(create_graph=True)
        self._compute_second_derivatives()
        self._release_grads()

    def on_epoch_begin(self, dset, subset_inds, device, num_workers, batch_size, epoch_num, **kwargs):

        ####### meta for TrainingProgressTracker ######
        meta = {
            'hess_diag_negatives': {}
        }
        ###############################################

        if self._pruner_not_active(epoch_num):
            return False, {}
        self._add_hess_attr()
        self._compute_diag_hessian(dset, subset_inds, device, num_workers, batch_size)
        for idx, module in enumerate(self._modules):
            level = self._required_sparsity(epoch_num)
            w_stat, b_stat = self._get_param_stat(module.weight, module.weight_mask),\
                             self._get_param_stat(module.bias, module.bias_mask)
            module.weight_mask, module.bias_mask = self._get_pruning_mask(w_stat, level),\
                                                   self._get_pruning_mask(None if self._weight_only else b_stat, level)
            
            ############# adding proportion of negatives in diag hessian meta ############
            total_negatives, total = (module.weight.hess_diag < 0).sum().int(),\
                                      module.weight.numel()
            if module.bias_mask is not None:
                total_negatives += (module.bias.hess_diag < 0).sum().int()
                total += (module.bias.numel())
            meta['hess_diag_negatives'][self._module_names[idx]] = (total_negatives, total)
            ##############################################################################

        self._del_hess_attr()
        return True, meta


class SignSwitchPruner(GradualPruner):
    def __init__(self, model, **kwargs):
        super(SignSwitchPruner, self).__init__(model, **kwargs)
        self._update_old_modules()
        
    def _update_old_modules(self):
        self._old_modules = []
        for module in self._modules:
            self._old_modules.append(deepcopy(module))

    @staticmethod
    def _get_pruning_mask(param_stats):
        if param_stats is None: return None
        return (param_stats > 0.).float()

    @staticmethod
    def _get_param_stat(param, old_param, param_mask):
        if param is None or param_mask is None: return None
        param_stat = 1. + torch.sign(param) * torch.sign(old_param)
        print('stats')
        print(param_stat.sum() / 2, param.numel())
        return (param_stat * param_mask > 0).float()

    def on_epoch_begin(self, dset, subset_inds, device, num_workers, 
                       batch_size, epoch_num, **kwargs):
        meta = {}
        if self._pruner_not_active(epoch_num):
            return False, {}
        for idx, module in enumerate(self._modules):
            old_module = self._old_modules[idx]
            w_stat, b_stat = self._get_param_stat(module.weight, old_module.weight, 
                                                  module.weight_mask),\
                             self._get_param_stat(module.bias, old_module.bias, 
                                                  module.bias_mask)
            module.weight_mask, module.bias_mask = self._get_pruning_mask(w_stat),\
                                                   self._get_pruning_mask(None if self._weight_only else b_stat)
        self._update_old_modules()
        return True, meta


class AdjustedTaylorPruner(GradualPruner):
    def __init__(self, model, **kwargs):
        super(AdjustedTaylorPruner, self).__init__(model, **kwargs)

    @staticmethod
    def _get_param_stat(param, param_mask):
        if param is None or param_mask is None: return None
        param_stat = (
            param.pow(2).mul(0.5).mul(param.hess_diag)
            - param.mul(param.grad_tmp)
        )
        return (param_stat - param_stat.min() + 1e-10) * param_mask

    def _release_grads(self):
        optim.SGD(self._model.parameters(), lr=1e-10).zero_grad()

    def _add_attrs(self):
        self._release_grads()
        for param in self._model.parameters():
            setattr(param, 'hess_diag', 0)
            setattr(param, 'grad_tmp', 0)

    def _del_attrs(self):
        self._release_grads()
        for param in self._model.parameters():
            delattr(param, 'hess_diag')
            delattr(param, 'grad_tmp')

    def _compute_first_second_derivatives(self):
        for module in self._modules:
            for param in module.parameters():
                 param.grad_tmp += param.grad.data
                 param.hess_diag += torch.autograd.grad(param.grad, param, grad_outputs=torch.ones_like(param),
                                                        retain_graph=True)[0]

    def _compute_derivatives(self, dset, subset_inds, device, num_workers, batch_size):
        dummy_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, num_workers=num_workers,
                                                   sampler=SubsetRandomSampler(subset_inds))

        for in_tensor, target in dummy_loader:
            in_tensor, target = in_tensor.to(device), target.to(device)
            output = self._model(in_tensor)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward(create_graph=True)
            self._compute_first_second_derivatives()
            self._release_grads()

    def on_epoch_begin(self, dset, subset_inds, device, num_workers, batch_size, epoch_num, **kwargs):
        meta = {}
        if self._pruner_not_active(epoch_num):
            return False, {}
        self._add_attrs()
        self._compute_derivatives(dset, subset_inds, device, num_workers, batch_size)
        for idx, module in enumerate(self._modules):
            level = self._required_sparsity(epoch_num)
            w_stat, b_stat = self._get_param_stat(module.weight, module.weight_mask),\
                             self._get_param_stat(module.bias, module.bias_mask)
            module.weight_mask, module.bias_mask = self._get_pruning_mask(w_stat, level),\
                                                   self._get_pruning_mask(None if self._weight_only else b_stat, level)
        for _module in self._modules:
            _module.apply_masks_to_data()
        self._del_attrs()
        return True, meta


if __name__ == '__main__':
    pass
