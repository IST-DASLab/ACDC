"""
Implement weight freezing policies here

WARNING: We currently do not support using both freezers
and pruners in one run to simplify the code logics

TODO: THINK OF BETTER REALIZATION USE THIS OR CUSTOM OPTIMIZERS?!

THE CODE BELOW IS PROTOTYPE!!!
"""
import torch
import torch.nn as nn

from policies.policy import PolicyBase
from utils import percentile

import logging

def build_freezer_from_config(model, freezer_config):
    """
    This function build freezer given the model (only need for weigths typically)
    and freezer configuration.
    """
    freezer_class = freezer_config['class']
    freezer_args = {k: v for k, v in freezer_config.items() if k != 'class'}
    freezer = globals()[freezer_class](model, **freezer_args)
    return freezer

def build_freezers_from_config(model, config):
    """
    This function takes *general* config file for current run and model 
    and returns a list of freezers which are build by build_freezer_from_config.

    Example config.yaml for freezer instances:

    >>> freezers:
    >>>   freezer1:
    >>>     class: AdagradOneShotFreezer # freezer method to use
    >>>     levels: 0.5 # percentage of freezed
    >>>     epoch: 10 # epoch when the weights are freezed (currently one-shot)
    >>>     modules: [net.0] # modules to apply freezing policy
    >>>     weight_only: True # if freezer is applied only to weights of module

    Levels is eighter a list of levels or a single float in [0,1] corresponding 
    to shared percentage of freezed across all modules.
    """
    if 'regularizers' not in config: return []
    regs_config = config['regularizers']
    regs = [build_reg_from_config(model, reg_config)
            for reg_config in regs_config.values()]
    return regs


class OneShotFreezer(PolicyBase):
    def __init__(self, model, **kwargs):
        self._model = model
        if not isinstance(self._model, nn.Module):
            raise ValueError('model should be an instance of nn.Module')
        modules_dict = dict(self._model.named_modules())
        self._weight_only, self._epoch = kwargs['weight_only'], kwargs['epoch']

        prefix = ''
        if isinstance(self._model, torch.nn.DataParallel):
            prefix = 'module.'
        self._module_names = [prefix + _name for _name in kwargs['modules']]

        self._modules = [
            modules_dict[module_name] for module_name in self._module_names
        ]

        self._levels = kwargs['levels']
        if not isinstance(self._levels, list):
            self._levels = [self._levels] * len(self._modules)

        self.optimizer = kwargs['optimizer']

        logging.debug(f'Constructed {self.__class__.__name__} with config:')
        logging.debug('\n'.join([f'    -{k}:{v}' for k,v in kwargs.items()]) + '\n')

    def on_epoch_begin(self, epoch_num, **kwargs):
        if self._epoch != epoch_num: return

        self._freeze_masks, self._params = [], []

        for module in self._modules:
            mask = self._get_freeze_mask()
            self._freeze_masks.append(mask)
            module.weight = mask * self.

    def _get_freeze_mask(self, **kwargs):
        """
        Base function for one-shot freezers. 
        Implements freezing strategy
        """
        raise ValueError('Implement in a child class')

    def after_parameter_optimization(self, **kwargs):
        if not hasattr(self, freeze_masks): return
        for i, module in enumerate(self._modules):


class AdagradOneShotFreezer(OneShotFreezer):
    def __init__(self, model, **kwargs):
        super(AdagradOneShotFreezer, self).__init__(model, **kwargs)
        self._optimizer = kwargs['optimizer']

    def _compute_weights(self):
        weights = []
        k = float('-inf')
        for group in self.param_groups:
            k = group['k']
            for p in group['params']:
                state = self.state[p]
                tmp_avg = state['sum']
                for i in tmp_avg.view(-1).tolist():
                    weights.append(i)
        weights.sort(reverse=True)
        if k==1 or k==0:
            return weights[-1]
        else:
            return weights[math.ceil(k*len(weights))-1]

    def _get_freeze_mask(self):
        pass



if __name__ == '__main__':
    pass

