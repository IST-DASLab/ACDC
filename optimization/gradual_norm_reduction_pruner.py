import torch

from torch.optim import Optimizer
from utils import percentile


__all__ = ['_preprocess_params_for_pruner_optim', 'GradualNormPrunerSGD']
DEFAULTS = ['conv', 'fc', 'bn', 'downsample.']



def _preprocess_params_for_pruner_optim(model, modules):
    prefix = ''
    if isinstance(model, torch.nn.DataParallel):
        prefix = 'module.'
    
    module_to_sparsity = {}
    for sparsity, name_list in modules.items():
        for name in name_list:
            module_to_sparsity[prefix + name] = sparsity
    modules_dict = dict(model.named_modules())

    sparsity_to_params = {0.: []}
    for sparsity in modules.keys():
        sparsity_to_params[sparsity] = []
    
    for module_name, module in modules_dict.items():
        if any(default in module_name for default in DEFAULTS):
            if module_name in module_to_sparsity.keys():
                sparsity_to_params[module_to_sparsity[module_name]].append(module.weight)
                if module.bias is not None:
                    sparsity_to_params[0.].append(module.bias)
            else:
                sparsity_to_params[0.] += list(module.parameters())
        else:
            continue
            
    params = []
    for sparsity, param_list in sparsity_to_params.items():
        group_dict = {'params': param_list, 'sparsity': sparsity}
        params.append(group_dict)
        
    return params


class GradualNormPrunerSGD(Optimizer):
    """
    TODO: currently tasted only with ResNet20 on CIFAR-10
    """
    def __init__(self, params, lr=None, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, 
                 start=None, freq=None, end=None, num_batches=None, 
                 initial_sparsity = 0.05):
        if lr is None or lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if start is None or end is None or freq is None:
            raise ValueError("Arguments start, freq and end should be always specified")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(GradualNormPrunerSGD, self).__init__(params, defaults)
        
        # specific params
        self.epoch_counter = -1
        self.start, self.freq, self.end, self.len = start, freq, end, num_batches
        self.initial_sparsity = initial_sparsity
        self.step_counter = 0

    def _prepare_masks(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'mask' not in self.state[p].keys():
                    self.state[p]['mask'] = None
                mask = self.state[p]['mask']
                target_sparsity = group['sparsity']
                sparsity = self._polynomial_schedule(target_sparsity, self.epoch_counter)
                if target_sparsity != 0:
                    if mask is None:
                        p_stat = p.data.abs()
                    else:
                        p_stat = (p.data.abs() + 1e-8) * mask
                    value = percentile(p_stat, sparsity)
                    mask = (p_stat > value).float()
                self.state[p]['mask'] = mask

    @property
    def _pruner_is_active(self):
        return (
            self.epoch_counter >= self.start and 
            (self.epoch_counter - self.start) % self.freq == 0 and
            self.epoch_counter <= self.end
        )

    def on_epoch_begin(self):
        """
        TODO: Currently the schedule is polinomial for norm rate decrease
        """
        self.epoch_counter += 1
        if self._pruner_is_active:
            self.step_counter = 0
            self._prepare_masks()

    def _polynomial_schedule(self, target_sparsity, curr_epoch):
        scale = target_sparsity - self.initial_sparsity
        progress = min(float(curr_epoch - self.start) / (self.end - self.start), 1.0)
        remaining_progress = (1.0 - progress) ** 3
        return target_sparsity - scale * remaining_progress

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        self.step_counter += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            sparsity = group['sparsity']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p = d_p.add(weight_decay, p.data)
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                mask = param_state['mask'] if 'mask' in param_state.keys() else None
                if mask is not None:
                    p.data.add_(-group['lr'], d_p.mul(mask))
                    if self.epoch_counter > self.end:
                        continue
                    prev_alpha = 1 - (self.step_counter - 1) / (self.len * self.freq)
                    curr_alpha = 1 - self.step_counter / (self.len * self.freq)
                    p.data[~mask.byte()] *= (curr_alpha / prev_alpha)
                    if self.step_counter == (self.len * self.freq):
                        print(p.data[~mask.byte()].sum(), curr_alpha, self.len * self.freq,
                              p.data.numel() - mask.sum(), p.data.numel(),
                              1.0 * (p.data.numel() - mask.sum()) / p.data.numel(),
                              self.epoch_counter) 
                else:
                    p.data.add_(-group['lr'], d_p)

        return loss