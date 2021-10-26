import os
import errno
import torch
import shutil
import logging
import inspect

from models import get_model
from utils.utils import normalize_module_name, _fix_double_wrap_dict
from utils.masking_utils import (is_wrapped_layer, 
                                 WrappedLayer, 
                                 get_wrapped_model,
                                 )

import pdb

__all__ = ['save_eval_checkpoint', 'save_checkpoint', 
           'load_eval_checkpoint', 'load_checkpoint', 'get_unwrapped_model']


# Add dummy stuff here if you need to add an optimizer or a lr_scheduler
# that have required constructor arguments (and you want to recover it 
# from state dict later and need some dummy value)
DUMMY_VALUE_FOR_OPT_ARG = {'lr': 1e-3, 'gamma': 0.9}


def should_unwrap_layer(layer: 'nn.Module') -> bool:
    return isinstance(layer, WrappedLayer)

def unwrap_module(module: 'nn.Module', prefix='.'):
    """
    Recursive function which iterates over WRAPPED_MODULES of this
    module and unwraps them.
    """
    module_dict = dict(module.named_children())
    for name, sub_module in module_dict.items():
        if should_unwrap_layer(sub_module):
            setattr(module, name, sub_module.unwrap())
            print(f'Module {prefix + name} was successfully unwrapped')
            continue
        unwrap_module(sub_module, prefix + name + '.')

def get_unwrapped_model(model: 'nn.Module') -> 'nn.Module':
    """
    Function which unwrappes the wrapped layers of received model.
    """
    unwrap_module(model)
    return model

def save_eval_checkpoint(model_config: str, model: 'nn.Module', checkpoint_path: str):
    """
    Save the model state dict with all layer unwrapped and 
    pruning masks applied.
    
    Arguments:
        model_config {dict} -- {'arch': arch, 'dataset': dataset}
        path {str} -- path to save wrapped model (e.g.: exps_root/sample_run/run_id)
    """
    if not os.path.isdir(checkpoint_path):
        raise IOError(errno.ENOENT, 'Checkpoint directory does not exist at', os.path.abspath(checkpoint_path))

    model = get_unwrapped_model(model)
    if isinstance(model, torch.nn.DataParallel):
        # TODO: not sure it should be here
        logging.debug('Was using data parallel')
        model = model.module
    model_state_dict = model.state_dict()
    state_dict = dict()
    state_dict['model_config'] = model_config
    state_dict['model_state_dict'] = model_state_dict
    torch.save(state_dict, os.path.join(checkpoint_path, 'eval_ready_state_dict.ckpt'))

def load_eval_checkpoint(checkpoint_path: str) -> 'nn.Module':
    """
    Load the evaluation ready model given the chepoint path.
    """
    try:
        state_dict = torch.load(os.path.join(checkpoint_path, 'eval_ready_state_dict.ckpt'))
    except:
        raise IOError(errno.ENOENT, 'Evaluation checkpoint does not exist at', os.path.abspath(checkpoint_path))
    model_config = state_dict['model_config']
    model = get_model(model_config['arch'], model_config['dataset'])
    model.load_state_dict(state_dict['model_state_dict'])
    return model

def save_checkpoint(epoch, model_config, model, optimizer, lr_scheduler,
                    checkpoint_path: str,
                    is_best_sparse=False, is_best_dense=False, is_scheduled_checkpoint=False):
    """
    This function damps the full checkpoint for the running manager.
    Including the epoch, model, optimizer and lr_scheduler states. 
    """
    if not os.path.isdir(checkpoint_path):
        raise IOError(errno.ENOENT, 'Checkpoint directory does not exist at', os.path.abspath(checkpoint_path))

    checkpoint_dict = dict()
    checkpoint_dict['epoch'] = epoch
    checkpoint_dict['model_config'] = model_config
    checkpoint_dict['model_state_dict'] = model.state_dict()
    checkpoint_dict['optimizer'] = {
        'type': type(optimizer),
        'state_dict': optimizer.state_dict()
    }
    checkpoint_dict['lr_scheduler'] = {
        'type': type(lr_scheduler),
        'state_dict': lr_scheduler.state_dict()
    }

    path_regular = os.path.join(checkpoint_path, f'regular_checkpoint{epoch}.ckpt')
    path_best_sparse = os.path.join(checkpoint_path, 'best_sparse_checkpoint.ckpt')
    path_best_dense = os.path.join(checkpoint_path, 'best_dense_checkpoint.ckpt')
    path_last = os.path.join(checkpoint_path, 'last_checkpoint.ckpt')
    torch.save(checkpoint_dict, path_last)
    if is_best_sparse:
        print("util - saving best sparse")
        shutil.copyfile(path_last, path_best_sparse)
    if is_best_dense:
        print("util - saving best dense")
        shutil.copyfile(path_last, path_best_dense)
    if is_scheduled_checkpoint:
        print("util - saving on schedule")
        shutil.copyfile(path_last, path_regular)

# TODO: common problem for both of the loaders below:
# if we have a different class of optimizer and scheduler,
# they will need different positional arguments that don't have defaults
# question: how do we supply these default parameters?
# the below is a crazy solution
def _load_optimizer(cls, state_dict, model):
    try:
        varnames, _, _, defaults = inspect.getargspec(cls.__init__)
        # logging.debug(f"Varnames {varnames}, defaults {defaults}")
        vars_needing_vals = varnames[2:-len(defaults)]
        # logging.debug(f"Need values: {vars_needing_vals}")
        kwargs = {v: DUMMY_VALUE_FOR_OPT_ARG[v] for v in vars_needing_vals}
        optimizer = cls(model.parameters(), **kwargs)
    except KeyError as e:
        logging.debug(f"You need to add a dummy value for {e} to DUMMY_VALUE_FOR_OPT_ARG dictionary in checkpoints module.")
        raise
    optimizer.load_state_dict(state_dict)
    return optimizer

def _load_lr_scheduler(cls, state_dict, optimizer):
    try:
        varnames, _, _, defaults = inspect.getargspec(cls.__init__)
        vars_needing_vals = varnames[2:-len(defaults)]
        kwargs = {v: DUMMY_VALUE_FOR_OPT_ARG[v] for v in vars_needing_vals}
        lr_scheduler = cls(optimizer, **kwargs)
    except KeyError as e:
        logging.debug(f"You need to add a dummy value for {e} to DUMMY_VALUE_FOR_OPT_ARG dictionary in checkpoints module.")
        raise
    lr_scheduler.load_state_dict(state_dict)
    return lr_scheduler

def load_checkpoint(full_checkpoint_path: str):
    """
    Loads checkpoint give full checkpoint path.
    """
    try:
        checkpoint_dict = torch.load(full_checkpoint_path, map_location='cpu')
    except:
        raise IOError(errno.ENOENT, 'Checkpoint file does not exist at', os.path.abspath(full_checkpoint_path))
    
    try:
        model_config = checkpoint_dict['model_config']
        kwargs = {}
        if 'resnet50_mixed' in model_config['arch']:
            kwargs = {'use_se': False, 'se_ratio': False, 
                           'kernel_sizes': 4, 'p': 1}
        model = get_wrapped_model(get_model(*model_config.values(), **kwargs))
        # TODO: May need to discuss this
        # This is needed because the newly created model contains _bias_masks
        # that the saved checkpoint doesn't have. In this case we can either:
        # remove this attribute from the model at all, or leave it as is (just a ones mask, no problem)
        # and update the entries in the model's state_dict that appear in the saved state_dict
        # -- the latter option is implemented below:
        updated_state_dict = model.state_dict()
        if 'module' in list(checkpoint_dict['model_state_dict'].keys())[0]:
            checkpoint_dict['model_state_dict'] = {normalize_module_name(k): v for k, v in checkpoint_dict['model_state_dict'].items()}
        # get_wrapped_model adds new _layer keys to the state_dict (e.g. conv1._layer.weight, instead of conv1.weight)
        # The following line is commented because it changes layer._layer.weight to layer.weight and model.state_dict() has only layer._layer.weight keys:
        # (why was it used in the first place?)
        #checkpoint_dict['model_state_dict'] = _fix_double_wrap_dict(checkpoint_dict['model_state_dict'])
        updated_state_dict.update(checkpoint_dict['model_state_dict'])
        model.load_state_dict(updated_state_dict)

        optimizer_dict = checkpoint_dict['optimizer']['state_dict']
        lr_scheduler_dict = checkpoint_dict['lr_scheduler']['state_dict']
        epoch = checkpoint_dict['epoch']
        # optimizer_class, optimizer_state_dict = checkpoint_dict['optimizer'].values()
        # optimizer = _load_optimizer(optimizer_class, optimizer_state_dict, model)

        # lr_scheduler_class, lr_scheduler_state_dict = checkpoint_dict['lr_scheduler'].values()
        # lr_scheduler = _load_lr_scheduler(lr_scheduler_class, lr_scheduler_state_dict, optimizer)
    except Exception as e:
        raise TypeError(f'Checkpoint file is not valid. {e}')
    
    return epoch + 1, model, optimizer_dict, lr_scheduler_dict

