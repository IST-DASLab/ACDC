"""
Managing class for different Policies.

"""

from models import get_model
from policies.pruners import build_pruners_from_config
from policies.trainers import build_training_policy_from_config
from policies.recyclers import build_recyclers_from_config
from policies.regularizers import build_regs_from_config

from utils import read_config, get_datasets, get_wrapped_model, get_total_sparsity, get_total_grad_sparsity, get_unwrapped_model
from utils import (preprocess_for_device,
                   recompute_bn_stats,
                   load_checkpoint,
                   save_checkpoint,
                   TrainingProgressTracker,
                   normalize_module_name,
                   get_weights_hist_meta,
                   get_macs_sparse, get_flops)
from utils.masking_utils import WrappedLayer

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.onnx
import time
import os
import logging
import wandb
import collections
import copy 
import sys

import pdb

# TODO (Discuss): do we want to have tqdm in the terminal output?
USE_TQDM = True
if not USE_TQDM:
    tqdm = lambda x: x

DEFAULT_TRAINER_NAME = "default_trainer"




# noinspection SyntaxError
class Manager:
    """
    Class for invoking trainers and pruners.
    Trainers
    """

    def __init__(self, args):
        args = preprocess_for_device(args)


        # REFACTOR: Model setup
        if args.manual_seed:
            torch.manual_seed(args.manual_seed)
            np.random.seed(args.manual_seed)
        else:
            np.random.seed(0)

        self.model_config = {'arch': args.arch, 'dataset': args.dset}
        self.model = get_model(args.arch, dataset=args.dset, pretrained=args.pretrained, kernel_sizes=args.kernel_sizes,
                               hidden_size=args.hidden_size, p=args.p)
        self.config = args.config_path if isinstance(args.config_path, dict) else read_config(args.config_path)

        self.logging_function = print
        if args.use_wandb:
            import wandb
            self.logging_function = wandb.log
            wandb.watch(self.model)

        self.data = (args.dset, args.dset_path)
        self.n_epochs = args.epochs
        self.num_workers = args.workers
        self.batch_size = args.batch_size
        self.steps_per_epoch = args.steps_per_epoch
        self.num_samples = args.num_samples
        self.warmup_epochs = args.warmup_epochs
        self.reset_momentum_after_recycling = args.reset_momentum_after_recycling
        self.num_random_labels = args.num_random_labels
        self.device = args.device
        self.initial_epoch = 0
        self.best_val_acc = {"sparse": {"sparsity": 0., "val_acc": 0.}, "dense": 0.}
        self.current_sparsity = 0
        # logic is a little subtle, as we could load a checkpoint
        # which has a worse validation accuracy and this value
        # would be "ghost of the past"
        self.pruned_state = 'dense'
        self.reset_momentum = None

        # for mixed precision training:
        if args.fp16:
            self.fp16_scaler = torch.cuda.amp.GradScaler()
            logging.info("==============!!!!Train with mixed precision (FP 16 enabled)!!!!================")
        else:
            self.fp16_scaler = None

        self.masks_folder = os.path.join(args.run_dir, 'pruned_masks')
        os.makedirs(self.masks_folder, exist_ok=True)

        self.recompute_bn_stats = args.recompute_bn_stats
        self.training_stats_freq = args.training_stats_freq

        use_data_aug = True
        if self.num_random_labels>0:
            use_data_aug = False

        # Define datasets
        self.data_train, self.data_test = get_datasets(*self.data, use_data_aug=use_data_aug)

        # perturb the labels of some of the train samples
        if self.num_random_labels > 0:
            perturbed_idxs = np.random.permutation(len(self.data_train))[:self.num_random_labels]
            n_classes = np.array(self.data_train.targets).max() + 1
            perturbed_labels = np.random.choice(n_classes, size=self.num_random_labels)
            cloned_data = copy.deepcopy(self.data_train)
            # quick and dirty fix for Imagenet or Imagenette (in general anything derived from torchvision.datasets.ImageFolder)
            if args.dset in ['imagenet']:
                real_labels = self.data_train.targets
                modified_data = []
                for idx in range(len(self.data_train)):
                    if idx in perturbed_idxs:
                        i = np.where(perturbed_idxs==idx)[0][0]
                        modified_data.append((self.data_train.samples[idx][0], perturbed_labels[i]))
                    else:
                        modified_data.append((self.data_train.samples[idx][0], real_labels[idx]))
                self.data_train.samples = modified_data 
            
            for (i, idx) in enumerate(perturbed_idxs):
                self.data_train.targets[idx] = perturbed_labels[i]


            self.perturbed_data_correct_labels = torch.utils.data.Subset(cloned_data, perturbed_idxs)
            self.perturbed_data_correct_labels_loader = DataLoader(self.perturbed_data_correct_labels, batch_size=self.batch_size,
                                                                   shuffle=False, num_workers=self.num_workers)
            self.perturbed_data = torch.utils.data.Subset(self.data_train, perturbed_idxs)
            self.perturbed_data_loader = DataLoader(self.perturbed_data, batch_size=self.batch_size, shuffle=False,
                                                    num_workers=self.num_workers)
        self.train_loader = DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        self.test_loader = DataLoader(self.data_test, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)
        self.only_model = args.only_model
        optimizer_dict, lr_scheduler_dict = None, None
        if args.from_checkpoint_path is not None:
            epoch, model, opt_dict, lr_sch_dict = load_checkpoint(args.from_checkpoint_path)
            self.model = model
            if not self.only_model:
                optimizer_dict, lr_scheduler_dict = opt_dict, lr_sch_dict
       
        else:
            if (self.config['trainers'][DEFAULT_TRAINER_NAME]['optimizer']['class'] !=
                    'GradualNormPrunerSGD'):
                self.model = get_wrapped_model(self.model)
            else:
                self.config['trainers'][DEFAULT_TRAINER_NAME]['optimizer']['num_batches'] = len(self.train_loader)
        

        if args.device.type == 'cuda':
            torch.cuda.manual_seed(args.manual_seed)
            self.model = torch.nn.DataParallel(self.model, device_ids=args.gpus)
            self.model.to(args.device)
        

        self.pruners = build_pruners_from_config(self.model, self.config)
        self.recyclers = build_recyclers_from_config(self.model, self.config)
        self.regularizers = build_regs_from_config(self.model, self.config)

        self.track_weight_hist = args.track_weight_hist

        self.trainers = [build_training_policy_from_config(self.model, self.config, 
                                                           trainer_name=DEFAULT_TRAINER_NAME,
                                                           fp16_scaler=self.fp16_scaler)]
        # TODO: when loading from a pretrained model, what happens to the initial epochs for the pruners? do the pruners get ignored if epoch < self.initial_epoch?
        if optimizer_dict is not None and lr_scheduler_dict is not None:
            self.initial_epoch = epoch
            self.trainers[0].optimizer.load_state_dict(optimizer_dict)
            self.trainers[0].lr_scheduler.load_state_dict(lr_scheduler_dict)
       

        self.setup_logging(args)
        self.schedule_prunes_and_recycles()

        self.eval_only = args.eval_only
        if self.eval_only and (args.from_checkpoint_path is None):
            raise ValueError("Must provide a checkpoint for validation")

        if args.export_onnx is True:
            self.model = get_unwrapped_model(self.model)
            self.model.eval()
            onnx_batch = 1
            x = torch.randn(onnx_batch, 3, 224, 224, requires_grad=True).to(args.device)
            if args.onnx_nick:
                onnx_nick = args.onnx_nick
            else:
                onnx_nick = 'resnet_pruned.onnx'

            torch.onnx.export(self.model.module,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  onnx_nick,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'],
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

            print("ONNX EXPORT COMPLETED. EXITING")
            sys.exit()


    def setup_logging(self, args):
        self.logging_level = args.logging_level
        self.checkpoint_freq = args.checkpoint_freq
        self.exp_dir = args.exp_dir
        self.run_dir = args.run_dir

    def schedule_prunes_and_recycles(self):
        pruners_config = 'pruners' in self.config and self.config['pruners']
        self.pruning_epochs = {}
        if pruners_config:
            for i, [pruner, config] in enumerate(pruners_config.items()):
                start, freq, end = config["epochs"]
                for epoch in range(start, end, freq):
                    if epoch in self.pruning_epochs:
                        logging.info(f"o.O - already have a pruner for epoch {epoch}")
                        continue
                    self.pruning_epochs[epoch] = i
            logging.info("Pruning epochs are %s", str(self.pruning_epochs))
        recyclers_config =  'recyclers' in self.config and self.config['recyclers']
        self.recycling_epochs = {}
        if recyclers_config:
            for i, [recycler, config] in enumerate(recyclers_config.items()):
                start, freq, end = config["epochs"]
                for epoch in  range(start, end, freq):
                    if epoch in self.recycling_epochs:
                        logging.info(f"o.O - already have a recycler for epoch {epoch}")
                        continue
                    if epoch in self.pruning_epochs:
                        logging.info(f"o.O - already have a pruner for epoch {epoch}")
                        continue
                    self.recycling_epochs[epoch] = i
            logging.info("Recycling epochs are %s", str(self.recycling_epochs))


    # noinspection SyntaxError,Annotator
    def run_policies_for_method(self, policy_type: str, method_name: str, **kwargs):
        if 'agg_func' not in kwargs:
            agg_func = None
        else:
            agg_func = kwargs['agg_func']
            del kwargs['agg_func']
        res = []
        for policy_obj in getattr(self, f"{policy_type}s"):
            retval = getattr(policy_obj, method_name)(**kwargs)
            res.append(retval)

        if policy_type == 'pruner' and method_name == 'on_epoch_begin':
            is_active, meta = [r[0] for r in res], [r[1] for r in res]
            if any(is_active) and self.recompute_bn_stats:
                dataloader = DataLoader(get_datasets(*self.data)[0], batch_size=self.batch_size, shuffle=True)
                recompute_bn_stats(self.model, dataloader, self.device)
            return meta
        return res if agg_func is None else agg_func(res)


    def pruned_masks_similarity(self, epoch):
        pruned_masks_dict = collections.defaultdict(dict)
        total_masks_diff = 0.
        total_prunable_params = 0.
        
        for module_name, module in self.model.named_modules():
            # only the WrappedLayer instances have weight_mask
            if not isinstance(module, WrappedLayer):
                continue 

            w_size = module.weight.data.numel()
            if module.bias is not None:
                b_size = module.bias.data.numel()
            
            # ignore the layers that are not pruned
            if (module.weight_mask.sum().item() == w_size):
                continue
            
            pruned_masks_dict[module_name]['weight'] = module.weight_mask
            total_prunable_params += w_size

            if module.bias_mask is not None:
                pruned_masks_dict[module_name]['bias'] = module.bias_mask
                total_prunable_params += b_size
            
            # check the mask similarity if the mask at previous pruning point exists
            # the current difference between the masks is the Hamming distance
            if hasattr(module, 'prev_weight_mask'):
                sparsity_curr_mask_w = (1. - module.weight_mask).sum().item()
                sparsity_prev_mask_w = (1. - module.prev_weight_mask).sum().item()
                sparsity_diff_w = sparsity_curr_mask_w - sparsity_prev_mask_w
                module_masks_diff_w = (module.weight_mask - module.prev_weight_mask).abs().sum().item()
                module_masks_diff_w -= sparsity_diff_w
                total_masks_diff += module_masks_diff_w
                logging.info(f'different mask elements for {module_name}.weight: {module_masks_diff_w} / {w_size}')
                if module.bias_mask is not None:
                    sparsity_curr_mask_b = (1. - module.bias_mask).sum().item()
                    sparsity_prev_mask_b = (1. - module.prev_bias_mask).sum().item()
                    sparsity_diff_b = sparsity_curr_mask_b - sparsity_prev_mask_b
                    module_masks_diff_b = (module.bias_mask - module.prev_bias_mask).abs().sum().item()
                    module_masks_diff_b -= sparsity_diff_b
                    total_masks_diff += module_masks_diff_b
                    logging.info(f'different mask elements for {module_name}.bias: {module_masks_diff_b} / {b_size}')
            # update the masks for the previous pruning point
            module.set_previous_masks()
        torch.save(pruned_masks_dict, os.path.join(self.masks_folder, f'masks_epoch{epoch}.pt'))
        relative_masks_diff = 100 * total_masks_diff / total_prunable_params 
        self.logging_function({'epoch':epoch, 'relative mask difference (%)':relative_masks_diff})


    def end_epoch(self, epoch, test_loader):
        sparsity_dicts = self.run_policies_for_method('pruner',
                                                      'measure_sparsity')

        num_zeros, num_params = get_total_sparsity(self.model)
        num_zeros_grad, num_grad_params = get_total_grad_sparsity(self.model)

        self.training_progress.sparsity_info(epoch,
                                             sparsity_dicts,
                                             num_zeros,
                                             num_params,
                                             num_zeros_grad,
                                             num_grad_params,
                                             self.logging_function)

        self.run_policies_for_method('trainer',
                                     'on_epoch_end',
                                     bn_loader=None,
                                     swap_back=False,
                                     device=self.device,
                                     epoch_num=epoch,
                                     agg_func=lambda x: np.sum(x, axis=0))


    def get_eval_stats(self, epoch, dataloader, type = 'val'):
        res = self.run_policies_for_method('trainer', 'eval_model',
                                                     loader=dataloader,
                                                     device=self.device,
                                                     epoch_num=epoch)
        loss, correct = res[0]
        # log validation stats
        if type == 'val':
            self.logging_function({'epoch': epoch, type + ' loss':loss, type + ' acc':correct / len(dataloader.dataset)})
        logging.info({'epoch': epoch, type + ' loss':loss, type + ' acc':correct / len(dataloader.dataset)})

        if type == "val":
            self.training_progress.val_info(epoch, loss, correct)
        return loss, correct


    def run(self):
        train_loader = self.train_loader
        test_loader = self.test_loader
        
        # If the evaluation flag is enabled, then only compute the validation accuracy and exit the method
        if self.eval_only:
            trainer = self.trainers[0]
            eval_loss, eval_correct = trainer.eval_model(self.test_loader, self.device, 100)
            eval_acc = eval_correct / len(self.test_loader.dataset)
            print(f'Validation loss: {eval_loss} \t Top-1 validation accuracy: {eval_acc}')
            return 

        total_train_flops = 0

        self.training_progress = TrainingProgressTracker(self.initial_epoch,
                                                         len(train_loader),
                                                         len(test_loader.dataset),
                                                         self.training_stats_freq,
                                                         self.run_dir)


        for epoch in range(self.initial_epoch, self.n_epochs):
            # noinspection Annotator
            logging.info(f"Starting epoch {epoch} with number of batches {self.steps_per_epoch or len(train_loader)}")

            subset_inds = np.random.choice(len(self.data_train), self.num_samples, replace=False)

            metas = {}

            if epoch in self.pruning_epochs:
                metas = self.run_policies_for_method('pruner',
                                                     'on_epoch_begin',
                                                     num_workers=self.num_workers,
                                                     dset=self.data_train,
                                                     subset_inds=subset_inds,
                                                     device=self.device,
                                                     batch_size=64,
                                                     epoch_num=epoch)
                # track the differences in masks between consecutive mask updates
                self.pruned_masks_similarity(epoch)
                self.pruned_state = "sparse"
                level = None
                for meta in metas:
                    if "level" in meta:
                        level = meta["level"]
                        self.current_sparsity = level
                        break

            if self.track_weight_hist:
                metas = get_weights_hist_meta(self.model, metas)

            
            self.training_progress.meta_info(epoch, metas)
            if epoch in self.recycling_epochs:
                metas = self.run_policies_for_method('recycler',
                                             'on_epoch_begin',
                                             num_workers=self.num_workers,
                                             dset=self.data_train,
                                             subset_inds=subset_inds,
                                             device=self.device,
                                             batch_size=64,
                                             epoch_num=epoch,
                                             optimizer=self.trainers[0].optimizer)
                for meta in metas:
                    if "level" in meta[1]:
                        self.current_sparsity = meta[1]["level"]
                        break
                #if len(metas) > 0 and "level" in metas[0][1] and metas[0][1]["level"] <= 0.00002:
                self.pruned_state = "dense"
                self.reset_momentum=self.reset_momentum_after_recycling

            
            epoch_loss, epoch_acc = 0., 0.
            n_samples = len(train_loader.dataset)
            for i, batch in enumerate(train_loader):
                if self.steps_per_epoch and i > self.steps_per_epoch:
                    break
                # here KD can be added through the `loss` param
                start = time.time()
                val  = self.run_policies_for_method('trainer',
                                                         'on_minibatch_begin',
                                                         minibatch=batch,
                                                         device=self.device,
                                                         loss=0.,
                                                         agg_func=None)
                                                         # agg_func=lambda x: torch.sum(x, dim=0))
                loss, acc = val[0]
                epoch_acc += acc * batch[0].size(0) / n_samples

                def sum_pytorch_nums(lst):
                    res = 0.
                    for el in lst:
                        res = res + el
                    return res

                reg_loss = self.run_policies_for_method('regularizer',
                                                        'on_minibatch_begin',
                                                        agg_func=sum_pytorch_nums)
                loss = loss + reg_loss

                epoch_loss += loss.item() * batch[0].size(0) / n_samples
                
                
                self.run_policies_for_method('trainer',
                                             'on_parameter_optimization',
                                             loss=loss,
                                             reset_momentum=self.reset_momentum,
                                             epoch_num=epoch)
                # If we were supposed to reset momentum, we just have.
                self.reset_momentum=False
                
                self.run_policies_for_method('pruner',
                                             'after_parameter_optimization',
                                             model=self.model)
                self.run_policies_for_method('recycler',
                                             'after_parameter_optimization',
                                             model=self.model)

                ############################### tracking the training statistics ############################
                self.training_progress.step(loss=loss,
                                            acc=acc,
                                            time=time.time() - start,
                                            lr=self.trainers[0].optim_lr)
                #############################################################################################

            # log train stats
            self.logging_function({'epoch': epoch, 'train loss': epoch_loss, 'train acc': epoch_acc})

            # in case of using perturbed labels, track loss and accuracy on the perturbed samples 
            if self.num_random_labels > 0:
                loss_perturbed_samples, correct_perturbed_samples = self.trainers[0].eval_model(self.perturbed_data_loader,
                                                                                                self.device, epoch)
                acc_perturbed_samples = correct_perturbed_samples / len(self.perturbed_data)
                loss_perturbed_rightlabels, correct_perturbed_rightlabels = self.trainers[0].eval_model(self.perturbed_data_correct_labels_loader,
                                                                                                self.device, epoch)
                acc_perturbed_rightlabels = correct_perturbed_rightlabels / len(self.perturbed_data)
                self.logging_function({'epoch': epoch, 'perturbed loss': loss_perturbed_samples, 'perturbed acc': acc_perturbed_samples})
                self.logging_function({'epoch': epoch, 'perturbed loss (correct)': loss_perturbed_rightlabels, 'perturbed acc (correct)': acc_perturbed_rightlabels})

            # What should happen?
            # - Start tracking dense gain if we just finished warming up
            # - Update dense gain if we are in dense regime.

            self.end_epoch(epoch, test_loader)

            ##################################################
            # Record FLOPs :
            # calculate FLOPs for backward as in https://arxiv.org/pdf/1911.11134.pdf (Appendix H)
            forward_flops_per_sample, list_layers, module_names = get_macs_sparse(self.data[0], self.device, self.model)
            backward_flops_per_sample = 2 * forward_flops_per_sample
            trn_flops_per_sample = forward_flops_per_sample + backward_flops_per_sample
            total_trn_flops_epoch = trn_flops_per_sample * n_samples
            
            total_train_flops += total_trn_flops_epoch

            self.logging_function({'epoch': epoch, 'train FLOPs per sample': trn_flops_per_sample})
            self.logging_function({'epoch': epoch, 'train FLOPs epoch': total_trn_flops_epoch})
            self.logging_function({'epoch': epoch, 'test FLOPs per sample': forward_flops_per_sample})
            ####################################################

            current_lr = self.trainers[0].optim_lr
            self.logging_function({'epoch': epoch, 'lr': current_lr})
            
            val_loss, val_correct = self.get_eval_stats(epoch, test_loader)

            val_acc = 1.0 * val_correct / len(test_loader.dataset)
            best_sparse = False
            best_dense = False
            scheduled = False
            if self.pruned_state == "sparse":
                if int(self.current_sparsity * 10000) > int(self.best_val_acc["sparse"]["sparsity"] * 10000):
                    best_sparse = True
                    self.best_val_acc["sparse"]["sparsity"] = self.current_sparsity
                    self.best_val_acc["sparse"]["val_acc"] = val_acc
                    logging.info("saving best sparse checkpoint with new sparsity")
                elif int(self.current_sparsity  * 10000) == int(self.best_val_acc["sparse"]["sparsity"] * 10000) \
                   and val_acc > self.best_val_acc["sparse"]["val_acc"]:
                    best_sparse = True
                    self.best_val_acc["sparse"]["val_acc"] = val_acc
                    logging.info("saving best sparse checkpoint")
            if self.pruned_state == "dense" and val_acc > self.best_val_acc["dense"]:
                best_dense = True
                self.best_val_acc["dense"] = val_acc
            if (epoch + 1)  % self.checkpoint_freq == 0:
                logging.info("scheduled checkpoint")
                scheduled = True

            save_checkpoint(epoch, self.model_config, self.model, self.trainers[0].optimizer,
                                self.trainers[0].lr_scheduler, self.run_dir,
                            is_best_sparse=best_sparse, is_best_dense=best_dense,
                            is_scheduled_checkpoint=scheduled)

        # end_epoch even if epochs==0
        logging.info('====>Final summary for the run:')
        val_correct = self.end_epoch(self.n_epochs, self.test_loader)

        get_flops(self.data[0], self.device, self.model)


        return val_correct, len(test_loader.dataset)

