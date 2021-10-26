import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class StageExponentialLR(_LRScheduler):
    def __init__(self, optimizer, init_gamma, final_gamma, freq_reset, last_epoch=-1):
        self.init_gamma = init_gamma
        self.final_gamma = final_gamma
        self.freq_reset = freq_reset
        self.gamma = self.init_gamma
        self.mode_not_changed = True
        super(StageExponentialLR, self).__init__(optimizer, last_epoch)
        
    def change_mode(self):
        if self.mode_not_changed:
            self.gamma = self.final_gamma
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i]
            self.mode_not_changed = False

    def _should_reset(self):
        return (
            self.last_epoch % self.freq_reset == 0 
            and self.last_epoch != 0
            and self.mode_not_changed
        )

    def get_lr(self):
        # reset the lr each freq_reset calls and if mode is not changed
        if self._should_reset():
            return self.base_lrs

        if self.last_epoch == 0:
            return self.base_lrs

        return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]



class CosineLR(_LRScheduler):
    # Adapted from the cosine lr function here:
    # https://github.com/adityakusupati/STR/blob/master/utils/schedulers.py
    def __init__(self, optimizer, warmup_length, end_epoch, last_epoch=-1):
        self.warmup_length = warmup_length
        self.end_epoch = end_epoch
        super(CosineLR, self).__init__(optimizer, last_epoch)


    def _warmup_lr(self):
        lrs = [base_lr * (self.last_epoch + 1) / self.warmup_length for base_lr in self.base_lrs]
        return lrs

    def _cosine_lr(self):
        e = self.last_epoch - self.warmup_length
        es = self.end_epoch - self.warmup_length
        lrs = [0.5 * (1 + np.cos(np.pi * e / es)) * base_lr for base_lr in self.base_lrs]
        return lrs

    def get_lr(self):
        if self.last_epoch < self.warmup_length:
            updated_lrs = self._warmup_lr()
        else:
            updated_lrs = self._cosine_lr()
        return updated_lrs

    def _get_closed_form_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
