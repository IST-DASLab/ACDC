"""
File implementing static mixed convolution operations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy

from copy import deepcopy



class _Conv2dSamePaddingStatic(nn.Conv2d):
    """ Class implementing 2d adaptively padded convolutions """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True, 
                 pad_w=None, pad_h=None):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, 0, dilation, groups, bias)
        self.pad_w, self.pad_h = pad_w, pad_h

    def forward(self, x):
        x = F.pad(x, [self.pad_w // 2, self.pad_w - self.pad_w // 2, 
                      self.pad_h // 2, self.pad_h - self.pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, 
                        self.padding, self.dilation, self.groups)


def _get_split_sizes_old(channels, num_kernels):
        sizes = [channels // num_kernels] * num_kernels
        sizes[0] += channels % num_kernels
        return sizes 


def _get_split_sizes(channels, num_kernels, p):
    sizes = []
    for i in range(1, num_kernels + 1):
        progress = min(float(i) / num_kernels , 1)
        remaining_progress = (1.0 - progress) ** p
        sizes.append(int(channels - channels * remaining_progress) 
                     - numpy.sum(sizes).astype(int))
    sizes[0] += channels - numpy.sum(sizes)
    return sizes


class Conv2dMixedSizeStatic(nn.Module):
    """
    Class that implements mixed 2d convolution
    It now support only the uniform channel groups
    and same **kwargs for all convs
    """
    def __init__(self, in_channels, out_channels, kernel_sizes, p, pads, **kwargs):
        """
        Args:
          in_channels: number of input channels
          out_channels: number of output channels
          kernel_sizes: list of kernel size to consider in mixing
          kwargs: possibly containing {'stride': int_1, 
                                       'dilation': int_2, 
                                       'groups': int_3,
                                       'bias': bool_1}
        """
        super(Conv2dMixedSizeStatic, self).__init__()

        self.p = p
        num_kernels = len(kernel_sizes)
        self.in_channels = _get_split_sizes(in_channels, num_kernels, self.p)
        self.out_channels = _get_split_sizes(out_channels, num_kernels, self.p)
        self.kernel_sizes = kernel_sizes
    
        for i in range(num_kernels):
            new_kwargs = deepcopy(kwargs)
            new_kwargs['pad_w'] = pads[i][0]
            new_kwargs['pad_h'] = pads[i][1]

            args = (self.in_channels[i], self.out_channels[i], self.kernel_sizes[i])
            setattr(self, f'conv{self.kernel_sizes[i]}x{self.kernel_sizes[i]}', 
                _Conv2dSamePaddingStatic(*args, **new_kwargs))

    def forward(self, x):
        outputs, chunks = [], torch.split(x, self.in_channels, dim=1)
        for i, chunk in enumerate(chunks):
            output = getattr(self, f'conv{self.kernel_sizes[i]}x{self.kernel_sizes[i]}')(chunk)
            outputs.append(output)
        return torch.cat(outputs, dim=1)


if __name__ == '__main__':
    print(_get_split_sizes(64,3))