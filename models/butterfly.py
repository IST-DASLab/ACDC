"""
Butterfly convolution implementation.

TODO: 
* The asserts here assume that the output channel sizes are 2^k.
  Check if it is necessary.
"""

from typing import Callable
import math
import numpy as np
import torch
from torch import Tensor
from torch.nn import (
    init, Module, Sequential, Conv2d, BatchNorm2d, ReLU
)


class ConvBN(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                cardinality: int, include_batchnorm: bool = True, act_construct: Callable = ReLU):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                           groups=cardinality, bias=not include_batchnorm)
        self.bn = BatchNorm2d(out_channels) if include_batchnorm else None
        self.act = act_construct() if act_construct is not None else None
        self.initialize()

    def forward(self, inp: Tensor):
        out = self.conv(inp)
        if self.bn is not None:
            out = self.bn(out)
        if self.act is not None:
        	out = self.act(out)
        return out

    def initialize(self):
        _init_conv(self.conv)
        if self.bn is not None:
            _init_batch_norm(self.bn)

def _butterfly_permutation(num_channels: int, cardinality: int, depth: int) -> Tensor:
    wing_size = 2 ** depth
    groups = [index + wing_size if index % (2 * wing_size) / (2 * wing_size) < 0.5 else index - wing_size
        for index in range(cardinality)]
    group_size = num_channels // cardinality
    indices = []

    for group in groups:
        indices.extend([group * group_size + index for index in range(group_size)])
    return torch.tensor(indices, dtype=torch.int64)


class ButterflyConvStep(Module):
    def __init__(self, depth: int, in_channels: int, out_channels: int, kernel_size: int, stride: int,
                 padding: int, cardinality: int, include_batchnorm: bool = True, act_construct: Callable = ReLU):
        super().__init__()
        self.body = ConvBN(in_channels, out_channels, kernel_size, stride, padding, cardinality,
                           include_batchnorm, act_construct)
        self.wing = ConvBN(in_channels, out_channels, kernel_size, stride, padding, cardinality,
                           include_batchnorm, act_construct)
        self.perm = _butterfly_permutation(out_channels, cardinality, depth)

    def forward(self, inp: Tensor):
        out = self.body(inp)
        cross = self.wing(inp)[:, self.perm]
        out = out + cross
        return out


class ButterflyConv(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                stride: int = 1, padding: int = 0,
                cardinality: int = 2, include_batchnorm: bool = True, act_construct: Callable = ReLU,
                include_last_act: bool = True):
        """
        Arguments:
            in_channels {int} - should be a power of 2
            out_channels {int} - should be a power of 2
            kernel_size {int}
            stride {int}
            padding {int}
            cardinality {int or None} - number of steps to convolve with butterfly before falling back to usual conv.
                Should be a divisor of both input and output channel sizes. If None, will be set to GCD(in_channels, out_channels)
        """
        super().__init__()

        if cardinality is None:
            cardinality = int(np.gcd(in_channels, out_channels))
        if cardinality < 2:
            raise Exception('cardinality must be greater than 1, given {}'.format(cardinality))

        ButterflyConv._check_valid(in_channels, out_channels, cardinality)
        steps = []
        num_steps = round(math.log2(cardinality))

        for depth in range(num_steps):
            last = depth == num_steps - 1
            steps.append(
                ButterflyConvStep(
                    depth, in_channels, out_channels, kernel_size, stride, padding,
                    cardinality, include_batchnorm,
                    act_construct=act_construct if not last or include_last_act else None)
            )
            in_channels = out_channels
            stride = 1

        self.steps = Sequential(*steps)

    def forward(self, inp: Tensor):
        out = self.steps(inp)
        return out

    @staticmethod
    def _check_valid(in_channels: int, out_channels: int, cardinality: int):
        if (out_channels & (out_channels - 1)) != 0:
            raise Exception(f'out_channels must be a power of 2, given {out_channels}')

        if out_channels % cardinality != 0:
            raise Exception('out_channels must be divisible by the cardinality')

        if in_channels % cardinality != 0:
            raise Exception('in_channels must be divisible by the cardinality')


def _init_conv(conv: Conv2d):
    init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')


def _init_batch_norm(norm: BatchNorm2d, weight_const: float = 1.0):
    init.constant_(norm.weight, weight_const)
    init.constant_(norm.bias, 0.0)


def run_tests():
    raise NotImplementedError


if __name__ == "__main__":
    run_tests()
