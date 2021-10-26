"""
Example models to train and prune.
Interface is provided by the get_model function.
"""
import torch

from models.resnet_imagenet import *
from models.resnet_cifar10 import *
from models.resnet_cifar10_swish import *
from models.efficientnet import *
from models.simplenet import *
from models.logistic_regression import *
from models.resnet_mixed_cifar10 import *
from models.resnet_mixed_imagenet import *
from models.wide_resnet_imagenet import *
from models.resnet_mixed_imagenet_static import *
from models.mobilenet import *
from models.wide_resnet_cifar import *

from torchvision.models import resnet50 as torch_resnet50
from torchvision.models import vgg16_bn
from torchvision.models import vgg11, vgg19, vgg11_bn

import pdb

CIFAR10_MODELS = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet20_sw', 'resnet32_sw', 'resnet44_sw', 'resnet56_sw', 'resnet20_mixed']
CIFAR100_MODELS = ['wideresnet', 'resnet20']
IMAGENET_MODELS = ['resnet18', 'resnet34', 'resnet50', 'vgg19', 'resnet101', 'resnet152', 'resnet50_mixed',
    'wide_resnet50_2_mixed', 'mobilenet', 'resnet50_mixed_static']
MNIST_MODELS = ['mnist_mlp']

def get_model(name, dataset, pretrained=False, use_butterfly=False, 
              use_se=False, se_ratio=None, kernel_sizes=None, p=None, hidden_size=500, args=None):
    if name.startswith('efficientnet'):
        which = name[-2:]
        return construct_effnet_for_dataset(which, dataset, pretrained, use_butterfly)
    if name.startswith('resnet') and dataset == 'cifar10':
        assert_no_butterfly(use_butterfly)
        if name == 'resnet50' and pretrained:
            return torch_resnet50(pretrained=True)
        try:
            if 'mixed' in name:
                assert_use_se(name, use_se)
                return globals()[name](**{'use_se':use_se, 'se_ratio': se_ratio})
            return globals()[name]()
        except:
            raise ValueError(f'Model {name} is not supported for {dataset}, list of supported: {", ".join(CIFAR10_MODELS)}')
    if 'resnet' in name and any([dataset == 'imagenet', dataset == 'imagenette']):
        assert_no_butterfly(use_butterfly)
        if 'mixed' in name:
            assert_use_se(name, use_se)
            kwargs_dict = {'use_se':use_se, 'se_ratio': se_ratio, 
                           'kernel_sizes': kernel_sizes, 'p': p}
            if dataset == 'imagenette':
                kwargs_dict['num_classes'] = 10
            return globals()[name](**kwargs_dict)
        return globals()[name](pretrained)
        #print("Use torchvision model!")
        #return torch_resnet50(pretrained=False)

        try:
            if 'mixed' in name:
                assert_use_se(name, use_se)
                kwargs_dict = {'use_se':use_se, 'se_ratio': se_ratio, 
                               'kernel_sizes': kernel_sizes, 'p': p}
                if dataset == 'imagenette':
                    kwargs_dict['num_classes'] = 10
                return globals()[name](**kwargs_dict)
            return globals()[name](pretrained)
        except:
            raise ValueError(f'Model {name} is not supported for {dataset}, list of supported: {", ".join(IMAGENET_MODELS)}')
    if name == 'simplenet' and dataset == 'mnist':
        return SimpleNet(dataset, use_butterfly)
    if name=='mnist_mlp' and dataset=='mnist':
        return MnistMLP(hidden_size=hidden_size)
    if name == 'simplenet' and dataset == 'cifar10':
        assert_no_butterfly(use_butterfly)
        return SimpleNet_cifar()
    if name == 'logistic_regression' and dataset == 'blobs':
        assert_no_butterfly(use_butterfly)
        return LogisticRegression()
    if 'mobilenet' in name:
        return globals()[name]()

    if name=='wideresnet' and dataset=='cifar100':
        model = Wide_ResNet(28, 10, 0.3, 100)
        return model
    if name=='resnet20' and dataset=='cifar100':
        model = resnet20(num_classes=100)
        return model

    return globals()[name](pretrained)   
        
    raise NotImplementedError

def assert_use_se(name, use_se):
    if any([name == 'resnet20_mixed', name == 'resnet50_mixed', 
        name == 'wide_resnet50_2_mixed']) and use_se:
        return
    elif use_se:
        raise NotImplementedError("SELayer is only implemented for mixed resnet20 model")

def assert_no_butterfly(use_butterfly):
    if use_butterfly:
        raise NotImplementedError("Butterfly convolutions are not yet supported for this model.")

if __name__ == '__main__':
    get_model('resnet', 'cifar10', pretrained=False)
