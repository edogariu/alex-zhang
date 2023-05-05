from typing import List
import os
import numpy as np

import torch
import torch.nn as nn

TOP_DIR_NAME = os.path.abspath('.')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SPLIT_INTERVALS = {'train': 0.75,
                   'val': 0.2,
                   'test': 0.05}

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def exponential_linspace_int(start, end, num, divisible_by=1) -> List[int]:
    """Exponentially increasing values of integers."""
    base = np.exp(np.log(end / start) / (num - 1))
    return [int(np.round(start * base**i / divisible_by) * divisible_by) for i in range(num)]

def zero_module(module: nn.Module) -> nn.Module:
    """
    Helpful method that zeros all the parameters in a nn.Module, used for initialization
    """
    for param in module.parameters():
        param.detach().zero_()
    return module