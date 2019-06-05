import torch
from torch import nn
from torch import distributed
from torch import optim
from torch import utils
from torch import backends
from torch import autograd
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from tensorboardX import SummaryWriter
from apex import amp
from apex import optimizers
from apex import parallel
from darts import *
from ops import *
import numpy as np
import skimage
import argparse
import copy
import json
import time
import os

parser = argparse.ArgumentParser(description='DARTS: Differentiable Architecture Search')
parser.add_argument('--config', type=str, default='config.json')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--training', action='store_true')
parser.add_argument('--validation', action='store_true')
parser.add_argument('--evaluation', action='store_true')
parser.add_argument('--inference', action='store_true')
parser.add_argument('--local_rank', type=int)
args = parser.parse_args()

backends.cudnn.benchmark = True


class Dict(dict):

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


class Function(object):

    def __init__(self, function, name):
        self.function = function
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __str__(self):
        return self.name

class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1))
    def forward(self):
        return self.p ** 2

def main():

    # python -m torch.distributed.launch --nproc_per_node=NUM_GPUS main.py
    distributed.init_process_group(backend='nccl')

    with open(args.config) as file:
        config = Dict(json.load(file))
    config.update(vars(args))
    config.update(dict(
        world_size=distributed.get_world_size(),
        global_rank=distributed.get_rank(),
        device_count=torch.cuda.device_count(),
        local_rank=distributed.get_rank() % torch.cuda.device_count()
    ))
    print(f'config: {config}')

    torch.manual_seed(0)
    torch.cuda.set_device(config.local_rank)

    m = Module().cuda()

    output = m()

    output.backward()

    distributed.all_reduce(m.p.grad)
    print(m.p.grad)

    m.p.grad.copy_(m.p.grad / 2)
    distributed.all_reduce(m.p.grad)
    print(m.p.grad)

if __name__ == '__main__':
    main()
