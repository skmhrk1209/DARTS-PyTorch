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
from darts import *
from ops import *
import numpy as np
import argparse
import copy
import json
import time
import os

parser = argparse.ArgumentParser(description='ResNet50 training on Imagenet')
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
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]


def main():

    # python -m torch.distributed.launch --nproc_per_node=NUM_GPUS main_amp.py
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

    if config.global_rank == 0:
        print(f'config: {config}')

    torch.manual_seed(0)
    torch.cuda.set_device(config.local_rank)

    model = DARTS(
        operations=[
            lambda in_channels, out_channels, stride: SeparableConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=3,
                padding=1
            ),
            lambda in_channels, out_channels, stride: SeparableConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=5,
                padding=2
            ),
            lambda in_channels, out_channels, stride: DilatedConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=3,
                padding=2,
                dilation=2
            ),
            lambda in_channels, out_channels, stride: DilatedConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=5,
                padding=4,
                dilation=2
            ),
            lambda in_channels, out_channels, stride: nn.AvgPool2d(
                stride=stride,
                kernel_size=3,
                padding=1
            ),
            lambda in_channels, out_channels, stride: nn.MaxPool2d(
                stride=stride,
                kernel_size=3,
                padding=1
            ),
            lambda in_channels, out_channels, stride: nn.Identity() if stride == 1 else Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=1,
                padding=0
            ),
            lambda in_channels, out_channels, stride: Zero()
        ],
        num_nodes=6,
        num_cells=8,
        reduction_cells=[2, 5],
        num_channels=16,
        num_classes=10
    ).cuda()

    network = model.network
    architecture = model.architecture

    criterion = nn.CrossEntropyLoss(reduction='mean').cuda()
    network_optimizer = torch.optim.SGD(
        params=network.parameters(),
        lr=config.network_lr,
        momentum=config.network_momentum,
        weight_decay=config.network_weight_decay
    )
    architecture_optimizer = torch.optim.Adam(
        params=architecture.parameters(),
        lr=config.architecture_lr,
        betas=(config.architecture_beta1, config.architecture_beta2),
        weight_decay=config.architecture_weight_decay
    )

    model = nn.parallel.distributed.DistributedDataParallel(
        module=model,
        device_ids=[config.local_rank],
        output_device=config.local_rank,
        find_unused_parameters=True
    )

    last_epoch = -1
    global_step = 0
    if config.checkpoint:
        checkpoint = Dict(torch.load(config.checkpoint))
        model.load_state_dict(checkpoint.model_state_dict)
        network_optimizer.load_state_dict(checkpoint.network_optimizer_state_dict)
        architecture_optimizer.load_state_dict(checkpoint.architecture_optimizer_state_dict)
        last_epoch = checkpoint.last_epoch
        global_step = checkpoint.global_step

    if config.global_rank == 0:
        os.makedirs(config.checkpoint_directory, exist_ok=True)
        os.makedirs(config.event_directory, exist_ok=True)
        summary_writer = SummaryWriter(config.event_directory)

    if config.training:

        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=network_optimizer,
            T_max=config.num_epochs,
            eta_min=config.network_lr_min,
            last_epoch=last_epoch
        )

        train_dataset = datasets.CIFAR10(
            root="cifar10",
            train=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.49139968, 0.48215827, 0.44653124),
                    std=(0.24703233, 0.24348505, 0.26158768)
                )
            ]),
            download=True
        )
        val_dataset = datasets.CIFAR10(
            root="cifar10",
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.49139968, 0.48215827, 0.44653124),
                    std=(0.24703233, 0.24348505, 0.26158768)
                )
            ]),
            download=True
        )

        train_sampler = utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = utils.data.distributed.DistributedSampler(val_dataset)

        train_data_loader = utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=config.local_batch_size,
            sampler=train_sampler,
            num_workers=config.num_workers,
            pin_memory=True
        )
        val_data_loader = utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=config.local_batch_size,
            sampler=val_sampler,
            num_workers=config.num_workers,
            pin_memory=True
        )

        training_begin = time.time()

        for epoch in range(last_epoch + 1, config.num_epochs):

            model.train()
            train_sampler.set_epoch(epoch)

            for local_step, ((train_images, train_labels), (val_images, val_labels)) in enumerate(zip(train_data_loader, val_data_loader)):

                step_begin = time.time()

                train_images = train_images.cuda()
                train_labels = train_labels.cuda()

                val_images = val_images.cuda()
                val_labels = val_labels.cuda()

                old_network_parameters  = [parameter.clone() for parameter in network.parameters()]

                network_optimizer.zero_grad()

                train_logits = model(train_images)
                train_loss = criterion(train_logits, train_labels)
                train_loss.backward()

                network_optimizer.step()

                network_optimizer.zero_grad()
                architecture_optimizer.zero_grad()

                val_logits = model(val_images)
                val_loss = criterion(val_logits, val_labels)
                val_loss.backward()

                new_network_parameters  = [parameter.clone() for parameter in network.parameters()]
                new_network_gradients  = [parameter.grad.clone() for parameter in network.parameters()]

                gradient_norm = torch.norm(torch.cat([gradient.reshape(-1) for gradient in new_network_gradients]))

                for parameter, old_parameter, new_gradient in zip(network.parameters(), old_network_parameters, new_network_gradients):
                    parameter.data.copy_(old_parameter + new_gradient * config.epsilon)

                train_logits = model(train_images)
                train_loss = criterion(train_logits, train_labels) * -(config.network_lr / (2 * config.epsilon / gradient_norm))
                train_loss.backward()

                for parameter, old_parameter, new_gradient in zip(network.parameters(), old_network_parameters, new_network_gradients):
                    parameter.data.copy_(old_parameter - new_gradient * config.epsilon)

                train_logits = model(train_images)
                train_loss = criterion(train_logits, train_labels) * +(config.network_lr / (2 * config.epsilon / gradient_norm))
                train_loss.backward()

                for parameter, new_parameter in zip(network.parameters(), new_network_parameters):
                    parameter.data.copy_(new_parameter)

                architecture_optimizer.step()

                val_predictions = torch.argmax(val_logits, dim=1)
                val_accuracy = torch.mean((val_predictions == val_labels).float()) / config.world_size

                distributed.all_reduce(val_loss)
                distributed.all_reduce(val_accuracy)

                step_end = time.time()

                if config.global_rank == 0:
                    summary_writer.add_scalars(
                        main_tag='loss',
                        tag_scalar_dict=dict(val=val_loss),
                        global_step=global_step
                    )
                    summary_writer.add_scalars(
                        main_tag='accuracy',
                        tag_scalar_dict=dict(val=val_accuracy),
                        global_step=global_step
                    )
                    print(f'[training] epoch: {epoch} global_step: {global_step} local_step: {local_step} '
                          f'val_loss: {val_loss:.4f} val_accuracy: {val_accuracy:.4f} [{step_end - step_begin:.4f}s]')

                global_step += 1

            torch.save(dict(
                model_state_dict=model.state_dict(),
                network_optimizer_state_dict=network_optimizer.state_dict(),
                architecture_optimizer_state_dict=architecture_optimizer.state_dict(),
                last_epoch=last_epoch,
                global_step=global_step
            ), f'{config.checkpoint_directory}/epoch_{epoch}')

            lr_scheduler.step()

        training_end = time.time()
        if config.global_rank == 0:
            print(f'training finished [{training_end - training_begin:.4f}s]')

    if config.global_rank == 0:
        summary_writer.close()


if __name__ == '__main__':
    main()
