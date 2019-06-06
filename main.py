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
import functools
import argparse
import copy
import json
import time
import os


class Dict(dict):
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]


# python -m torch.distributed.launch --nproc_per_node=NUM_GPUS main.py
def main(args):

    backends.cudnn.benchmark = True

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

    model = DARTS(
        operations=dict(
            sep_conv_3x3=functools.partial(SeparableConv2d, kernel_size=3, padding=1, affine=False),
            sep_conv_5x5=functools.partial(SeparableConv2d, kernel_size=5, padding=2, affine=False),
            dil_conv_3x3=functools.partial(DilatedConv2d, kernel_size=3, padding=2, dilation=2, affine=False),
            dil_conv_5x5=functools.partial(DilatedConv2d, kernel_size=5, padding=4, dilation=2, affine=False),
            avg_pool_3x3=functools.partial(AvgPool2d, kernel_size=3, padding=1),
            max_pool_3x3=functools.partial(MaxPool2d, kernel_size=3, padding=1),
            identity=functools.partial(Conv2d, kernel_size=1, padding=0, affine=False),
            zero=functools.partial(Zero)
        ),
        num_nodes=6,
        num_input_nodes=2,
        num_cells=8,
        reduction_cells=[2, 5],
        num_channels=16,
        num_classes=10
    ).cuda()

    criterion = nn.CrossEntropyLoss(reduction='mean').cuda()

    config.global_batch_size = config.local_batch_size * config.world_size
    config.network_lr = config.network_lr * config.global_batch_size / config.global_batch_denom
    config.architecture_lr = config.architecture_lr * config.global_batch_size / config.global_batch_denom

    network_optimizer = torch.optim.SGD(
        params=model.network.parameters(),
        lr=config.network_lr,
        momentum=config.network_momentum,
        weight_decay=config.network_weight_decay
    )
    architecture_optimizer = torch.optim.Adam(
        params=model.architecture.parameters(),
        lr=config.architecture_lr,
        betas=(config.architecture_beta1, config.architecture_beta2),
        weight_decay=config.architecture_weight_decay
    )

    model, [network_optimizer, architecture_optimizer] = amp.initialize(
        models=model,
        optimizers=[network_optimizer, architecture_optimizer],
        opt_level=config.opt_level
    )

    # nn.parallel.DistributedDataParallel and apex.parallel.DistributedDataParallel don't support multiple backward passes.
    # This means `all_reduce` is executed when the first backward pass.
    # So, we manually reduce all gradients.
    # model = parallel.DistributedDataParallel(model, delay_allreduce=True)
    def average_gradients(parameters):
        for parameter in parameters:
            distributed.all_reduce(parameter.grad.data)
            parameter.grad.data /= config.world_size

    def average_tensor(tensors):
        for tensor in tensors:
            distributed.all_reduce(tensor.data)
            tensor.data /= config.world_size

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
        os.makedirs(config.architecture_directory, exist_ok=True)
        summary_writer = SummaryWriter(config.event_directory)

    if config.training:

        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=network_optimizer,
            T_max=config.num_epochs,
            eta_min=config.network_lr_min,
            last_epoch=last_epoch
        )

        train_dataset = datasets.CIFAR10(
            root='cifar10',
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
            root='cifar10',
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

        for epoch in range(last_epoch + 1, config.num_epochs):

            model.train()
            train_sampler.set_epoch(epoch)

            for local_step, ((train_images, train_labels), (val_images, val_labels)) in enumerate(zip(train_data_loader, val_data_loader)):

                step_begin = time.time()

                train_images = train_images.cuda()
                train_labels = train_labels.cuda()

                val_images = val_images.cuda()
                val_labels = val_labels.cuda()

                # Sace current network parameters and optimizer.
                network_state_dict = copy.deepcopy(model.network.state_dict())
                network_optimizer_state_dict = copy.deepcopy(network_optimizer.state_dict())

                # `w` in the paper.
                network_parameters = [copy.deepcopy(parameter) for parameter in model.network.parameters()]

                # Approximate w*(α) by adapting w using only a single training step,
                # without solving the inner optimization completely by training until convergence.
                # ----------------------------------------------------------------
                network_optimizer.zero_grad()

                train_logits = model(train_images)
                train_loss = criterion(train_logits, train_labels)
                with amp.scale_loss(train_loss, network_optimizer) as scaled_train_loss:
                    scaled_train_loss.backward()

                average_gradients(model.network.parameters())
                network_optimizer.step()
                # ----------------------------------------------------------------

                # Apply chain rule to the approximate architecture gradient.
                # Backward validation loss, but don't update approximate parameter w'.
                # ----------------------------------------------------------------
                network_optimizer.zero_grad()
                architecture_optimizer.zero_grad()

                val_logits = model(val_images)
                val_loss = criterion(val_logits, val_labels)
                with amp.scale_loss(val_loss, [network_optimizer, architecture_optimizer]) as scaled_val_loss:
                    scaled_val_loss.backward()

                network_gradients = [copy.deepcopy(parameter.grad) for parameter in model.network.parameters()]
                gradient_norm = torch.norm(torch.cat([gradient.reshape(-1) for gradient in network_gradients]))
                # ----------------------------------------------------------------

                # Avoid calculate hessian-vector product using the finite difference approximation.
                # ----------------------------------------------------------------
                for parameter, prev_parameter, prev_gradient in zip(model.network.parameters(), network_parameters, network_gradients):
                    parameter.data = (prev_parameter + prev_gradient * config.epsilon).data

                train_logits = model(train_images)
                train_loss = criterion(train_logits, train_labels) * -(config.network_lr / (2 * config.epsilon / gradient_norm))
                with amp.scale_loss(train_loss, architecture_optimizer) as scaled_train_loss:
                    scaled_train_loss.backward()

                for parameter, prev_parameter, prev_gradient in zip(model.network.parameters(), network_parameters, network_gradients):
                    parameter.data = (prev_parameter - prev_gradient * config.epsilon).data

                train_logits = model(train_images)
                train_loss = criterion(train_logits, train_labels) * (config.network_lr / (2 * config.epsilon / gradient_norm))
                with amp.scale_loss(train_loss, architecture_optimizer) as scaled_train_loss:
                    scaled_train_loss.backward()
                # ----------------------------------------------------------------

                # Finally, update architecture parameter.
                average_gradients(model.architecture.parameters())
                architecture_optimizer.step()

                # Restore previous network parameters and optimizer.
                model.network.load_state_dict(network_state_dict)
                network_optimizer.load_state_dict(network_optimizer_state_dict)

                # Update network parameter.
                # ----------------------------------------------------------------
                network_optimizer.zero_grad()

                train_logits = model(train_images)
                train_loss = criterion(train_logits, train_labels)
                with amp.scale_loss(train_loss, network_optimizer) as scaled_train_loss:
                    scaled_train_loss.backward()

                average_gradients(model.network.parameters())
                network_optimizer.step()
                # ----------------------------------------------------------------

                train_predictions = torch.argmax(train_logits, dim=1)
                train_accuracy = torch.mean((train_predictions == train_labels).float())

                val_predictions = torch.argmax(val_logits, dim=1)
                val_accuracy = torch.mean((val_predictions == val_labels).float())

                average_tensor([train_loss, val_loss, train_accuracy, val_accuracy])

                step_end = time.time()

                if config.global_rank == 0:
                    summary_writer.add_scalars(
                        main_tag='loss',
                        tag_scalar_dict=dict(
                            train=train_loss,
                            val=val_loss
                        ),
                        global_step=global_step
                    )
                    summary_writer.add_scalars(
                        main_tag='accuracy',
                        tag_scalar_dict=dict(
                            train=train_accuracy,
                            val=val_accuracy
                        ),
                        global_step=global_step
                    )
                    print(f'[training] epoch: {epoch} global_step: {global_step} local_step: {local_step} '
                          f'train_loss: {train_loss:.4f} train_accuracy: {train_accuracy:.4f} '
                          f'val_loss: {val_loss:.4f} val_accuracy: {val_accuracy:.4f} [{step_end - step_begin:.4f}s]')

                global_step += 1

            if config.global_rank == 0:

                torch.save(dict(
                    model_state_dict=model.state_dict(),
                    network_optimizer_state_dict=network_optimizer.state_dict(),
                    architecture_optimizer_state_dict=architecture_optimizer.state_dict(),
                    last_epoch=last_epoch,
                    global_step=global_step
                ), f'{config.checkpoint_directory}/epoch_{epoch}')

                summary_writer.add_image(
                    tag='architecture/normal',
                    img_tensor=skimage.io.imread(model.draw_normal_architecture(
                        num_operations=2,
                        name=f'normal_cell_{epoch}',
                        directory=config.architecture_directory
                    )),
                    global_step=global_step,
                    dataformats='HWC'
                )
                summary_writer.add_image(
                    tag='architecture/reduction',
                    img_tensor=skimage.io.imread(model.draw_reduction_architecture(
                        num_operations=2,
                        name=f'reduction_cell_{epoch}',
                        directory=config.architecture_directory
                    )),
                    global_step=global_step,
                    dataformats='HWC'
                )

            lr_scheduler.step()

    if config.global_rank == 0:
        summary_writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DARTS: Differentiable Architecture Search')
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()

    main(args)
