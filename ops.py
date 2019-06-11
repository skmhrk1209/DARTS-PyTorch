import torch
from torch import nn


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 affine, preactivation=True, **kwargs):

        super().__init__()

        self.module = nn.Sequential(
            nn.ReLU() if preactivation else nn.Identity(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
                affine=affine
            )
        )

    def forward(self, input):
        return self.module(input)


class DilatedConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, affine, preactivation=True, **kwargs):

        super().__init__()

        self.module = nn.Sequential(
            nn.ReLU() if preactivation else nn.Identity(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=False
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
                affine=affine
            )
        )

    def forward(self, input):
        return self.module(input)


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 affine, preactivation=True, **kwargs):

        super().__init__()

        self.module = nn.Sequential(
            nn.ReLU() if preactivation else nn.Identity(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=in_channels,
                affine=affine
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=in_channels,
                bias=False
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels,
                affine=affine
            )
        )

    def forward(self, input):
        return self.module(input)


class AvgPool2d(nn.Module):

    def __init__(self, kernel_size, stride, padding, **kwargs):

        super().__init__()

        self.module = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, input):
        return self.module(input)


class MaxPool2d(nn.Module):

    def __init__(self, kernel_size, stride, padding, **kwargs):

        super().__init__()

        self.module = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, input):
        return self.module(input)


class Identity(nn.Module):

    def __init__(self, in_channels, out_channels, stride, affine, preactivation=True, **kwargs):

        super().__init__()

        self.module = nn.Identity() if stride == 1 and in_channels == out_channels else Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            affine=affine,
            preactivation=preactivation
        )

    def forward(self, input):
        return self.module(input)


class Zero(nn.Module):

    def __init__(self, **kwargs):

        super().__init__()

    def forward(self, input):
        return 0.0


class ScheduledDropPath(nn.Module):

    def __init__(self, drop_prob, gamma):

        super().__init__()

        self.drop_prob = drop_prob

    def forward(self, input):

        if self.drop_prob > 0:
            gamma = self.gamma(self.epoch) if callable(self.gamma) else self.gamma
            drop_prob = self.drop_prob * gamma
            keep_prob = 1 - drop_prob
            input = input * input.new_full((input.size(0), 1, 1, 1), keep_prob).bernoulli()
            input = input / keep_prob

        return input

    def set_epoch(self, epoch):

        self.epoch = epoch


class Cutout(object):

    def __init__(self, size):

        self.size = size

    def __call__(self, image):

        y_min = torch.randint(image.size(-2) - self.size[-2], (1,))
        x_min = torch.randint(image.size(-1) - self.size[-1], (1,))

        y_max = y_min + self.size[-2]
        x_max = x_min + self.size[-1]

        image[..., y_min:y_max, x_min:x_max] = 0

        return image
