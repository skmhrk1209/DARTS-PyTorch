import torch
from torch import nn


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 affine, preactivation=True, **kwargs):

        super().__init__()

        self.module = nn.Sequential(
            nn.ReLU() if preactivation else Identity(),
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


class Zero(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, input):
        return 0.0


class Identity(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, input):
        return input
