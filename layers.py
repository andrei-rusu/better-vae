import numpy as np
import math

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

import utils

class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class Debug(nn.Module):
    """
    Executes a lambda function and then returns the input. Useful for debugging.
    """
    def __init__(self, lambd):
        super(Debug, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        self.lambd(x)
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view( (input.size(0),) + self.shape)


class Block(nn.Module):

    def __init__(self, in_channels, channels, num_convs = 3, kernel_size = 3, batch_norm=False, use_weight=True, use_res=True, deconv=False):
        super().__init__()

        layers = []
        self.use_weight = use_weight
        self.use_res = use_res

        padding = int(math.floor(kernel_size / 2))

        self.upchannels = nn.Conv2d(in_channels, channels, kernel_size=1)

        for i in range(num_convs):
            if deconv:
                layers.append(nn.ConvTranspose2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=not batch_norm))
            else:
                layers.append(nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=not batch_norm))

            if batch_norm:
                layers.append(module=nn.BatchNorm2d(channels))

            layers.append(nn.ReLU())

        self.seq = nn.Sequential(*layers)

        if use_weight:
            self.weight = nn.Parameter(torch.randn(1))

    def forward(self, x):

        x = self.upchannels(x)

        out = self.seq(x)

        if not self.use_res:
            return out

        if not self.use_weight:
            return out + x

        return out + self.weight * x
    
# PixelCNN decoder layers

class CMaskedConv2d(nn.Module):
    """
    Masked convolution, with location independent conditional.
    """
    def __init__(self, input_size, conditional_size, channels, colors=3, self_connection=False, res_connection=True, hv_connection=True, gates=True, k=7, padding=3):

        super().__init__()

        assert (k // 2) * 2 == k - 1 # only odd numbers accepted

        self.gates = gates
        self.res_connection = res_connection
        self.hv_connection = hv_connection

        f = 2 if self.gates else 1

        self.vertical   = nn.Conv2d(channels,   channels*f, kernel_size=k, padding=padding, bias=False)
        self.horizontal = nn.Conv2d(channels,   channels*f, kernel_size=(1, k), padding=(0, padding), bias=False)
        self.tohori     = nn.Conv2d(channels*f, channels*f, kernel_size=1, padding=0, bias=False, groups=colors)
        self.tores      = nn.Conv2d(channels,   channels,   kernel_size=1, padding=0, bias=False, groups=colors)

        self.register_buffer('vmask', self.vertical.weight.data.clone())
        self.register_buffer('hmask', self.horizontal.weight.data.clone())

        self.vmask.fill_(1)
        self.hmask.fill_(1)

        # zero the bottom half rows of the vmask
        self.vmask[:, :, k // 2 :, :] = 0

        # zero the right half of the hmask
        self.hmask[:, :, :, k // 2:] = 0

        # Add connections to "previous" colors (G is allowed to see R, and B is allowed to see R and G)

        m = k // 2  # index of the middle of the convolution
        pc = channels // colors  # channels per color

        # print(self_connection + 0, self_connection, m)

        for c in range(0, colors):
            f, t = c * pc, (c+1) * pc

            if f > 0:
                self.hmask[f:t, :f, 0, m] = 1
                self.hmask[f+channels:t+channels, :f, 0, m] = 1

            # Connections to "current" colors (but not "future colors", R is not allowed to see G and B)
            if self_connection:
                self.hmask[f:t, :f+pc, 0, m] = 1
                self.hmask[f + channels:t + channels, :f+pc, 0, m] = 1

        print(self.hmask[:, :, 0, m])

        fr = utils.prod(conditional_size)
        to = utils.prod(input_size)

        # The conditional weights
        self.vhf = nn.Linear(fr, to)
        self.vhg = nn.Linear(fr, to)
        self.vvf = nn.Linear(fr, to)
        self.vvg = nn.Linear(fr, to)

    def forward(self, vxin, hxin, h):

        self.vertical.weight.data   *= self.vmask
        self.horizontal.weight.data *= self.hmask

        vx =   self.vertical.forward(vxin)
        hx = self.horizontal.forward(hxin)

        if self.hv_connection:
            hx = hx + self.tohori(vx)

        if self.gates:
            vx = self.gate(vx, h,  (self.vvf, self.vvg))
            hx = self.gate(hx, h, (self.vhf, self.vhg))

        if self.res_connection:
            hx = hxin + self.tores(hx)

        return vx, hx

    def gate(self, x, cond, weights):
        """
        Takes a batch x channels x rest... tensor and applies an LTSM-style gate activation.
        - The top half of the channels are fed through a tanh activation, functioning as the activated neurons
        - The bottom half are fed through a sigmoid, functioning as a mask
        - The two are element-wise multiplied, and the result is returned.
        Conditional and weights are used to compute a bias based on the conditional element
        :param x: The input tensor.
        :return: The input tensor x with the activation applied.
        """
        b, c, h, w = x.size()

        # compute conditional term
        vf, vg = weights

        tan_bias = vf(cond.view(b, -1)).view((b, c//2, h, w))
        sig_bias = vg(cond.view(b, -1)).view((b, c//2, h, w))

        # compute convolution term
        half = c // 2

        top = x[:, :half]
        bottom = x[:, half:]

        # apply gate and return
        return torch.tanh(top + tan_bias) * torch.sigmoid(bottom + sig_bias)
    
class LMaskedConv2d(nn.Module):
    """
    Masked convolution, with location dependent conditional.
    The conditional must be an 'image' tensor (BCHW) with the same resolution as the instance (no of channels can be different)
    """
    def __init__(self, input_size, conditional_channels, channels, colors=3, self_connection=False, res_connection=True, hv_connection=True, gates=True, k=7, padding=3):

        super().__init__()

        assert (k // 2) * 2 == k - 1 # only odd numbers accepted

        self.gates = gates
        self.res_connection = res_connection
        self.hv_connection = hv_connection

        f = 2 if self.gates else 1

        self.vertical   = nn.Conv2d(channels,   channels*f, kernel_size=k, padding=padding, bias=False)
        self.horizontal = nn.Conv2d(channels,   channels*f, kernel_size=(1, k), padding=(0, padding), bias=False)
        self.tohori     = nn.Conv2d(channels*f, channels*f, kernel_size=1, padding=0, bias=False, groups=colors)
        self.tores      = nn.Conv2d(channels,   channels,   kernel_size=1, padding=0, bias=False, groups=colors)

        self.register_buffer('vmask', self.vertical.weight.data.clone())
        self.register_buffer('hmask', self.horizontal.weight.data.clone())

        self.vmask.fill_(1)
        self.hmask.fill_(1)

        # zero the bottom half rows of the vmask
        self.vmask[:, :, k // 2 :, :] = 0

        # zero the right half of the hmask
        self.hmask[:, :, :, k // 2:] = 0

        # Add connections to "previous" colors (G is allowed to see R, and B is allowed to see R and G)

        m = k // 2  # index of the middle of the convolution
        pc = channels // colors  # channels per color

        # print(self_connection + 0, self_connection, m)

        for c in range(0, colors):
            f, t = c * pc, (c+1) * pc

            if f > 0:
                self.hmask[f:t, :f, 0, m] = 1
                self.hmask[f+channels:t+channels, :f, 0, m] = 1

            # Connections to "current" colors (but not "future colors", R is not allowed to see G and B)
            if self_connection:
                self.hmask[f:t, :f+pc, 0, m] = 1
                self.hmask[f + channels:t + channels, :f+pc, 0, m] = 1

        print(self.hmask[:, :, 0, m])

        # The conditional weights
        self.vhf = nn.Conv2d(conditional_channels, channels, 1)
        self.vhg = nn.Conv2d(conditional_channels, channels, 1)
        self.vvf = nn.Conv2d(conditional_channels, channels, 1)
        self.vvg = nn.Conv2d(conditional_channels, channels, 1)

    def forward(self, vxin, hxin, h):

        self.vertical.weight.data   *= self.vmask
        self.horizontal.weight.data *= self.hmask

        vx =   self.vertical.forward(vxin)
        hx = self.horizontal.forward(hxin)

        if self.hv_connection:
            hx = hx + self.tohori(vx)

        if self.gates:
            vx = self.gate(vx, h,  (self.vvf, self.vvg))
            hx = self.gate(hx, h,  (self.vhf, self.vhg))

        if self.res_connection:
            hx = hxin + self.tores(hx)

        return vx, hx

    def gate(self, x, cond, weights):
        """
        Takes a batch x channels x rest... tensor and applies an LTSM-style gate activation.
        - The top half of the channels are fed through a tanh activation, functioning as the activated neurons
        - The bottom half are fed through a sigmoid, functioning as a mask
        - The two are element-wise multiplied, and the result is returned.
        Conditional and weights are used to compute a bias based on the conditional element
        :param x: The input tensor.
        :return: The input tensor x with the activation applied.
        """
        b, c, h, w = x.size()

        # compute conditional term
        vf, vg = weights

        tan_bias = vf(cond)
        sig_bias = vg(cond)

        # compute convolution term
        b = x.size(0)
        c = x.size(1)

        half = c // 2

        top = x[:, :half]
        bottom = x[:, half:]

        # apply gate and return
        return torch.tanh(top + tan_bias) * torch.sigmoid(bottom + sig_bias)
    
    
# IntroVAE layers

class IntroReshape(nn.Module):

    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1, *self.shape)

class Add(nn.Module):

    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, residual):
        """
        :param x:
        :param residual:
        :return:
        """
        return x + residual

    def extra_repr(self):
        return "ResNet Element-wise Add Layer"


class ResBlk(nn.Module):

    def __init__(self, kernels, chs):
        """
        :param kernels: [1, 3, 3], as [kernel_1, kernel_2, kernel_3]
        :param chs: [ch_in, 64, 64, 64], as [ch_in, ch_out1, ch_out2, ch_out3]
        :return:
        """
        super(ResBlk, self).__init__()

        layers = []

        assert len(chs)-1 == len(kernels), "mismatching between chs and kernels"

        for idx in range(len(kernels)):
            layers.extend([
                nn.Conv2d(chs[idx], chs[idx+1], kernel_size=kernels[idx], stride=1,
                          padding=1 if kernels[idx]!=1 else 0), # no padding for kernel=1
                nn.BatchNorm2d(chs[idx+1]),
                nn.ReLU(inplace=True)
            ])

        self.net = nn.Sequential(*layers)

        self.shortcut = nn.Sequential()
        if chs[0] != chs[-1]: # convert from ch_int to ch_out3
            self.shortcut = nn.Sequential(
                nn.Conv2d(chs[0], chs[-1], kernel_size=1),
                nn.BatchNorm2d(chs[-1]),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        """
        :param x:
        :return:
        """
        res = self.net(x)
        x_ = self.shortcut(x)
        # print(x.shape, x_.shape, res.shape)
        return x_ + res
    
class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=True):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x