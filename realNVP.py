import argparse

import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as add_weight_norm
import torch.distributions as distributions
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt
import numpy as np

class WeightNormConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, 
        bias=True, weight_norm=True, scale=False):
        """Intializes a Conv2d augmented with weight normalization.

        (See torch.nn.utils.weight_norm for detail.)

        Args:
            in_dim: number of input channels.
            out_dim: number of output channels.
            kernel_size: size of convolving kernel.
            stride: stride of convolution.
            padding: zero-padding added to both sides of input.
            bias: True if include learnable bias parameters, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            scale: True if include magnitude parameters, False otherwise.
        """
        super(WeightNormConv2d, self).__init__()

        if weight_norm:
            self.conv = add_weight_norm(
                nn.Conv2d(in_dim, out_dim, kernel_size, 
                    stride=stride, padding=padding, bias=bias))
            if not scale:
                self.conv.weight_g.data = torch.ones_like(self.conv.weight_g.data)
                self.conv.weight_g.requires_grad = False    # freeze scaling
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, 
                stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim, bottleneck, down_factor, weight_norm):
        """Initializes a ResidualBlock.

        Args:
            dim: number of input features.
            bottleneck: True if use bottleneck, False otherwise.
            down_factor: by how much to reduce feature numbers in bottleneck.
            weight_norm: True if apply weight normalization, False otherwise.
        """
        super(ResidualBlock, self).__init__()
        
        self.in_block = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU())
        if bottleneck:
            mid_dim = dim // down_factor
            self.res_block = nn.Sequential(
                WeightNormConv2d(dim, mid_dim, (1, 1), stride=1, padding=0, 
                    bias=False, weight_norm=weight_norm, scale=False),
                nn.BatchNorm2d(mid_dim),
                nn.ReLU(),
                WeightNormConv2d(mid_dim, mid_dim, (3, 3), stride=1, padding=1, 
                    bias=False, weight_norm=weight_norm, scale=False),
                nn.BatchNorm2d(mid_dim),
                nn.ReLU(),
                WeightNormConv2d(mid_dim, dim, (1, 1), stride=1, padding=0, 
                    bias=True, weight_norm=weight_norm, scale=True))
        else:
            self.res_block = nn.Sequential(
                WeightNormConv2d(dim, dim, (3, 3), stride=1, padding=1, 
                    bias=False, weight_norm=weight_norm, scale=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                WeightNormConv2d(dim, dim, (3, 3), stride=1, padding=1, 
                    bias=True, weight_norm=weight_norm, scale=True))

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        return x + self.res_block(self.in_block(x))

class ResidualModule(nn.Module):
    def __init__(self, in_dim, dim, out_dim, 
        res_blocks, bottleneck, down_factor, skip, weight_norm):
        """Initializes a ResidualModule.

        Args:
            in_dim: number of input features.
            dim: number of features in residual blocks.
            out_dim: number of output features.
            res_blocks: number of residual blocks to use.
            bottleneck: True if use bottleneck, False otherwise.
            down_factor: by how much to reduce feature numbers in bottleneck.
            skip: True if use skip architecture, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
        """
        super(ResidualModule, self).__init__()
        self.res_blocks = res_blocks
        self.skip = skip
        
        if res_blocks > 0:
            self.in_block = WeightNormConv2d(in_dim, dim, (3, 3), stride=1, 
                padding=1, bias=True, weight_norm=weight_norm, scale=False)
            self.core_block = nn.ModuleList(
                [ResidualBlock(dim, bottleneck, down_factor, weight_norm) 
                for _ in range(res_blocks)])
            self.out_block = nn.Sequential(
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                WeightNormConv2d(dim, out_dim, (1, 1), stride=1, padding=0, 
                    bias=True, weight_norm=weight_norm, scale=True))
        
            if skip:
                self.in_skip = WeightNormConv2d(dim, dim, (1, 1), stride=1, 
                    padding=0, bias=True, weight_norm=weight_norm, scale=True)
                self.core_skips = nn.ModuleList(
                    [WeightNormConv2d(
                        dim, dim, (1, 1), stride=1, padding=0, bias=True, 
                        weight_norm=weight_norm, scale=True) 
                    for _ in range(res_blocks)])
        else:
            if bottleneck:
                self.block = nn.Sequential(
                    WeightNormConv2d(in_dim, dim, (1, 1), stride=1, padding=0, 
                        bias=False, weight_norm=weight_norm, scale=False),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(),
                    WeightNormConv2d(dim, dim, (3, 3), stride=1, padding=1, 
                        bias=False, weight_norm=weight_norm, scale=False),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(),
                    WeightNormConv2d(dim, out_dim, (1, 1), stride=1, padding=0, 
                        bias=True, weight_norm=weight_norm, scale=True))
            else:
                self.block = nn.Sequential(
                    WeightNormConv2d(in_dim, dim, (3, 3), stride=1, padding=1, 
                        bias=False, weight_norm=weight_norm, scale=False),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(),
                    WeightNormConv2d(dim, out_dim, (3, 3), stride=1, padding=1, 
                        bias=True, weight_norm=weight_norm, scale=True))

    def forward(self, x):
        """Forward pass.

        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        if self.res_blocks > 0:
            x = self.in_block(x)
            if self.skip:
                out = self.in_skip(x)
            for i in range(len(self.core_block)):
                x = self.core_block[i](x)
                if self.skip:
                    out = out + self.core_skips[i](x)
            if self.skip:
                x = out
            return self.out_block(x)
        else:
            return self.block(x)

class AbstractCoupling(nn.Module):
    def __init__(self, mask_config, coupling_bn):
        """Initializes an AbstractCoupling.

        Args:
            mask_config: mask configuration (see build_mask() for more detail).
        """
        super(AbstractCoupling, self).__init__()
        self.mask_config = mask_config
        self.coupling_bn = coupling_bn

    def build_mask(self, B, H, W, config=1.):
        """Builds a binary checkerboard mask.

        (Only for constructing masks for checkboard coupling layers.)

        Args:
            B: batch size.
            H: height of feature map.
            W: width of feature map.
            config:    mask configuration that determines which pixels to mask up.
                    if 1:        if 0:
                        1 0            0 1
                        0 1         1 0
        Returns:
            a binary mask (1: pixel on, 0: pixel off).
        """
        mask = np.arange(H).reshape((-1, 1)) + np.arange(W)
        mask = np.mod(config + mask, 2)
        mask = np.reshape(mask, [-1, 1, H, W])
        mask = np.tile(mask, [B, 1, 1, 1])
        return torch.tensor(mask.astype('float32'))

    def batch_stat(self, x):
        """Compute (spatial) batch statistics.

        Args:
            x: input minibatch.
        Returns:
            batch mean and variance.
        """
        mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
        var = torch.mean((x - mean)**2, dim=(0, 2, 3), keepdim=True)
        return mean, var

class CheckerboardAdditiveCoupling(AbstractCoupling):
    def __init__(self, in_out_dim, mid_dim, res_blocks, mask_config, 
        bottleneck, down_factor, skip, weight_norm, coupling_bn):
        """Initializes a CheckerboardAdditiveCoupling.

        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
            res_blocks: number of residual blocks to use.
            mask_config: mask configuration (see build_mask() for more detail).
            bottleneck: True if use bottleneck, False otherwise.
            down_factor: by how much to reduce feature numbers in bottleneck.
            skip: True if use skip architecture, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            coupling_bn: True if apply batch normalization on output of 
                coupling layer, False otherwise.
        """
        super(CheckerboardAdditiveCoupling, self).__init__(
            mask_config, coupling_bn)
        
        self.in_bn = nn.BatchNorm2d(in_out_dim)
        self.block = nn.Sequential(
            nn.ReLU(),
            ResidualModule(2*in_out_dim+1, mid_dim, in_out_dim, res_blocks, 
                bottleneck, down_factor, skip, weight_norm))
        self.out_bn = nn.BatchNorm2d(in_out_dim, affine=False)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """
        [B, _, H, W] = list(x.size())
        mask = self.build_mask(B, H, W, config=self.mask_config).cuda()
        x_ = self.in_bn(x * mask)
        x_ = torch.cat((x_, -x_), dim=1)
        x_ = torch.cat((x_, mask), dim=1)     # 2C+1 channels
        shift = self.block(x_) * (1. - mask)

        log_diag_J = torch.zeros_like(x)     # unit Jacobian determinant
        # See Eq(3) and Eq(4) in NICE and Section 3.7 in real NVP
        if reverse:
            if self.coupling_bn:
                mean, var = self.out_bn.running_mean, self.out_bn.running_var
                mean = mean.reshape((-1, 1, 1, 1)).transpose(0, 1)
                var = var.reshape((-1, 1, 1, 1)).transpose(0, 1)
                x = x * torch.exp(0.5 * torch.log(var + 1e-5) * (1. - mask)) \
                    + mean * (1. - mask)
            x = x - shift
        else:
            x = x + shift
            if self.coupling_bn:
                if self.training:
                    _, var = self.batch_stat(x)
                else:
                    var = self.out_bn.running_var
                    var = var.reshape((-1, 1, 1, 1)).transpose(0, 1)
                x = self.out_bn(x) * (1. - mask) + x * mask
                log_diag_J = log_diag_J - 0.5 * torch.log(var + 1e-5) * (1. - mask)
        return x, log_diag_J

class CheckerboardAffineCoupling(AbstractCoupling):
    def __init__(self, in_out_dim, mid_dim, res_blocks, mask_config, 
        bottleneck, down_factor, skip, weight_norm, coupling_bn):
        """Initializes a CheckerboardAffineCoupling.

        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
            res_blocks: number of residual blocks to use.
            mask_config: mask configuration (see build_mask() for more detail).
            bottleneck: True if use bottleneck, False otherwise.
            down_factor: by how much to reduce feature numbers in bottleneck.
            skip: True if use skip architecture, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            coupling_bn: True if apply batch normalization on output of 
                coupling layer, False otherwise.
        """
        super(CheckerboardAffineCoupling, self).__init__(
            mask_config, coupling_bn)

        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.in_bn = nn.BatchNorm2d(in_out_dim)
        self.block = nn.Sequential(        # 1st half of resnet: shift
            nn.ReLU(),                    # 2nd half of resnet: log_rescale
            ResidualModule(2*in_out_dim+1, mid_dim, 2*in_out_dim, res_blocks, 
                bottleneck, down_factor, skip, weight_norm))
        self.out_bn = nn.BatchNorm2d(in_out_dim, affine=False)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """
        [B, C, H, W] = list(x.size())
        mask = self.build_mask(B, H, W, config=self.mask_config).cuda()
        x_ = self.in_bn(x * mask)
        x_ = torch.cat((x_, -x_), dim=1)
        x_ = torch.cat((x_, mask), dim=1)    # 2C+1 channels
        (shift, log_rescale) = self.block(x_).split(C, dim=1)
        log_rescale = self.scale * torch.tanh(log_rescale) + self.scale_shift
        shift = shift * (1. - mask)
        log_rescale = log_rescale * (1. - mask)
        
        log_diag_J = log_rescale     # See Eq(6) in real NVP 
        # See Eq(7) and Eq(8) and Section 3.7 in real NVP
        if reverse:
            if self.coupling_bn:
                mean, var = self.out_bn.running_mean, self.out_bn.running_var
                mean = mean.reshape((-1, 1, 1, 1)).transpose(0, 1)
                var = var.reshape((-1, 1, 1, 1)).transpose(0, 1)
                x = x * torch.exp(0.5 * torch.log(var + 1e-5) * (1. - mask)) \
                    + mean * (1. - mask)
            x = (x - shift) * torch.exp(-log_rescale)
        else:
            x = x * torch.exp(log_rescale) + shift
            if self.coupling_bn:
                if self.training:
                    _, var = self.batch_stat(x)
                else:
                    var = self.out_bn.running_var
                    var = var.reshape((-1, 1, 1, 1)).transpose(0, 1)
                x = self.out_bn(x) * (1. - mask) + x * mask
                log_diag_J = log_diag_J - 0.5 * torch.log(var + 1e-5) * (1. - mask)
        return x, log_diag_J

class CheckerboardCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, res_blocks, mask_config, 
        bottleneck, down_factor, skip, weight_norm, coupling_bn, affine):
        """Initializes a CheckerboardCoupling.

        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
            res_blocks: number of residual blocks to use.
            mask_config: mask configuration (see build_mask() for more detail).
            bottleneck: True if use bottleneck, False otherwise.
            down_factor: by how much to reduce feature numbers in bottleneck.
            skip: True if use skip architecture, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            coupling_bn: True if apply batch normalization on output of 
                coupling layer, False otherwise.
            affine: True if use affine map, False if use additive map.
        """
        super(CheckerboardCoupling, self).__init__()

        if affine:
            self.coupling = CheckerboardAffineCoupling(
                in_out_dim, mid_dim, res_blocks, mask_config, 
                bottleneck, down_factor, skip, weight_norm, coupling_bn)
        else:
            self.coupling = CheckerboardAdditiveCoupling(
                in_out_dim, mid_dim, res_blocks, mask_config, 
                bottleneck, down_factor, skip, weight_norm, coupling_bn)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """
        return self.coupling(x, reverse)

class ChannelwiseAdditiveCoupling(AbstractCoupling):
    def __init__(self, in_out_dim, mid_dim, res_blocks, mask_config, 
        bottleneck, down_factor, skip, weight_norm, coupling_bn):
        """Initializes a ChannelwiseAdditiveCoupling.

        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
            res_blocks: number of residual blocks to use.
            mask_config: 1 if change the top half, 0 if change the bottom half.
            bottleneck: True if use bottleneck, False otherwise.
            down_factor: by how much to reduce feature numbers in bottleneck.
            skip: True if use skip architecture, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            coupling_bn: True if apply batch normalization on output of 
                coupling layer, False otherwise.
        """
        super(ChannelwiseAdditiveCoupling, self).__init__(
            mask_config, coupling_bn)

        self.in_bn = nn.BatchNorm2d(in_out_dim//2)
        self.block = nn.Sequential(
            nn.ReLU(),
            ResidualModule(in_out_dim, mid_dim, in_out_dim//2, res_blocks, 
                bottleneck, down_factor, skip, weight_norm))
        self.out_bn = nn.BatchNorm2d(in_out_dim//2, affine=False)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """
        [_, C, _, _] = list(x.size())
        if self.mask_config:
            (on, off) = x.split(C//2, dim=1)
        else:
            (off, on) = x.split(C//2, dim=1)
        off_ = self.in_bn(off)
        off_ = torch.cat((off_, -off_), dim=1)    # C channels
        shift = self.block(off_)
        
        log_diag_J = torch.zeros_like(x)    # unit Jacobian determinant
        # See Eq(3) and Eq(4) in NICE and Section 3.7 in real NVP
        if reverse:
            if self.coupling_bn:
                mean, var = self.out_bn.running_mean, self.out_bn.running_var
                mean = mean.reshape((-1, 1, 1, 1)).transpose(0, 1)
                var = var.reshape((-1, 1, 1, 1)).transpose(0, 1)
                on = on * torch.exp(0.5 * torch.log(var + 1e-5)) + mean
            on = on - shift
        else:
            on = on + shift
            if self.coupling_bn:
                if self.training:
                    _, var = self.batch_stat(on)
                else:
                    var = self.out_bn.running_var
                    var = var.reshape((-1, 1, 1, 1)).transpose(0, 1)
                on = self.out_bn(on)
                log_diag_J = log_diag_J - 0.5 * torch.log(var + 1e-5)
        if self.mask_config:
            x = torch.cat((on, off), dim=1)
        else:
            x = torch.cat((off, on), dim=1)
        return x, log_diag_J

class ChannelwiseAffineCoupling(AbstractCoupling):
    def __init__(self, in_out_dim, mid_dim, res_blocks, mask_config, 
        bottleneck, down_factor, skip, weight_norm, coupling_bn):
        """Initializes a ChannelwiseAffineCoupling.

        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
            res_blocks: number of residual blocks to use.
            mask_config: 1 if change the top half, 0 if change the bottom half.
            bottleneck: True if use bottleneck, False otherwise.
            down_factor: by how much to reduce feature numbers in bottleneck.
            skip: True if use skip architecture, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            coupling_bn: True if apply batch normalization on output of 
                coupling layer, False otherwise.
        """
        super(ChannelwiseAffineCoupling, self).__init__(
            mask_config, coupling_bn)

        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.in_bn = nn.BatchNorm2d(in_out_dim//2)
        self.block = nn.Sequential(        # 1st half of resnet: shift
            nn.ReLU(),                    # 2nd half of resnet: log_rescale
            ResidualModule(in_out_dim, mid_dim, in_out_dim, res_blocks, 
                bottleneck, down_factor, skip, weight_norm))
        self.out_bn = nn.BatchNorm2d(in_out_dim//2, affine=False)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """
        [_, C, _, _] = list(x.size())
        if self.mask_config:
            (on, off) = x.split(C//2, dim=1)
        else:
            (off, on) = x.split(C//2, dim=1)
        off_ = self.in_bn(off)
        off_ = torch.cat((off_, -off_), dim=1)     # C channels
        out = self.block(off_)
        (shift, log_rescale) = out.split(C//2, dim=1)
        log_rescale = self.scale * torch.tanh(log_rescale) + self.scale_shift
        
        log_diag_J = log_rescale     # See Eq(6) in real NVP
        # See Eq(7) and Eq(8) and Section 3.7 in real NVP
        if reverse:
            if self.coupling_bn:
                mean, var = self.out_bn.running_mean, self.out_bn.running_var
                mean = mean.reshape((-1, 1, 1, 1)).transpose(0, 1)
                var = var.reshape((-1, 1, 1, 1)).transpose(0, 1)
                on = on * torch.exp(0.5 * torch.log(var + 1e-5)) + mean
            on = (on - shift) * torch.exp(-log_rescale)
        else:
            on = on * torch.exp(log_rescale) + shift
            if self.coupling_bn:
                if self.training:
                    _, var = self.batch_stat(on)
                else:
                    var = self.out_bn.running_var
                    var = var.reshape((-1, 1, 1, 1)).transpose(0, 1)
                on = self.out_bn(on)
                log_diag_J = log_diag_J - 0.5 * torch.log(var + 1e-5)
        if self.mask_config:
            x = torch.cat((on, off), dim=1)
            log_diag_J = torch.cat((log_diag_J, torch.zeros_like(log_diag_J)), 
                dim=1)
        else:
            x = torch.cat((off, on), dim=1)
            log_diag_J = torch.cat((torch.zeros_like(log_diag_J), log_diag_J), 
                dim=1)
        return x, log_diag_J

class ChannelwiseCoupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, res_blocks, mask_config, 
        bottleneck, down_factor, skip, weight_norm, coupling_bn, affine):
        """Initializes a ChannelwiseCoupling.

        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
            res_blocks: number of residual blocks to use.
            mask_config: 1 if change the top half, 0 if change the bottom half.
            bottleneck: True if use bottleneck, False otherwise.
            down_factor: by how much to reduce feature numbers in bottleneck.
            skip: True if use skip architecture, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            coupling_bn: True if apply batch normalization on output of 
                coupling layer, False otherwise.
            affine: True if use affine map, False if use additive map.
        """
        super(ChannelwiseCoupling, self).__init__()

        if affine:
            self.coupling = ChannelwiseAffineCoupling(
                in_out_dim, mid_dim, res_blocks, mask_config, 
                bottleneck, down_factor, skip, weight_norm, coupling_bn)
        else:
            self.coupling = ChannelwiseAdditiveCoupling(
                in_out_dim, mid_dim, res_blocks, mask_config, 
                bottleneck, down_factor, skip, weight_norm, coupling_bn)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """
        return self.coupling(x, reverse)

class RealNVP(nn.Module):
    def __init__(self, prior, base_dim=64, res_blocks=8, mask_config=1., 
        bottleneck=False, down_factor=1, skip=False, 
        weight_norm=True, coupling_bn=True, affine=False):
        """Initializes a RealNVP.

        Args:
            prior: prior distribution over latent space Z.
            base_dim: feature numbers in residual blocks of first few layers.
            res_blocks: number of residual blocks to use in coupling layers.
            mask_config: mask configuration.
            bottleneck: True if use bottleneck, False otherwise.
            down_factor: by how much to reduce feature numbers in bottleneck.
            skip: True if use skip architecture, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            coupling_bn: True if apply batch normalization on output of 
                coupling layer, False otherwise.
            affine: True if use affine map, False if use additive map.
        """
        super(RealNVP, self).__init__()
        self.prior = prior
        dim = base_dim

        # multi-scale architecture for CIFAR-10 (down to 16 x 16 x C)
        # SCALE 1: 32 x 32
        self.scale_1_ckbd = nn.ModuleList([    # in/out (C x H x W): 3 x 32 x 32
            CheckerboardCoupling(in_out_dim=3, mid_dim=dim, 
                                 res_blocks=res_blocks, 
                                 mask_config=mask_config, 
                                 bottleneck=bottleneck, 
                                 down_factor=down_factor, 
                                 skip=skip, 
                                 weight_norm=weight_norm, 
                                 coupling_bn=coupling_bn, 
                                 affine=affine),
            CheckerboardCoupling(in_out_dim=3, mid_dim=dim, 
                                 res_blocks=res_blocks, 
                                 mask_config=1.-mask_config, 
                                 bottleneck=bottleneck, 
                                 down_factor=down_factor, 
                                 skip=skip, 
                                 weight_norm=weight_norm, 
                                 coupling_bn=coupling_bn, 
                                 affine=affine),
            CheckerboardCoupling(in_out_dim=3, mid_dim=dim, 
                                 res_blocks=res_blocks, 
                                 mask_config=mask_config, 
                                 bottleneck=bottleneck, 
                                 down_factor=down_factor, 
                                 skip=skip, 
                                 weight_norm=weight_norm, 
                                 coupling_bn=coupling_bn, 
                                 affine=affine)])
        
        self.scale_1_chan = nn.ModuleList([    # in/out (C x H x W): 12 x 16 x 16
            ChannelwiseCoupling(in_out_dim=12, mid_dim=dim, 
                                res_blocks=res_blocks, 
                                mask_config=1.-mask_config, 
                                bottleneck=bottleneck, 
                                down_factor=down_factor, 
                                skip=skip, 
                                weight_norm=weight_norm, 
                                coupling_bn=coupling_bn, 
                                affine=affine),
            ChannelwiseCoupling(in_out_dim=12, mid_dim=dim, 
                                res_blocks=res_blocks, 
                                mask_config=mask_config, 
                                bottleneck=bottleneck, 
                                down_factor=down_factor, 
                                skip=skip, 
                                weight_norm=weight_norm, 
                                coupling_bn=coupling_bn, 
                                affine=affine),
            ChannelwiseCoupling(in_out_dim=12, mid_dim=dim, 
                                res_blocks=res_blocks, 
                                mask_config=1.-mask_config, 
                                bottleneck=bottleneck, 
                                down_factor=down_factor, 
                                skip=skip, 
                                weight_norm=weight_norm, 
                                coupling_bn=coupling_bn, 
                                affine=affine)])

        # SCALE 2: 16 x 16
        self.scale_2_ckbd = nn.ModuleList([    # in/out (C x H x W): 6 x 16 x 16
            CheckerboardCoupling(in_out_dim=6, mid_dim=dim, 
                                 res_blocks=res_blocks, 
                                 mask_config=mask_config, 
                                 bottleneck=bottleneck, 
                                 down_factor=down_factor, 
                                 skip=skip, 
                                 weight_norm=weight_norm,
                                 coupling_bn=coupling_bn, 
                                 affine=affine),
            CheckerboardCoupling(in_out_dim=6, mid_dim=dim, 
                                 res_blocks=res_blocks, 
                                 mask_config=1.-mask_config, 
                                 bottleneck=bottleneck, 
                                 down_factor=down_factor, 
                                 skip=skip, 
                                 weight_norm=weight_norm, 
                                 coupling_bn=coupling_bn, 
                                 affine=affine),
            CheckerboardCoupling(in_out_dim=6, mid_dim=dim, 
                                 res_blocks=res_blocks, 
                                 mask_config=mask_config, 
                                 bottleneck=bottleneck, 
                                 down_factor=down_factor, 
                                 skip=skip, 
                                 weight_norm=weight_norm, 
                                 coupling_bn=coupling_bn, 
                                 affine=affine),
            CheckerboardCoupling(in_out_dim=6, mid_dim=dim, 
                                 res_blocks=res_blocks, 
                                 mask_config=1.-mask_config, 
                                 bottleneck=bottleneck, 
                                 down_factor=down_factor, 
                                 skip=skip, 
                                 weight_norm=weight_norm, 
                                 coupling_bn=coupling_bn, 
                                 affine=affine)])

    def squeeze(self, x):
        """Squeezes a C x H x W tensor into a 4C x H/2 x W/2 tensor.

        (See Fig 3 in the real NVP paper.)

        Args:
            x: input tensor (B x C x H x W).
        Returns:
            the squeezed tensor (B x 4C x H/2 x W/2).
        """
        [B, C, H, W] = list(x.size())
        x = x.reshape((B, C, H//2, 2, W//2, 2))
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape((B, C*4, H//2, W//2))
        return x

    def undo_squeeze(self, x):
        """unsqueezes a C x H x W tensor into a C/4 x 2H x 2W tensor.

        (See Fig 3 in the real NVP paper.)

        Args:
            x: input tensor (B x C x H x W).
        Returns:
            the squeezed tensor (B x C/4 x 2H x 2W).
        """
        [B, C, H, W] = list(x.size())
        x = x.reshape((B, C//4, 2, 2, H, W))
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape((B, C//4, H*2, W*2))
        return x

    def _downscale(self, x):
        """Mixes up the variables and downscales the tensor.

        Args:
            x: input tensor (B x C x H x W).
        Returns:
            downscaled tensor (B x 4C x H/2 x W/2).
        """
        [_, C, _, _] = list(x.size())
        weights = np.zeros((C*4, C, 2, 2))
        ordering = np.array([[[[1., 0.],
                               [0., 0.]]],
                             [[[0., 0.],
                               [0., 1.]]],
                             [[[0., 1.],
                               [0., 0.]]],
                             [[[0., 0.],
                               [1., 0.]]]])
        for i in range(C):
            s1 = slice(i, i+1)
            s2 = slice(4*i, 4*(i+1))
            weights[s2, s1, :, :] = ordering
        shuffle = np.array([4*i for i in range(C)]
                         + [4*i+1 for i in range(C)]
                         + [4*i+2 for i in range(C)]
                         + [4*i+3 for i in range(C)])
        weights = weights[shuffle, :, :, :].astype('float32')
        weights = torch.tensor(weights).cuda()
        return F.conv2d(x, weights, stride=2, padding=0)

    def _upscale(self, x):
        """Restores the order of variables and upscales the tensor.

        Args:
            x: input tensor (B x C x H x W).
        Returns:
            upscaled tensor (B x C/4 x 2H x 2W).
        """
        [_, C, _, _] = list(x.size())
        weights = np.zeros((C, C//4, 2, 2))
        ordering = np.array([[[[1., 0.],
                               [0., 0.]]],
                             [[[0., 0.],
                               [0., 1.]]],
                             [[[0., 1.],
                               [0., 0.]]],
                             [[[0., 0.],
                               [1., 0.]]]])
        for i in range(C//4):
            s1 = slice(i, i+1)
            s2 = slice(4*i, 4*(i+1))
            weights[s2, s1, :, :] = ordering
        shuffle = np.array([4*i for i in range(C//4)]
                         + [4*i+1 for i in range(C//4)]
                         + [4*i+2 for i in range(C//4)]
                         + [4*i+3 for i in range(C//4)])
        weights = weights[shuffle, :, :, :].astype('float32')
        weights = torch.tensor(weights).cuda()
        return F.conv_transpose2d(x, weights, stride=2, padding=0)

    def factor_out(self, x):
        """Downscales and factors out the bottom half of the tensor.

        (See Fig 4(b) in the real NVP paper.)

        Args:
            x: input tensor (B x C x H x W).
        Returns:
            the top half for further transformation (B x 2C x H/2 x W/2)
            and the Gaussianized bottom half (B x 2C x H/2 x W/2).
        """
        x = self._downscale(x)
        [_, C, _, _] = list(x.size())
        (on, off) = x.split(C//2, dim=1)
        return on, off

    def restore(self, on, off):
        """Merges variables and restores their ordering.

        (See Fig 4(b) in the real NVP paper.)

        Args:
            on: the active (transformed) variables (B x C x H x W).
            off: the inactive variables (B x C x H x W).
        Returns:
            combined variables (B x 2C x H x W).
        """
        x = torch.cat((on, off), dim=1)
        return self._upscale(x)

    def g(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        # downscale and factor out the bottom half of the variables
        x_on, x_off = self.factor_out(z)

        # SCALE 2: 16 x 16
        for i in reversed(range(len(self.scale_2_ckbd))):
            x_on, _ = self.scale_2_ckbd[i](x_on, reverse=True)

        # restore the ordering and upscale
        x = self.restore(x_on, x_off)
        
        # SCALE 1: 32 x 32
        x = self.squeeze(x)
        for i in reversed(range(len(self.scale_1_chan))):
            x, _ = self.scale_1_chan[i](x, reverse=True)
        x = self.undo_squeeze(x)

        for i in reversed(range(len(self.scale_1_ckbd))):
            x, _ = self.scale_1_ckbd[i](x, reverse=True)
        return x

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z.
        """
        z, log_diag_J = x, torch.zeros_like(x)
        
        # SCALE 1: 32 x 32
        for i in range(len(self.scale_1_ckbd)):
            z, inc_log_diag_J = self.scale_1_ckbd[i](z)
            log_diag_J = log_diag_J + inc_log_diag_J

        z, log_diag_J = self.squeeze(z), self.squeeze(log_diag_J)
        for i in range(len(self.scale_1_chan)):
            z, inc_log_diag_J = self.scale_1_chan[i](z)
            log_diag_J = log_diag_J + inc_log_diag_J
        z, log_diag_J = self.undo_squeeze(z), self.undo_squeeze(log_diag_J)

        # downscale and factor out the bottom half of the variables
        z_on, z_off = self.factor_out(z)
        log_diag_J_on, log_diag_J_off = self.factor_out(log_diag_J)

        # SCALE 2: 16 x 16
        for i in range(len(self.scale_2_ckbd)):
            z_on, inc_log_diag_J_on = self.scale_2_ckbd[i](z_on)
            log_diag_J_on = log_diag_J_on + inc_log_diag_J_on

        # restore the ordering and upscale
        z = self.restore(z_on, z_off)
        log_diag_J = self.restore(log_diag_J_on, log_diag_J_off)
        return z, log_diag_J

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Eq(2) and Eq(3) in the real NVP paper.)

        Args:
            x: input minibatch.
        Returns:
            latent code and log-likelihood of input.
        """
        z, log_diag_J = self.f(x)
        [B, C, H, W] = list(z.size())
        log_det_J = torch.sum(log_diag_J, dim=(1, 2, 3))
        log_prior_prob = torch.sum(self.prior.log_prob(z), dim=(1, 2, 3))
        return  log_prior_prob + log_det_J

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        z = self.prior.sample((size, 3*32*32))
        z = z.reshape((size, 3, 32, 32))
        return self.g(z)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)

def logit_transform(x, constraint=0.9, reverse=False):
    '''Transform data from [0, 1] into unbounded space.

    First restrict data into [0.05, 0.95]. Then do logit(alpha+(1-alpha)*x).

    Args:
        x: input tensor.
        constraint: data constraint before logit.
        reverse: True if transform data back to [0, 1].
    Returns:
        transformed tensor and log-determinant of Jacobian from the transform.
        (if reverse=True, no log-determinant is returned.)
    '''
    if reverse:
        x = 1. / (torch.exp(-x) + 1)    # [0.05, 0.95]
        x *= 2.             # [0.1, 1.9]
        x -= 1.             # [-0.9, 0.9]
        x /= constraint     # [-1, 1]
        x += 1.             # [0, 2]
        x /= 2.             # [0, 1]
        return x, 0
    else:
        [B, C, H, W] = list(x.size())
        
        # dequantization
        noise = distributions.Uniform(0, 1).sample((B*C*H*W, ))
        x = (x * 255. + noise.reshape((B, C, H, W))) / 256.
        
        # restrict the data
        x *= 2.             # [0, 2]
        x -= 1.             # [-1, 1]
        x *= constraint     # [-0.9, 0.9]
        x += 1.             # [0.1, 1.9]
        x /= 2.             # [0.05, 0.95]

        # logit the data
        logit_x = torch.log(x) - torch.log(1. - x)

        # log-determinant of Jacobian from the transform
        pre_logit_scale = torch.tensor(
            np.log(constraint) - np.log(1. - constraint))
        log_diag_J = F.softplus(logit_x) + F.softplus(-logit_x) \
            - F.softplus(-pre_logit_scale)
        return logit_x, torch.sum(log_diag_J, dim=(1, 2, 3))

def main(args):
    device = torch.device("cuda:0")

    # model hyperparameters
    batch_size = args.batch_size
    base_dim = args.base_dim
    res_blocks = args.res_blocks
    mask_config = args.mask_config
    bottleneck = args.bottleneck
    down_factor = args.down_factor
    scale_reg = args.scale_reg
    skip = args.skip
    weight_norm = args.weight_norm
    coupling_bn = args.coupling_bn
    affine = args.affine

    filename = 'bs%d_' % batch_size \
             + 'bd%d_' % base_dim \
             + 'rb%d_' % res_blocks \
             + 'bn%d_' % bottleneck \
             + 'sr%d_' % scale_reg \
             + 'sk%d_' % skip \
             + 'wn%d_' % weight_norm \
             + 'cb%d_' % coupling_bn \
             + 'af%d_' % affine

    # optimization hyperparameters
    lr = args.lr
    momentum = args.momentum
    decay = args.decay

    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5),
         transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root='~/torch/data',
        train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=batch_size, shuffle=True, num_workers=2)

    image_size = 32
    full_dim = 3 * image_size**2
    prior = distributions.Normal( # isotropic gaussian
        torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
    flow = RealNVP(prior=prior, 
                   base_dim=base_dim, 
                   res_blocks=res_blocks, 
                   mask_config=mask_config, 
                   bottleneck=bottleneck, 
                   down_factor=down_factor, 
                   skip=skip, 
                   weight_norm=weight_norm, 
                   coupling_bn=coupling_bn, 
                   affine=affine).to(device)
    optimizer = optim.Adam(flow.parameters(), lr=lr, betas=(momentum, decay))

    total_iter = 0
    train = True
    running_loss = 0

    while train:
        for _, data in enumerate(trainloader, 1):
            flow.train()    # set to training mode
            if total_iter == args.max_iter:
                train = False
                break

            total_iter += 1
            optimizer.zero_grad()    # clear gradient tensors

            inputs, _ = data
            # log-determinant of Jacobian from the logit transform
            inputs, log_det_J = logit_transform(inputs)
            inputs = inputs.to(device)
            log_det_J = log_det_J.to(device)

            # log-likelihood of input minibatch
            # (log-determinant of Jacobian from logit transform NOT included)
            log_ll = flow(inputs)
            loss = -(log_ll + log_det_J).mean()

            # L2 regularization on the weight scale parameters
            if weight_norm:
                weight_scale = 0
                for name, param in flow.named_parameters():
                    tokens = name.split('.')
                    param_name = tokens[-1]
                    if param_name in ['weight_g', 'scale'] and param.requires_grad:
                        weight_scale = weight_scale + torch.sum(param.data**2)
                loss = loss + scale_reg * weight_scale

            running_loss += float(loss)

            # backprop and update parameters
            loss.backward()
            optimizer.step()

            if total_iter % 2000 == 0:
                mean_loss = running_loss / 2000
                bit_per_dim = (mean_loss + np.log(256.) * full_dim) \
                            / (full_dim * np.log(2.))
                print('iter %s:' % total_iter, 
                    'loss = %.3f' % mean_loss, 
                    'bits/dim = %.3f' % bit_per_dim)
                running_loss = 0.0

                flow.eval()        # set to inference mode
                with torch.no_grad():
                    z, _ = flow.f(inputs)
                    reconst = flow.g(z)
                    reconst, _ = logit_transform(reconst, reverse=True)
                    samples = flow.sample(args.sample_size)
                    samples, _ = logit_transform(samples, reverse=True)
                    utils.save_image(utils.make_grid(reconst),
                        './reconstruction/' + filename +'iter%d.png' % total_iter)
                    utils.save_image(utils.make_grid(samples),
                        './samples/' + filename +'iter%d.png' % total_iter)

    print('Finished training!')

    torch.save({
        'total_iter': total_iter,
        'model_state_dict': flow.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'batch_size': batch_size,
        'base_dim': base_dim,
        'res_blocks': res_blocks,
        'mask_config': mask_config,
        'bottleneck': bottleneck,
        'down_factor': down_factor,
        'scale_reg': scale_reg,
        'skip': skip,
        'weight_norm': weight_norm,
        'coupling_bn': coupling_bn,
        'affine': affine}, 
        './models/realNVP/cifar10/' + filename +'iter%d.tar' % total_iter)

    print('Checkpoint Saved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CIFAR-10 realNVP PyTorch implementation')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch',
                        type=int,
                        default=64)
    parser.add_argument('--base_dim',
                        help='features in residual blocks of first few layers.',
                        type=int,
                        default=64)
    parser.add_argument('--res_blocks',
                        help='number of residual blocks per group',
                        type=int,
                        default=8)
    parser.add_argument('--mask_config',
                        help='mask configuration',
                        type=float,
                        default=1.)
    parser.add_argument('--bottleneck',
                        help='whether to use bottleneck in residual blocks',
                        type=int,
                        default=0)
    parser.add_argument('--down_factor',
                        help='by how much to reduce features in bottleneck',
                        type=int,
                        default=1)
    parser.add_argument('--scale_reg',
                        help='L2 regularization on weight scale parameters',
                        type=float,
                        default=5*1e-5)
    parser.add_argument('--skip',
                        help='whether to use skip connection in coupling layers',
                        type=int,
                        default=1)
    parser.add_argument('--weight_norm',
                        help='whether to apply weight normalization',
                        type=int,
                        default=1)
    parser.add_argument('--coupling_bn',
                        help='whether to apply batchnorm after coupling layers',
                        type=int,
                        default=1)
    parser.add_argument('--affine',
                        help='whether to use affine coupling',
                        type=int,
                        default=1)
    parser.add_argument('--max_iter',
                        help='maximum number of iterations',
                        type=int,
                        default=250000)
    parser.add_argument('--sample_size',
                        help='number of images to generate',
                        type=int,
                        default=64)
    parser.add_argument('--lr',
                        help='initial learning rate',
                        type=float,
                        default=1e-3)
    parser.add_argument('--momentum',
                        help='beta1 in Adam optimizer',
                        type=float,
                        default=0.9)
    parser.add_argument('--decay',
                        help='beta2 in Adam optimizer',
                        type=float,
                        default=0.999)
    args = parser.parse_args()
    main(args)