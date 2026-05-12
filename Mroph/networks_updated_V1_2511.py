import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import functools
from torch.nn import functional as F
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from monai.networks.nets import SwinUNETR

import itertools
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type, Union

import einops
import einops.layers.torch as elt

from timm.layers import DropPath

from .swin_shared import FeedForward, PatchMode, RelativePositionalEmeddingMode


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=60, min_lr=0)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)

    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[], ifattention=False, attention_form='AG'):
    # net = None
    # norm_layer = get_norm_layer(norm_type=norm)

    if norm == "batch":
        layer_order = 'bcr'
    elif norm == "group":
        layer_order = 'gcr'
    else:
        raise NotImplementedError('norm method [%s] is not recognized' % norm)

    if netG == 'UNet3D':
        net = UNet3D(input_nc, output_nc, f_maps=[16, 32, 64, 128], layer_order=layer_order,
                     num_groups=8,
                     num_levels=4, conv_padding=1, attention=False)

    elif netG == "Swin_MTL":
        net = ResidualUNet3D(input_nc, output_nc, f_maps=[16, 32, 64, 128],
                             layer_order=layer_order, num_groups=8,
                             num_levels=4, conv_padding=1, attention=ifattention,attention_form=attention_form)
    elif netG == "Attention_ResidualUnet3D":
        net = ResidualUNet3D(input_nc, output_nc, f_maps=[16, 32, 64, 128],
                             layer_order=layer_order, num_groups=8,
                             num_levels=4, conv_padding=1, attention=True,
                             attention_form=attention_form)
    elif netG == "Swin_Unetr":
        net = SwinUNETR(img_size=(64, 320, 448),
                        in_channels=input_nc,
                        out_channels=output_nc,
                        depths=(2, 2, 2, 2),
                        num_heads=(3, 6, 12, 24),
                        feature_size=12,
                        drop_rate=0.0,
                        attn_drop_rate=0.0,
                        dropout_path_rate=0.0,
                        use_checkpoint=False)

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)
    return net


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    if lambda_gp > 0.0:
        if type == 'real':
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


def conv3d(in_channels, out_channels, kernel_size, bias, padding):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding):
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules


class SingleConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super(DoubleConv, self).__init__()
        if encoder:

            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:

            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups,
                                   padding=padding))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups,
                                   padding=padding))


class ExtResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, **kwargs):
        super(ExtResNetBlock, self).__init__()
        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order,
                                num_groups=num_groups)

        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        residual = out

        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=2, pool_type='max', basic_module=DoubleConv, conv_layer_order='gcr',
                 num_groups=8, padding=1):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, scale_factor=(2, 2, 2), basic_module=DoubleConv,
                 conv_layer_order='gcr', num_groups=8, mode='nearest', padding=1, upsample=True, attention=False,
                 attention_form='AG'):
        super(Decoder, self).__init__()
        self.if_attention = attention
        self.attention_form = attention_form

        self.upsampling1 = nn.Upsample(scale_factor=2, mode='trilinear',align_corners=False)
        self.Conv_for_upsampling1 = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=conv_kernel_size,stride=1, padding=1),
                                         nn.BatchNorm3d(out_channels),
                                         nn.ReLU(inplace=True))

        if attention == True:
            self.upsampling2 = TransposeConvUpsampling(in_channels=in_channels, out_channels=out_channels,
                                                       kernel_size=conv_kernel_size, scale_factor=scale_factor)
            if attention_form == 'AG':
                        self.attention_gate = AG_Attention_block(F_g=out_channels, F_l=out_channels,
                                                                 F_int=int(out_channels / 2), num_groups=num_groups)
            elif attention_form == 'CBAM':
                        self.CBAM_attention_1 = CBAM_Attention_block(out_channels, reduction=16, spatial_kernel=7)
                        self.CBAM_attention_2 = CBAM_Attention_block(out_channels, reduction=16, spatial_kernel=7)
            self.joining = functools.partial(self._joining, concat=True)
        else:
            self.joining = functools.partial(self._joining, concat=False)
            in_channels = out_channels

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding)

    def forward(self, encoder_features, x):
        if self.if_attention == True:
            x1 = self.upsampling1(x)
            if self.attention_form == 'AG':
                a = self.attention_gate(encoder_features, x1)
            elif self.attention_form == 'CBAM':
                a_1 = self.CBAM_attention_1(encoder_features)
                a_2 = self.CBAM_attention_2(x1)
                a = a_1 + a_2
            x2 = self.upsampling2(encoder_features, x)
            x = self.joining(a, x2)
            x = self.basic_module(x)
        else:
            x = self.upsampling1(x)
            x = self.Conv_for_upsampling1(x.contiguous())
            x = self.joining(encoder_features, x)
            x = self.basic_module(x.contiguous())
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x

class AG_Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int, num_groups=8):
        super(AG_Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=num_groups, num_channels=F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=num_groups, num_channels=F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(num_groups=1, num_channels=1),
            nn.Sigmoid()
        )
        # self.W_g = nn.Sequential(
        #     nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm3d(F_int), nn.GroupNorm(num_groups=num_groups, num_channels=num_groups)
        # )
        #
        # self.W_x = nn.Sequential(
        #     nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm3d(F_int), nn.GroupNorm(num_groups=num_groups, num_channels=num_groups)
        # )
        #
        # self.psi = nn.Sequential(
        #     nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm3d(1), nn.GroupNorm(num_groups=num_groups, num_channels=num_groups),
        #     nn.Sigmoid()
        # )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class CBAM_Attention_block(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM_Attention_block, self).__init__()

        # channel attention
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            # nn.Conv3d(channel, channel // reduction, 1, bias=False),
            # inplace=True
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel,bias=False),
            # nn.Conv3d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        assert spatial_kernel in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if spatial_kernel == 7 else 1
        self.conv1 = nn.Conv3d(2, 1, kernel_size=spatial_kernel, padding=padding, bias=False)

        # self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
        #                       padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv1(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


def create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                    pool_kernel_size):
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            encoder = Encoder(in_channels, out_feature_num,
                              apply_pooling=False,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              padding=conv_padding)
        else:
            # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to make the data isotropic after 1-2 pooling operations
            encoder = Encoder(f_maps[i - 1], out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              pool_kernel_size=pool_kernel_size,
                              padding=conv_padding)

        encoders.append(encoder)

    return nn.ModuleList(encoders)


def create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, upsample,
                    attention=False, attention_form='AG'):
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        if basic_module == DoubleConv:
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
        else:
            in_feature_num = reversed_f_maps[i]

        out_feature_num = reversed_f_maps[i + 1]

        # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
        # currently strides with a constant stride: (2, 2, 2)

        _upsample = True
        if i == 0:
            _upsample = upsample

        decoder = Decoder(in_feature_num, out_feature_num,
                          basic_module=basic_module,
                          conv_layer_order=layer_order,
                          conv_kernel_size=conv_kernel_size,
                          num_groups=num_groups,
                          padding=conv_padding,
                          upsample=_upsample, attention=attention, attention_form=attention_form)
        decoders.append(decoder)
    return nn.ModuleList(decoders)


class AbstractUpsampling(nn.Module):
    def __init__(self, upsample):
        super(AbstractUpsampling, self).__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):
    def __init__(self, mode='nearest'):
        upsample = functools.partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class TransposeConvUpsampling(AbstractUpsampling):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=3, scale_factor=(2, 2, 2)):
        upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor,
                                      padding=1)
        super().__init__(upsample)


class NoUpsampling(AbstractUpsampling):
    def __init__(self):
        super().__init__(self._no_upsampling)

    @staticmethod
    def _no_upsampling(x, size):
        return x


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


class Abstract3DUNet(nn.Module):
    def __init__(self, in_channels, out_channels, basic_module, f_maps=16, layer_order='gcr',
                 num_groups=8, num_levels=4, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, attention=False, attention_form='AG', **kwargs):
        super(Abstract3DUNet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        config = SwinTransformerConfig3D(
            input_size=(64, 192, 288),
            in_channels=in_channels,
            embed_dim=16,
            num_blocks=[3, 3, 6, 6],
            patch_window_size=[(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
            block_window_size=[(2, 7, 7), (2, 7, 7), (2, 7, 7), (2, 7, 7)],
            num_heads=[2, 2, 4, 4]
        )

        self.encoders_0 = SwinTransformer3D(config)

        self.encoders_1 = create_encoders(2, f_maps[0:2], basic_module, conv_kernel_size, conv_padding,
                                          layer_order,
                                          num_groups, pool_kernel_size)

        self.encoders_2 = create_encoders(1, f_maps[0:2], basic_module, conv_kernel_size, conv_padding,
                                          layer_order,
                                          num_groups, pool_kernel_size)

        self.encoders_3=create_encoders(f_maps[2], [64,64,128], basic_module, conv_kernel_size, conv_padding,
                                          layer_order,
                                          num_groups, pool_kernel_size)



        self.decoders_0 = create_decoders([32,64,256], basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                          upsample=True, attention=attention, attention_form=attention_form)

        self.decoders_1 = create_decoders(f_maps[0:2], basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                          upsample=True, attention=attention, attention_form=attention_form)

        self.decoders_2 = create_decoders(f_maps[0:2], basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                          upsample=True, attention=attention, attention_form=attention_form)

        self.decoders_3 = create_decoders(f_maps[0:2], basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                          upsample=True, attention=attention, attention_form=attention_form)

        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        self.final_activation = nn.Sigmoid()

    def forward(self, x, y, z):

        encoders_0_features = []
        # encoders_1_features = []
        # encoders_2_features = []
        encoders_3_features = []
        # decoders_1_features = []

        # encoders_features = self.encoders(x)
        # encoders_features = list(reversed(encoders_features))

        encoders_0_features_origin = self.encoders_0(x)
        encoders_0_features_reverse = list(reversed(encoders_0_features_origin))
        decay = pow(2, len(encoders_0_features_reverse) - 1)
        D, H, W = int(x.shape[2] / decay), int(x.shape[3] / decay), int(x.shape[4] / decay)
        for i in range(len(encoders_0_features_reverse)):
            encoders_0_features.append(
                einops.rearrange(encoders_0_features_reverse[i], 'b (d h w) c -> b c d h w', d=D, h=H, w=W))
            D, H, W = int(D * 2), int(H * 2), int(W * 2)
        x_0 = encoders_0_features[0]

        for encoder_1 in self.encoders_1:
            y = encoder_1(y)
            # reverse the encoder outputs to be aligned with the decoder
            # encoders_1_features.insert(0, y)
        for encoder_2 in self.encoders_2:
            z = encoder_2(z)
            # reverse the encoder outputs to be aligned with the decoder
            # encoders_2_features.insert(0, z)
        y_z=torch.cat((y,z),1)

        for encoder_3 in self.encoders_3:
            y_z = encoder_3(y_z)
            # reverse the encoder outputs to be aligned with the decoder
            # encoders_3_features.insert(0, y_z)

        encoders_0_features_main = encoders_0_features[1:3]
        # encoders_1_features_main = encoders_1_features[1:3]
        # encoders_2_features_main = encoders_2_features[1:3]

        x_decoder_main=torch.cat((x_0,y_z),1)
        # for decoder_0, encoder_features, encoder_1_features, encoder_2_features in zip(self.decoders_0, encoders_0_features_main,
        #                                                            encoders_1_features_main, encoders_2_features_main):
        #     encoder_features_all = encoder_features + encoder_1_features + encoder_2_features
        #     x_decoder_main = decoder_0(encoder_features_all, x_decoder_main)
        for decoder_0, encoder_features in zip(self.decoders_0, encoders_0_features_main):
            x_decoder_main = decoder_0(encoder_features, x_decoder_main)


        encoders_0_features_sub = encoders_0_features[3]
        # encoders_1_features_sub = encoders_1_features[3]
        # encoders_2_features_sub = encoders_2_features[3]

        # x_decoder_sub_1 = x_decoder_main
        # for decoder_1, encoder_features, encoder_1_features, encoder_2_features in zip(self.decoders_1, encoders_features_sub,
        #                                                            encoders_1_features_sub, encoders_2_features_sub):
        #
        #     encoder_features_all = encoder_features + encoder_1_features + encoder_2_features
        #     x_decoder_sub_1 = decoder_1(encoder_features_all, x_decoder_sub_1)
        # x_decoder_sub_1 = self.final_conv(x_decoder_sub_1)
        #
        # x_decoder_sub_2 = x_decoder_main
        # for decoder_2, encoder_features, encoder_1_features, encoder_2_features in zip(self.decoders_2, encoders_features_sub,
        #                                                                                  encoders_1_features_sub,encoders_2_features_sub):
        #     encoder_features_all = encoder_features + encoder_1_features + encoder_2_features
        #     x_decoder_sub_2 = decoder_2(encoder_features_all, x_decoder_sub_2)
        # x_decoder_sub_2 = self.final_conv(x_decoder_sub_2)
        #
        # x_decoder_sub_3 = x_decoder_main
        # for decoder_3, encoder_features, encoder_1_features, encoder_2_features in zip(self.decoders_3,encoders_features_sub,
        #                                                                                encoders_1_features_sub,encoders_2_features_sub):
        #     encoder_features_all = encoder_features + encoder_1_features + encoder_2_features
        #     x_decoder_sub_3 = decoder_3(encoder_features_all, x_decoder_sub_3)
        # x_decoder_sub_3 = self.final_conv(x_decoder_sub_3)

        x_decoder_sub_1 = x_decoder_main
        for decoder_1, encoder_0_features_sub in zip(self.decoders_1,encoders_0_features_sub):
            x_decoder_sub_1 = decoder_1(encoder_0_features_sub, x_decoder_sub_1)
        x_decoder_sub_1 = self.final_conv(x_decoder_sub_1)

        x_decoder_sub_2 = x_decoder_main
        for decoder_2, encoder_0_features_sub in zip(self.decoders_2,encoders_0_features_sub):
            x_decoder_sub_2 = decoder_2(encoder_0_features_sub, x_decoder_sub_2)
        x_decoder_sub_2 = self.final_conv(x_decoder_sub_2)

        x_decoder_sub_3 = x_decoder_main
        for decoder_3, encoder_0_features_sub in zip(self.decoders_3,encoders_0_features_sub):
            x_decoder_sub_3 = decoder_3(encoder_0_features_sub, x_decoder_sub_3)
        x_decoder_sub_3 = self.final_conv(x_decoder_sub_3)

        if self.final_activation is not None:
            x_decoder_sub_1 = self.final_activation(x_decoder_sub_1)
            x_decoder_sub_2 = self.final_activation(x_decoder_sub_2)
            x_decoder_sub_3 = self.final_activation(x_decoder_sub_3)
        return x_decoder_sub_1, x_decoder_sub_2, x_decoder_sub_3


class UNet3D(Abstract3DUNet):
    def __init__(self, in_channels, out_channels, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, conv_padding=1, attention=False, attention_form='AG',
                 **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     conv_padding=conv_padding, attention=attention, attention_form=attention_form,
                                     **kwargs)


class ResidualUNet3D(Abstract3DUNet):
    def __init__(self, in_channels, out_channels, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, conv_padding=1, attention=False, attention_form='AG',
                 **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             basic_module=ExtResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             conv_padding=conv_padding, attention=attention,
                                             attention_form=attention_form, **kwargs)


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm3d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d

        kw = 4
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class SR_Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=4, norm_layer=nn.BatchNorm3d):

        super(SR_Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm3d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d

        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.LeakyReLU(0.2, True),
                    ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                nn.LeakyReLU(0.2, True),
                norm_layer(ndf * nf_mult)
            ]
            sequence += [nn.Conv3d(ndf * nf_mult, ndf * nf_mult, kernel_size=3, stride=1, padding=1, bias=use_bias),
                         nn.LeakyReLU(0.2, True),
                         norm_layer(ndf * nf_mult)
                         ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 16)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            norm_layer(ndf * nf_mult)
        ]
        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=3, stride=1, padding=1),
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):

        return self.model(input)


class PatchEmbedding3D(nn.Module):
    def __init__(
            self,
            input_size: Tuple[int, int, int],
            patch_size: Tuple[int, int, int],
            in_channels: int,
            embed_dim: int,
            norm_layer: Optional[Type[nn.Module]] = None,
            mode: PatchMode = PatchMode.CONVOLUTION,
    ) -> None:
        super().__init__()
        assert (
                input_size[0] % patch_size[0] == 0
                and input_size[1] % patch_size[1] == 0
                and input_size[2] % patch_size[2] == 0
        ), f"Input size {input_size} must be divisible by the patch size {patch_size}"

        self.rearrange_input = (
            elt.Rearrange(
                "b c (ddm dm) (hdm hm) (wdm wm) -> b (ddm hdm wdm) (dm hm wm c)",
                dm=patch_size[0],
                hm=patch_size[1],
                wm=patch_size[2],
            )
            if mode == PatchMode.CONCATENATE
            else nn.Identity()
        )
        self.rearrange_output = (
            elt.Rearrange("b c d h w -> b (d h w) c") if mode == PatchMode.CONVOLUTION else nn.Identity()
        )

        self.proj = (
            nn.Linear(patch_size[0] * patch_size[1] * patch_size[2] * in_channels, embed_dim)
            if mode == PatchMode.CONCATENATE
            else nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

        self.input_size = input_size
        self.output_size = (
            input_size[0] // patch_size[0],
            input_size[1] // patch_size[1],
            input_size[2] // patch_size[2],
        )
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rearrange_input(x)
        x = self.proj(x)
        x = self.rearrange_output(x)
        x = self.norm(x)
        return x


class PatchMerging3D(nn.Module):
    def __init__(
            self,
            input_size: Tuple[int, int, int],
            merge_size: Tuple[int, int, int],
            embed_dim: int,
            norm_layer: Optional[Type[nn.Module]] = nn.LayerNorm,
            mode: PatchMode = PatchMode.CONCATENATE,
    ) -> None:
        super().__init__()
        assert (
                input_size[0] % merge_size[0] == 0
                and input_size[1] % merge_size[1] == 0
                and input_size[2] % merge_size[2] == 0
        ), f"Input size {input_size} must be divisible by the merge size {merge_size}"

        self.rearrange_input = (
            elt.Rearrange("b (d h w) c -> b c d h w", d=input_size[0], h=input_size[1], w=input_size[2])
            if mode == PatchMode.CONVOLUTION
            else elt.Rearrange(
                "b (ddm dm hdm hm wdm wm) c -> b (ddm hdm wdm) (dm hm wm c)",
                ddm=input_size[0] // merge_size[0],
                hdm=input_size[1] // merge_size[1],
                wdm=input_size[2] // merge_size[2],
                dm=merge_size[0],
                hm=merge_size[1],
                wm=merge_size[2],
            )
        )
        self.rearrange_output = (
            elt.Rearrange("b c d h w -> b (d h w) c") if mode == PatchMode.CONVOLUTION else nn.Identity()
        )

        merge_dim = merge_size[0] * merge_size[1] * merge_size[2] * embed_dim

        self.proj = (
            nn.Conv3d(embed_dim, merge_dim // 4, merge_size, merge_size)
            if mode == PatchMode.CONVOLUTION
            else nn.Linear(merge_dim, merge_dim // 4)
        )
        self.norm = norm_layer(merge_dim // 4) if norm_layer is not None else nn.Identity()

        self.input_size = input_size
        self.output_size = (
            input_size[0] // merge_size[0],
            input_size[1] // merge_size[1],
            input_size[2] // merge_size[2],
        )
        self.merge_size = merge_size
        self.in_channels = embed_dim
        self.out_channels = merge_dim // 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rearrange_input(x)
        x = self.proj(x)
        x = self.rearrange_output(x)
        x = self.norm(x)
        return x


class WindowShift3D(nn.Module):

    def __init__(
            self, input_size: Tuple[int, int, int], shift_size: Tuple[int, int, int], reverse: bool = False
    ) -> None:
        super().__init__()
        self.expand = elt.Rearrange("b (d h w) c -> b d h w c", d=input_size[0], h=input_size[1], w=input_size[2])
        self.squeeze = elt.Rearrange("b d h w c -> b (d h w) c")

        self.input_size = input_size
        self.output_size = input_size
        self.reverse = reverse
        self.shift_size = shift_size if reverse else (-shift_size[0], -shift_size[1], -shift_size[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = torch.roll(x, shifts=self.shift_size, dims=(1, 2, 3))
        x = self.squeeze(x)
        return x


class Attention3D(nn.Module):
    def __init__(
            self,
            input_size: Tuple[int, int, int],
            window_size: Tuple[int, int, int],
            embed_dim: int,
            num_heads: int,
            qkv_bias: bool = False,
            qk_scale: Optional[float] = None,
            drop_attn: float = 0.0,
            drop_proj: float = 0.0,
            rpe_mode: RelativePositionalEmeddingMode = RelativePositionalEmeddingMode.BIAS,
            shift: bool = False,
    ) -> None:
        super().__init__()
        assert (
                input_size[0] % window_size[0] == 0
                and input_size[1] % window_size[1] == 0
                and input_size[2] % window_size[2] == 0
        ), f"Input size {input_size} must be divisible by the window size {window_size}"
        assert (
                embed_dim % num_heads == 0
        ), f"Embedding dimension {embed_dim} must be divisible by the number of heads {num_heads}"

        self.to_sequence = elt.Rearrange("b nh n c -> b n (nh c)")
        self.to_windows = elt.Rearrange(
            "b (ddm dm hdm hm wdm wm) c -> (b ddm hdm wdm) (dm hm wm) c",
            ddm=input_size[0] // window_size[0],
            hdm=input_size[1] // window_size[1],
            wdm=input_size[2] // window_size[2],
            dm=window_size[0],
            hm=window_size[1],
            wm=window_size[2],
        )
        self.to_spatial = elt.Rearrange(
            "(b ddm hdm wdm) (dm hm wm) c -> b (ddm dm hdm hm wdm wm) c",
            ddm=input_size[0] // window_size[0],
            hdm=input_size[1] // window_size[1],
            wdm=input_size[2] // window_size[2],
            dm=window_size[0],
            hm=window_size[1],
            wm=window_size[2],
        )

        max_distance = (window_size[0] - 1, window_size[1] - 1, window_size[2] - 1)
        self.bias_mode = rpe_mode == RelativePositionalEmeddingMode.BIAS
        self.context_mode = rpe_mode == RelativePositionalEmeddingMode.CONTEXT

        self.embedding_table = nn.Embedding(sum(2 * d + 1 for d in max_distance), num_heads) if self.bias_mode else None
        self.embedding_table_q = (
            nn.Embedding(sum(2 * d + 1 for d in max_distance), embed_dim) if self.context_mode else None
        )
        self.embedding_table_k = (
            nn.Embedding(sum(2 * d + 1 for d in max_distance), embed_dim) if self.context_mode else None
        )
        self.register_buffer("indices", None)  # to cleanly move to device
        self.indices = self._create_indices(window_size, max_distance) if self.bias_mode or self.context_mode else None

        self.reshape_embedding = (
            elt.Rearrange(
                "n m nh -> nh n m" if self.bias_mode else "n m (nh c) -> n m nh c",
                nh=num_heads,
            )
            if self.bias_mode or self.context_mode
            else nn.Identity()
        )

        self.shift = shift
        self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        self.register_buffer("shift_mask", torch.tensor(0.0))  # to cleanly move to device
        self.shift_mask = (
            self._create_shift_mask(input_size, window_size, self.shift_size) if shift else torch.tensor(0.0)
        )
        self.shift_win = WindowShift3D(input_size, self.shift_size, reverse=False) if shift else nn.Identity()
        self.shift_win_rev = WindowShift3D(input_size, self.shift_size, reverse=True) if shift else nn.Identity()

        self.to_broadcast_mask = elt.Rearrange("bw ... -> () bw () ...") if shift else nn.Identity()
        bw = (input_size[0] // window_size[0]) * (input_size[1] // window_size[1]) * (input_size[2] // window_size[2])
        self.to_broadcast_score = elt.Rearrange("(b bw) ... -> b bw ...", bw=bw)
        self.to_merged_score = elt.Rearrange("b bw ... -> (b bw) ...", bw=bw)

        self.num_heads = num_heads
        self.scale = qk_scale or (embed_dim // num_heads) ** -0.5

        self.proj_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.drop_attn = nn.Dropout(drop_attn)
        self.drop_proj = nn.Dropout(drop_proj)
        self.softmax = nn.Softmax(dim=-1)

        self.to_qkv = elt.Rearrange("b n (qkv nh c) -> qkv b nh n c", qkv=3, nh=num_heads)

        self.input_size = input_size
        self.out_size = input_size
        self.in_channels = embed_dim
        self.out_channels = embed_dim
        self.window_size = window_size

        self.attn_weights: torch.Tensor = torch.tensor(0)

    def _create_shift_mask(
            self, input_size: Tuple[int, int, int], window_size: Tuple[int, int, int], shift_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        id_map = torch.zeros((1, input_size[0], input_size[1], input_size[2], 1))
        shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)

        d_slices = [(0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None)]
        h_slices = [(0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None)]
        w_slices = [(0, -window_size[2]), (-window_size[2], -shift_size[2]), (-shift_size[2], None)]

        cnt = 0
        for d_start, d_stop in d_slices:
            for h_start, h_stop in h_slices:
                for w_start, w_stop in w_slices:
                    id_map[:, d_start:d_stop, h_start:h_stop, w_start:w_stop, :] = cnt
                    cnt += 1

        id_map = id_map.view(1, -1, 1).contiguous()
        id_windows = self.to_windows(id_map).squeeze(-1)
        id_diff_windows = id_windows.unsqueeze(1) - id_windows.unsqueeze(2)

        return id_diff_windows.masked_fill(id_diff_windows != 0, float(-1e9)).masked_fill(
            id_diff_windows == 0, float(0.0)
        )

    def _create_indices(self, window_size: Tuple[int, int, int], max_distance: Tuple[int, int, int]) -> torch.Tensor:
        offsets = [0] + list(itertools.accumulate((2 * d + 1 for d in max_distance[:-1])))

        d_abs_dist = torch.arange(window_size[0])
        h_abs_dist = torch.arange(window_size[1])
        w_abs_dist = torch.arange(window_size[2])
        d_abs_dist = einops.repeat(d_abs_dist, "p -> p w h", h=window_size[1], w=window_size[2]).flatten()
        h_abs_dist = einops.repeat(h_abs_dist, "p -> d p w", d=window_size[0], w=window_size[2]).flatten()
        w_abs_dist = einops.repeat(w_abs_dist, "p -> d h p", d=window_size[0], h=window_size[1]).flatten()

        d_rel_dist = d_abs_dist.unsqueeze(0) - d_abs_dist.unsqueeze(1)
        h_rel_dist = h_abs_dist.unsqueeze(0) - h_abs_dist.unsqueeze(1)
        w_rel_dist = w_abs_dist.unsqueeze(0) - w_abs_dist.unsqueeze(1)

        d_idx = torch.clamp(d_rel_dist, -max_distance[0], max_distance[0]) + max_distance[0] + offsets[0]
        h_idx = torch.clamp(h_rel_dist, -max_distance[1], max_distance[1]) + max_distance[1] + offsets[1]
        w_idx = torch.clamp(w_rel_dist, -max_distance[2], max_distance[2]) + max_distance[2] + offsets[2]

        return torch.stack([d_idx, h_idx, w_idx])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.shift:
            x = self.shift_win(x)
        x = self.to_windows(x)
        qkv = self.to_qkv(self.proj_qkv(x))
        q, k, v = qkv[0], qkv[1], qkv[2]
        s = torch.einsum("b h n c, b h m c -> b h n m", q, k)

        if self.bias_mode:
            assert self.indices is not None, "Indices must be created for bias mode."
            assert self.embedding_table is not None, "Embedding table must be created for bias mode."
            biases = self.embedding_table(self.indices).sum(0)
            biases = self.reshape_embedding(biases)
            s = s + biases
        elif self.context_mode:
            assert self.indices is not None, "Indices must be created for context mode."
            assert self.embedding_table_q is not None, "Embedding table Q must be created for context mode."
            assert self.embedding_table_k is not None, "Embedding table K must be created for context mode."
            q_embedding = self.embedding_table_q(self.indices).sum(0)
            k_embedding = self.embedding_table_k(self.indices).sum(0)
            q_embedding = self.reshape_embedding(q_embedding)
            k_embedding = self.reshape_embedding(k_embedding)
            s = s + torch.einsum("b h n c, n m h c -> b h n m", q, k_embedding)
            s = s + torch.einsum("b h n c, n m h c -> b h n m", k, q_embedding)

        s = s * self.scale

        if self.shift:
            s = self.to_broadcast_score(s) + self.to_broadcast_mask(self.shift_mask)
            s = self.to_merged_score(s)

        if mask is not None:
            s = self.to_broadcast_score(s) + mask
            s = self.to_merged_score(s)

        a = self.softmax(s)
        self.attn_weights = a
        a = self.drop_attn(a)

        x = torch.einsum("b h n m, b h m c -> b h n c", a, v)
        x = self.to_sequence(x)

        x = self.proj(x)
        x = self.drop_proj(x)

        x = self.to_spatial(x)

        if self.shift:
            x = self.shift_win_rev(x)

        return x


class SwinTransformerBlock3D(nn.Module):
    def __init__(
            self,
            input_size: Tuple[int, int, int],
            embed_dim: int,
            num_heads: int,
            window_size: Tuple[int, int, int],
            mlp_ratio: float = 4.0,
            qkv_bias: bool = False,
            qk_scale: Optional[float] = None,
            drop: float = 0.0,
            drop_attn: float = 0.0,
            drop_path: float = 0.0,
            rpe_mode: RelativePositionalEmeddingMode = RelativePositionalEmeddingMode.BIAS,
            shift: bool = False,
            act_layer: nn.Module = nn.GELU(),
            norm_layer: Optional[Type[nn.Module]] = nn.LayerNorm,
    ) -> None:
        super().__init__()

        assert (
                embed_dim % num_heads == 0
        ), f"Embedding dimension {embed_dim} must be divisible by the number of heads {num_heads}"

        # compute padding for input
        padding_size = (
            window_size[0] - input_size[0] % window_size[0] if input_size[0] % window_size[0] != 0 else 0,
            window_size[1] - input_size[1] % window_size[1] if input_size[1] % window_size[1] != 0 else 0,
            window_size[2] - input_size[2] % window_size[2] if input_size[2] % window_size[2] != 0 else 0,
        )
        pad_input_size = (
            input_size[0] + padding_size[0],
            input_size[1] + padding_size[1],
            input_size[2] + padding_size[2],
        )
        self.pad = nn.ConstantPad3d((0, padding_size[2], 0, padding_size[1], 0, padding_size[0]), 0.0)
        self.register_buffer("padding_mask", torch.tensor(0.0))  # to cleanly move to device
        self.padding_mask = self._create_padding_mask(pad_input_size, window_size, padding_size)
        self.rearrange_input = elt.Rearrange(
            "b (d h w) c -> b c d h w", d=input_size[0], h=input_size[1], w=input_size[2]
        )
        self.rearrange_pad_input = elt.Rearrange(
            "b (d h w) c -> b c d h w", d=pad_input_size[0], h=pad_input_size[1], w=pad_input_size[2]
        )
        self.rearrange_output = elt.Rearrange("b c d h w -> b (d h w) c")
        self.use_padding = padding_size != (0, 0, 0)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm_attn = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
        self.attn = Attention3D(
            pad_input_size,
            window_size,
            embed_dim,
            num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_attn=drop_attn,
            drop_proj=drop,
            rpe_mode=rpe_mode,
            shift=shift,
        )
        self.norm_proj = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
        self.proj = FeedForward(embed_dim, int(embed_dim * mlp_ratio), embed_dim, act_layer, norm_layer)

        self.input_size = input_size
        self.output_size = input_size
        self.in_channels = embed_dim
        self.out_channels = embed_dim
        self.padding_size = padding_size

    def _create_padding_mask(
            self, input_size: Tuple[int, int, int], window_size: Tuple[int, int, int], pad_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        id_map = torch.zeros((1, input_size[0], input_size[1], input_size[2], 1))
        cnt = 1
        for d, p in itertools.product(range(input_size[0]), range(1, pad_size[0] + 1)):
            id_map[:, d, -p, :, :] = cnt
            cnt += 1
        for h, p in itertools.product(range(input_size[1]), range(1, pad_size[1] + 1)):
            id_map[:, -p, h, :, :] = cnt
            cnt += 1
        for w, p in itertools.product(range(input_size[2]), range(1, pad_size[2] + 1)):
            id_map[:, -p, :, w, :] = cnt
            cnt += 1
        id_map = id_map.view(1, -1, 1).contiguous()
        id_windows = einops.rearrange(
            (id_map),
            "b (ddm dm hdm hm wdm wm) c -> (b ddm hdm wdm) (dm hm wm) c",
            ddm=input_size[0] // window_size[0],
            hdm=input_size[1] // window_size[1],
            wdm=input_size[2] // window_size[2],
            dm=window_size[0],
            hm=window_size[1],
            wm=window_size[2],
        ).squeeze(-1)
        id_diff_windows = id_windows.unsqueeze(1) - id_windows.unsqueeze(2)
        id_diff_windows = einops.rearrange(
            id_diff_windows,
            "bw ... -> () bw () ...",
        )
        return id_diff_windows.masked_fill(id_diff_windows != 0, float(1)).masked_fill(id_diff_windows == 0, float(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (B, N, C).

        Returns:
            torch.Tensor: Output tensor (B, N, C).
        """

        if self.use_padding:
            x = self.rearrange_input(x)
            x = self.pad(x)
            x = self.rearrange_output(x)

        skip = x
        x = self.drop_path(self.attn(x, self.padding_mask)) + skip
        x = self.norm_attn(x)

        if self.use_padding:
            x = self.rearrange_pad_input(x)
            x = x[..., : self.input_size[0], : self.input_size[1], : self.input_size[2]]
            x = self.rearrange_output(x)

        skip = x
        x = self.drop_path(self.proj(x)) + skip
        x = self.norm_proj(x)

        return x


class SwinTransformerStage3D(nn.Module):
    def __init__(
            self,
            # Stage parameters
            input_size: Tuple[int, int, int],
            in_channels: int,
            embed_dim: int,
            num_blocks: int,
            patch_module: Union[Type[PatchEmbedding3D], Type[PatchMerging3D]],
            # Window parameters
            patch_window_size: Tuple[int, int, int],
            block_window_size: Tuple[int, int, int],
            # Block parameters
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = False,
            qk_scale: Optional[float] = None,
            drop: float = 0.0,
            drop_attn: float = 0.0,
            drop_path: Optional[List[float]] = None,
            act_layer: nn.Module = nn.GELU(),
            # Normalisation parameters
            norm_layer_pre_block: Optional[Type[nn.Module]] = nn.LayerNorm,
            norm_layer_block: Optional[Type[nn.Module]] = nn.LayerNorm,
            # Mode parameters
            patch_mode: PatchMode = PatchMode.CONCATENATE,
            rpe_mode: RelativePositionalEmeddingMode = RelativePositionalEmeddingMode.BIAS,
    ) -> None:
        super().__init__()

        assert (
                drop_path is None or len(drop_path) == num_blocks
        ), "Length of drop_path must be equal to the number of blocks."

        if patch_module == PatchEmbedding3D:
            patch_config = dict(
                input_size=input_size,
                patch_size=patch_window_size,
                in_channels=in_channels,
                embed_dim=embed_dim,
                norm_layer=norm_layer_pre_block,
                mode=patch_mode,
            )
        else:
            patch_config = dict(
                input_size=input_size,
                merge_size=patch_window_size,
                embed_dim=embed_dim,
                norm_layer=norm_layer_pre_block,
                mode=patch_mode,
            )
            in_channels = embed_dim
        self.patch = patch_module(**patch_config)  # type: ignore

        input_size = (
            input_size[0] // patch_window_size[0],
            input_size[1] // patch_window_size[1],
            input_size[2] // patch_window_size[2],
        )
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                SwinTransformerBlock3D(
                    input_size,
                    self.patch.out_channels,
                    num_heads,
                    block_window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    drop_attn=drop_attn,
                    drop_path=0.0 if drop_path is None else drop_path[i],
                    rpe_mode=rpe_mode,
                    act_layer=act_layer,
                    norm_layer=norm_layer_block,
                    shift=i % 2 == 1,
                )
            )

        self.input_size = (
            input_size[0] * patch_window_size[0],
            input_size[1] * patch_window_size[1],
            input_size[2] * patch_window_size[2],
        )
        self.output_size = input_size
        self.in_channels = in_channels
        self.out_channels = self.patch.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (B, C, D, H, W) or (B, N, C). Depending on the patch module.

        Returns:
            torch.Tensor: Output tensor (B, N, C).
        """

        x = self.patch(x)
        for block in self.blocks:
            x = block(x)
        return x


@dataclass
class SwinTransformerConfig3D:
    input_size: Tuple[int, int, int]
    in_channels: int
    embed_dim: int
    num_blocks: List[int]
    # Window parameters
    patch_window_size: List[Tuple[int, int, int]]
    block_window_size: List[Tuple[int, int, int]]
    # Block parameters
    num_heads: List[int]
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    qk_scale: Optional[float] = None
    drop: float = 0.0
    drop_attn: float = 0.0
    drop_path: float = 0.1
    act_layer: nn.Module = nn.GELU()
    # Mode parameters
    # patch_mode: Optional[List[PatchMode] | List[str]] = None
    # rpe_mode: RelativePositionalEmeddingMode | str = RelativePositionalEmeddingMode.BIAS
    patch_mode: Optional[List[Union[PatchMode, str]]] = None
    rpe_mode: Union[RelativePositionalEmeddingMode, str] = RelativePositionalEmeddingMode.BIAS

    def __post_init__(self):
        if not (
                len(self.num_blocks) == len(self.patch_window_size) == len(self.block_window_size) == len(
            self.num_heads)
        ):
            raise ValueError(
                "Lengths of num_blocks, patch_window_size, " "block_window_size, and num_heads must be equal."
            )

        if not len(self.num_blocks) > 0:
            raise ValueError("At least one stage must be defined.")

        if self.patch_mode is None:
            self.patch_mode = [PatchMode.CONVOLUTION] + [PatchMode.CONCATENATE] * (len(self.num_blocks) - 1)
        else:
            self.patch_mode = [PatchMode(str.lower(pm)) if isinstance(pm, str) else pm for pm in self.patch_mode]
            assert len(self.patch_mode) == len(self.num_blocks), "Length of patch_mode must be equal to num_blocks."
            assert all(pm in PatchMode for pm in self.patch_mode), "Patch mode must be one of PatchMode."

        if isinstance(self.rpe_mode, str):
            self.rpe_mode = RelativePositionalEmeddingMode(str.lower(self.rpe_mode))


class SwinTransformer3D(nn.Module):
    def __init__(self, config: SwinTransformerConfig3D) -> None:
        super().__init__()

        self.stages = nn.ModuleList()
        stochastic_depth_decay = [x.item() for x in torch.linspace(0, config.drop_path, sum(config.num_blocks))]
        out_channels = config.embed_dim
        input_size = config.input_size
        for i, (nb, pws, bws, nh) in enumerate(
                zip(config.num_blocks, config.patch_window_size, config.block_window_size, config.num_heads)
        ):
            patch_module = PatchEmbedding3D if i == 0 else PatchMerging3D
            norm_layer_pre_block = None if i == 0 else nn.LayerNorm
            stage = SwinTransformerStage3D(
                input_size,
                config.in_channels,
                out_channels,
                num_blocks=nb,
                patch_module=patch_module,
                patch_window_size=pws,
                block_window_size=bws,
                num_heads=nh,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                qk_scale=config.qk_scale,
                drop=config.drop,
                drop_attn=config.drop_attn,
                drop_path=stochastic_depth_decay[sum(config.num_blocks[:i]): sum(config.num_blocks[: i + 1])],
                act_layer=config.act_layer,
                norm_layer_pre_block=norm_layer_pre_block,
                norm_layer_block=nn.LayerNorm,
                patch_mode=config.patch_mode[i],  # type: ignore (checked in config)
                rpe_mode=config.rpe_mode,  # type: ignore (checked in config)
            )
            input_size = (input_size[0] // pws[0], input_size[1] // pws[1], input_size[2] // pws[2])
            out_channels = stage.out_channels
            self.stages.append(stage)

        self.input_size = [s.input_size for s in self.stages]
        self.output_size = [s.output_size for s in self.stages]
        self.in_channels = [s.in_channels for s in self.stages]
        self.out_channels = [s.out_channels for s in self.stages]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = []
        # print(x.shape)
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        return out
