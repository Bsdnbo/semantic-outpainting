import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlockWithSN
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
from models.networks.architecture import ResnetBlockWithSN_cat as ResnetBlockWithSN_cat
from models.networks.architecture import ResnetBlockWithSN_random as ResnetBlockWithSN_random
from models.networks.architecture import SegoutSPADEResnetBlock as SegoutSPADEResnetBlock

import numpy as np


class SPADEOutpaintingGenerator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        opt.shortcut_nc = opt.label_nc + 1 #noseg -> 1
        self.opt = opt
        nf = opt.ngf

        if self.opt.use_gpu:
            self.encoder_norm = torch.nn.SyncBatchNorm
        else:
            self.encoder_norm = torch.nn.BatchNorm2d
        self.actvn = nn.LeakyReLU(0.2, False)

        if opt.crop_size == 512 and hasattr(opt, 'downsample_first_layer') and opt.downsample_first_layer:
            self.conv_0 = nn.Conv2d(opt.semantic_nc, 1 * nf, kernel_size=7, padding=3, stride=2)
        else:
            self.conv_0 = nn.Conv2d(opt.semantic_nc, 1 * nf, kernel_size=7, padding=3)
        self.conv_1 = spectral_norm(nn.Conv2d(1 * nf, 2 * nf, kernel_size=3, padding=1))
        self.down_0 = spectral_norm(nn.Conv2d(2 * nf, 2 * nf, kernel_size=3, stride=2, padding=1))
        self.conv_2 = spectral_norm(nn.Conv2d(2 * nf, 4 * nf, kernel_size=3, padding=1))
        self.down_1 = spectral_norm(nn.Conv2d(4 * nf, 4 * nf, kernel_size=3, stride=2, padding=1))
        self.conv_3 = spectral_norm(nn.Conv2d(4 * nf, 8 * nf, kernel_size=3, padding=1))
        self.down_2 = spectral_norm(nn.Conv2d(8 * nf, 8 * nf, kernel_size=3, stride=2, padding=1))
        self.conv_4 = spectral_norm(nn.Conv2d(8 * nf, 16 * nf, kernel_size=3, padding=1))
        self.down_3 = spectral_norm(nn.Conv2d(16 * nf, 16 * nf, kernel_size=3, stride=2, padding=1))
        self.conv_5 = spectral_norm(nn.Conv2d(16 * nf, 16 * nf, kernel_size=3, padding=1))
        self.down_4 = spectral_norm(nn.Conv2d(16 * nf, 16 * nf, kernel_size=3, stride=2, padding=1))

        self.bn_0 = self.encoder_norm(1 * nf)
        self.bn_1 = self.encoder_norm(2 * nf)
        self.bn_2 = self.encoder_norm(2 * nf)
        self.bn_3 = self.encoder_norm(4 * nf)
        self.bn_4 = self.encoder_norm(4 * nf)
        self.bn_5 = self.encoder_norm(8 * nf)
        self.bn_6 = self.encoder_norm(8 * nf)
        self.bn_7 = self.encoder_norm(16 * nf)
        self.bn_8 = self.encoder_norm(16 * nf)
        self.bn_9 = self.encoder_norm(16 * nf)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_2 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        num_up_layers = 5

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, image_masked, seg, mask):
        if seg is not None:
            x = torch.cat([image_masked, seg, mask], dim=1)
            shortcut_input = torch.cat([seg, mask], dim=1)
        else:
            x = torch.cat([image_masked, mask], dim=1)
            shortcut_input = None

        out_0 = self.conv_0(x)                              # 64 * 256 * 256
        out_1 = self.conv_1(self.actvn(self.bn_0(out_0)))   # 128 * 256 * 256
        out_2 = self.down_0(self.actvn(self.bn_1(out_1)))   # 128 * 128 * 128
        out_3 = self.conv_2(self.actvn(self.bn_2(out_2)))   # 256 * 128 * 128
        out_4 = self.down_1(self.actvn(self.bn_3(out_3)))   # 256 * 64 * 64
        out_5 = self.conv_3(self.actvn(self.bn_4(out_4)))   # 512 * 64 *64
        out_6 = self.down_2(self.actvn(self.bn_5(out_5)))   # 512 * 32 * 32
        out_7 = self.conv_4(self.actvn(self.bn_6(out_6)))   # 1024 * 32 * 32
        out_8 = self.down_3(self.actvn(self.bn_7(out_7)))   # 1024 * 16 * 16
        out_9 = self.conv_5(self.actvn(self.bn_8(out_8)))   # 1024 * 16 * 16
        out_10 = self.down_4(self.actvn(self.bn_9(out_9)))  # 1024 * 8 * 8

        x = self.G_middle_0(out_10, shortcut_input)                 # 1024 * 8 * 8
        x = self.up(x)                                              # 1024 * 16 * 16
        x = self.G_middle_1(x, shortcut_input)                      # 1024 * 16 * 16
        x = self.G_middle_2(x, shortcut_input)                      # 1024 * 16 * 16

        x = self.up(x)                                              # 1024 * 32 * 32
        x = self.up_0(x, shortcut_input)                            # 512 * 32 * 32
        x = self.up(x)                                              # 512 * 64 * 64
        x = self.up_1(x, shortcut_input)                            # 256 * 64 * 64
        x = self.up(x)                                              # 256 * 128 * 128
        x = self.up_2(x, shortcut_input)                            # 128 * 128 * 128
        x = self.up(x)                                              # 128 * 256 * 256
        x = self.up_3(x, shortcut_input)                            # 64 * 256 * 256

        if self.opt.crop_size == 512 and hasattr(self.opt, 'downsample_first_layer') and self.opt.downsample_first_layer:
            x = self.up(x)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x


class OutpaintingGenerator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        opt.shortcut_nc = opt.label_nc + 1
        self.opt = opt
        nf = opt.ngf

        self.actvn = nn.LeakyReLU(0.2, False)

        if opt.crop_size == 512 and hasattr(opt, 'downsample_first_layer') and opt.downsample_first_layer:
            stride = 2
        else:
            stride = 1
        self.conv_s = nn.Conv2d(opt.semantic_nc, 1 * nf, kernel_size=7, padding=3, stride=stride)

        self.conv_0 = ResnetBlockWithSN(1 * nf, 1 * nf, opt)

        self.down_0 = ResnetBlockWithSN(1 * nf, 2 * nf, opt, stride=2)
        self.conv_1 = ResnetBlockWithSN(2 * nf, 2 * nf, opt)

        self.down_1 = ResnetBlockWithSN(2 * nf, 4 * nf, opt, stride=2)
        self.conv_2 = ResnetBlockWithSN(4 * nf, 4 * nf, opt)

        self.down_2 = ResnetBlockWithSN(4 * nf, 8 * nf, opt, stride=2)
        self.conv_3 = ResnetBlockWithSN(8 * nf, 8 * nf, opt)

        self.down_3 = ResnetBlockWithSN(8 * nf, 16 * nf, opt, stride=2)
        self.conv_4 = ResnetBlockWithSN(16 * nf, 16 * nf, opt)

        self.down_4 = ResnetBlockWithSN(16 * nf, 16 * nf, opt, stride=2)
        self.G_middle_0 = ResnetBlockWithSN(16 * nf, 16 * nf, opt)

        self.G_middle_1 = ResnetBlockWithSN(16 * nf, 16 * nf, opt)

        self.up_0 = ResnetBlockWithSN(16 * nf, 8 * nf, opt)
        self.up_10 = ResnetBlockWithSN(16 * nf, 8 * nf, opt)
        self.up_11 = ResnetBlockWithSN(8 * nf, 4 * nf, opt)
        self.up_20 = ResnetBlockWithSN(8 * nf, 4 * nf, opt)
        self.up_21 = ResnetBlockWithSN(4 * nf, 2 * nf, opt)
        self.up_30 = ResnetBlockWithSN(4 * nf, 2 * nf, opt)
        self.up_31 = ResnetBlockWithSN(2 * nf, 1 * nf, opt)

        if hasattr(opt, 'segmentation') and opt.segmentation:
            # generate seg map
            self.conv_img = nn.Conv2d(2 * nf, opt.label_nc, 3, padding=1)
        else:
            # generate image
            self.conv_img = nn.Conv2d(2 * nf, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def forward(self, image_masked, seg, mask):
        if hasattr(self.opt, 'segmentation') and self.opt.segmentation:
            x = torch.cat([image_masked, seg, mask], dim=1)
        else:
            x = torch.cat([image_masked, mask], dim=1)

        x = self.conv_s(x)                              # 64 * 256 * 256
        x = self.conv_0(x)
        short_0 = x

        x = self.down_0(x)                              # 128 * 128 * 128
        x = self.conv_1(x)
        short_1 = x

        x = self.down_1(x)                              # 256 * 64 * 64
        x = self.conv_2(x)
        short_2 = x

        x = self.down_2(x)                              # 512 * 32 * 32
        x = self.conv_3(x)
        short_3 = x

        x = self.down_3(x)                              # 1024 * 16 * 16
        x = self.conv_4(x)                              # 1024 * 16 * 16

        x = self.down_4(x)                              # 1024 * 8 * 8
        x = self.G_middle_0(x)                          # 1024 * 8 * 8

        x = self.up(x)                                  # 1024 * 16 * 16
        x = self.G_middle_1(x)                          # 1024 * 16 * 16

        x = self.up(x)                                  # 1024 * 32 * 32
        x = self.up_0(x)                                # 512 * 32 * 32

        x = torch.cat([x, short_3], dim=1)               # 1024 * 32 * 32
        x = self.up(x)                                   # 1024 * 64 * 64
        x = self.up_10(x)                                # 512 * 64 * 64
        x = self.up_11(x)                                # 256 * 64 * 64

        x = torch.cat([x, short_2], dim=1)               # 512 * 64 * 64
        x = self.up(x)                                   # 512 * 128 * 128
        x = self.up_20(x)                                # 256 * 128 * 128
        x = self.up_21(x)                                # 128 * 128 * 128

        x = torch.cat([x, short_1], dim=1)              # 256 * 128 * 128
        x = self.up(x)                                  # 256 * 256 * 256
        x = self.up_30(x)                               # 128 * 256 * 256
        x = self.up_31(x)                               # 64 * 256 * 256

        x = torch.cat([x, short_0], dim=1)              # 128 * 256 * 256

        if self.opt.crop_size == 512 and hasattr(self.opt, 'downsample_first_layer') and self.opt.downsample_first_layer:
            x = self.up(x)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        if hasattr(self.opt, 'segmentation') and self.opt.segmentation:
            # un-normalized log prob
            pass
        else:
            # rescale to [-1, 1]
            x = torch.tanh(x)

        return x





class BaseSegOutpaintingGenerator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        opt.shortcut_nc = opt.label_nc + 1 #noseg -> 1
        if hasattr(opt, 'segmentation') and opt.segmentation:
            opt.shortcut_nc = 3 + 1
        self.opt = opt
        nf = opt.ngf

        if self.opt.use_gpu:
            self.encoder_norm = torch.nn.SyncBatchNorm
        else:
            self.encoder_norm = torch.nn.BatchNorm2d
        self.actvn = nn.LeakyReLU(0.2, False)

        if opt.crop_size == 512 and hasattr(opt, 'downsample_first_layer') and opt.downsample_first_layer:
            self.conv_0 = nn.Conv2d(opt.semantic_nc, 1 * nf, kernel_size=7, padding=3, stride=2)
        else:
            self.conv_0 = nn.Conv2d(opt.semantic_nc, 1 * nf, kernel_size=7, padding=3)
        self.conv_1 = spectral_norm(nn.Conv2d(1 * nf, 2 * nf, kernel_size=3, padding=1))
        self.down_0 = spectral_norm(nn.Conv2d(2 * nf, 2 * nf, kernel_size=3, stride=2, padding=1))
        self.conv_2 = spectral_norm(nn.Conv2d(2 * nf, 4 * nf, kernel_size=3, padding=1))
        self.down_1 = spectral_norm(nn.Conv2d(4 * nf, 4 * nf, kernel_size=3, stride=2, padding=1))
        self.conv_3 = spectral_norm(nn.Conv2d(4 * nf, 8 * nf, kernel_size=3, padding=1))
        self.down_2 = spectral_norm(nn.Conv2d(8 * nf, 8 * nf, kernel_size=3, stride=2, padding=1))
        self.conv_4 = spectral_norm(nn.Conv2d(8 * nf, 16 * nf, kernel_size=3, padding=1))
        self.down_3 = spectral_norm(nn.Conv2d(16 * nf, 16 * nf, kernel_size=3, stride=2, padding=1))
        self.conv_5 = spectral_norm(nn.Conv2d(16 * nf, 16 * nf, kernel_size=3, padding=1))
        self.down_4 = spectral_norm(nn.Conv2d(16 * nf, 16 * nf, kernel_size=3, stride=2, padding=1))

        self.bn_0 = self.encoder_norm(1 * nf)
        self.bn_1 = self.encoder_norm(2 * nf)
        self.bn_2 = self.encoder_norm(2 * nf)
        self.bn_3 = self.encoder_norm(4 * nf)
        self.bn_4 = self.encoder_norm(4 * nf)
        self.bn_5 = self.encoder_norm(8 * nf)
        self.bn_6 = self.encoder_norm(8 * nf)
        self.bn_7 = self.encoder_norm(16 * nf)
        self.bn_8 = self.encoder_norm(16 * nf)
        self.bn_9 = self.encoder_norm(16 * nf)

        self.G_middle_0 = ResnetBlockWithSN(16 * nf, 16 * nf, opt)
        self.G_middle_1 = ResnetBlockWithSN(16 * nf, 16 * nf, opt)
        self.G_middle_2 = ResnetBlockWithSN(16 * nf, 16 * nf, opt)

        self.up_0 = ResnetBlockWithSN(16 * nf, 8 * nf, opt)
        self.up_1 = ResnetBlockWithSN(8 * nf, 4 * nf, opt)
        self.up_2 = ResnetBlockWithSN(4 * nf, 2 * nf, opt)
        self.up_3 = ResnetBlockWithSN(2 * nf, 1 * nf, opt)

        # self.G_middle_0 = AtResnetBlock(16 * nf, 16 * nf, opt)
        # self.G_middle_1 = AtResnetBlock(16 * nf, 16 * nf, opt)
        # self.G_middle_2 = AtResnetBlock(16 * nf, 16 * nf, opt)

        # self.up_0 = AtResnetBlock(16 * nf, 8 * nf, opt)
        # self.up_1 = AtResnetBlock(8 * nf, 4 * nf, opt)
        # self.up_2 = AtResnetBlock(4 * nf, 2 * nf, opt)
        # self.up_3 = AtResnetBlock(2 * nf, 1 * nf, opt)




        final_nc = nf

        
        if hasattr(opt, 'segmentation') and opt.segmentation:
            # generate seg map
            self.conv_img = nn.Conv2d(final_nc, opt.label_nc, 3, padding=1)
        else:
            # generate image
            self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        num_up_layers = 5

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, image_masked, seg, mask):
        if seg is not None:
            x = torch.cat([image_masked, seg, mask], dim=1)
            shortcut_input = torch.cat([seg, mask], dim=1)
            if hasattr(self.opt, 'segmentation') and self.opt.segmentation:
                shortcut_input = torch.cat([image_masked, mask], dim=1)
        else:
            x = torch.cat([image_masked, mask], dim=1)
            shortcut_input = None

        out_0 = self.conv_0(x)                              # 64 * 254 * 256
        out_1 = self.conv_1(self.actvn(self.bn_0(out_0)))   # 128 * 256 * 256
        out_2 = self.down_0(self.actvn(self.bn_1(out_1)))   # 128 * 128 * 128
        out_3 = self.conv_2(self.actvn(self.bn_2(out_2)))   # 256 * 128 * 128
        out_4 = self.down_1(self.actvn(self.bn_3(out_3)))   # 256 * 64 * 64
        out_5 = self.conv_3(self.actvn(self.bn_4(out_4)))   # 512 * 64 *64
        out_6 = self.down_2(self.actvn(self.bn_5(out_5)))   # 512 * 32 * 32
        out_7 = self.conv_4(self.actvn(self.bn_6(out_6)))   # 1024 * 32 * 32
        out_8 = self.down_3(self.actvn(self.bn_7(out_7)))   # 1024 * 16 * 16
        out_9 = self.conv_5(self.actvn(self.bn_8(out_8)))   # 1024 * 16 * 16
        out_10 = self.down_4(self.actvn(self.bn_9(out_9)))  # 1024 * 8 * 8

        x = self.G_middle_0(out_10)                 # 1024 * 8 * 8
        x = self.up(x)                                              # 1024 * 16 * 16
        x = self.G_middle_1(x)                      # 1024 * 16 * 16
        x = self.G_middle_2(x)                      # 1024 * 16 * 16

        x = self.up(x)                                              # 1024 * 32 * 32
        x = self.up_0(x)                            # 512 * 32 * 32
        x = self.up(x)                                              # 512 * 64 * 64
        x = self.up_1(x)                            # 256 * 64 * 64
        x = self.up(x)                                              # 256 * 128 * 128
        x = self.up_2(x)                            # 128 * 128 * 128
        x = self.up(x)                                              # 128 * 256 * 256
        x = self.up_3(x)                            # 64 * 256 * 256

        if self.opt.crop_size == 512 and hasattr(self.opt, 'downsample_first_layer') and self.opt.downsample_first_layer:
            x = self.up(x)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        if hasattr(self.opt, 'segmentation') and self.opt.segmentation:
            # un-normalized log prob
            pass
        else:
            # rescale to [-1, 1]
            x = torch.tanh(x)


        #x = torch.tanh(x)

        return x





class randomBaseSegOutpaintingGenerator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        opt.shortcut_nc = opt.label_nc + 1 #noseg -> 1
        if hasattr(opt, 'segmentation') and opt.segmentation:
            opt.shortcut_nc = 3 + 1
        self.opt = opt
        nf = opt.ngf



        if self.opt.use_gpu:
            self.encoder_norm = torch.nn.SyncBatchNorm
        else:
            self.encoder_norm = torch.nn.BatchNorm2d
        self.actvn = nn.LeakyReLU(0.2, False)


        #self.fc_0 = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        #self.fc_1 =


        if opt.crop_size == 512 and hasattr(opt, 'downsample_first_layer') and opt.downsample_first_layer:
            self.conv_0 = nn.Conv2d(opt.semantic_nc, 1 * nf, kernel_size=7, padding=3, stride=2)
        else:
            self.conv_0 = nn.Conv2d(opt.semantic_nc, 1 * nf, kernel_size=7, padding=3)
        self.conv_1 = spectral_norm(nn.Conv2d(1 * nf, 2 * nf, kernel_size=3, padding=1))
        self.down_0 = spectral_norm(nn.Conv2d(2 * nf, 2 * nf, kernel_size=3, stride=2, padding=1))
        self.conv_2 = spectral_norm(nn.Conv2d(2 * nf, 4 * nf, kernel_size=3, padding=1))
        self.down_1 = spectral_norm(nn.Conv2d(4 * nf, 4 * nf, kernel_size=3, stride=2, padding=1))
        self.conv_3 = spectral_norm(nn.Conv2d(4 * nf, 8 * nf, kernel_size=3, padding=1))
        self.down_2 = spectral_norm(nn.Conv2d(8 * nf, 8 * nf, kernel_size=3, stride=2, padding=1))
        self.conv_4 = spectral_norm(nn.Conv2d(8 * nf, 16 * nf, kernel_size=3, padding=1))
        self.down_3 = spectral_norm(nn.Conv2d(16 * nf, 16 * nf, kernel_size=3, stride=2, padding=1))
        self.conv_5 = spectral_norm(nn.Conv2d(16 * nf, 16 * nf, kernel_size=3, padding=1))
        self.down_4 = spectral_norm(nn.Conv2d(16 * nf, 16 * nf, kernel_size=3, stride=2, padding=1))


        self.bn_0 = self.encoder_norm(1 * nf)
        self.bn_1 = self.encoder_norm(2 * nf)
        self.bn_2 = self.encoder_norm(2 * nf)
        self.bn_3 = self.encoder_norm(4 * nf)
        self.bn_4 = self.encoder_norm(4 * nf)
        self.bn_5 = self.encoder_norm(8 * nf)
        self.bn_6 = self.encoder_norm(8 * nf)
        self.bn_7 = self.encoder_norm(16 * nf)
        self.bn_8 = self.encoder_norm(16 * nf)
        self.bn_9 = self.encoder_norm(16 * nf)

        self.G_middle_0 = ResnetBlockWithSN_random(16 * nf, 16 * nf, opt)
        self.G_middle_1 = ResnetBlockWithSN_random(16 * nf, 16 * nf, opt)
        self.G_middle_2 = ResnetBlockWithSN_random(16 * nf, 16 * nf, opt)

        self.up_0 = ResnetBlockWithSN_random(16 * nf, 8 * nf, opt)
        self.up_1 = ResnetBlockWithSN_random(8 * nf, 4 * nf, opt)
        self.up_2 = ResnetBlockWithSN_random(4 * nf, 2 * nf, opt)
        self.up_3 = ResnetBlockWithSN_random(2 * nf, 1 * nf, opt)

        # self.G_middle_0 = AtResnetBlock(16 * nf, 16 * nf, opt)
        # self.G_middle_1 = AtResnetBlock(16 * nf, 16 * nf, opt)
        # self.G_middle_2 = AtResnetBlock(16 * nf, 16 * nf, opt)

        # self.up_0 = AtResnetBlock(16 * nf, 8 * nf, opt)
        # self.up_1 = AtResnetBlock(8 * nf, 4 * nf, opt)
        # self.up_2 = AtResnetBlock(4 * nf, 2 * nf, opt)
        # self.up_3 = AtResnetBlock(2 * nf, 1 * nf, opt)




        final_nc = nf

        
        if hasattr(opt, 'segmentation') and opt.segmentation:
            # generate seg map
            self.conv_img = nn.Conv2d(final_nc, opt.label_nc, 3, padding=1)
        else:
            # generate image
            self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        num_up_layers = 5

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, image_masked, seg, mask):
        if seg is not None:
            x = torch.cat([image_masked, seg, mask], dim=1)
            shortcut_input = torch.cat([seg, mask], dim=1)
            if hasattr(self.opt, 'segmentation') and self.opt.segmentation:
                shortcut_input = torch.cat([image_masked, mask], dim=1)
        else:
            x = torch.cat([image_masked, mask], dim=1)
            shortcut_input = None


        mean = np.zeros(self.opt.z_dim)
        cov = np.identity(self.opt.z_dim)

        z = np.random.multivariate_normal(mean, cov, (self.opt.batchSize)) # bs*128
        #print(z)
        z = torch.from_numpy(z).float()   
        if self.opt.use_gpu:
            z = z.cuda()
        



        out_0 = self.conv_0(x)                              # 64 * 254 * 256
        out_1 = self.conv_1(self.actvn(self.bn_0(out_0)))   # 128 * 256 * 256
        out_2 = self.down_0(self.actvn(self.bn_1(out_1)))   # 128 * 128 * 128
        out_3 = self.conv_2(self.actvn(self.bn_2(out_2)))   # 256 * 128 * 128
        out_4 = self.down_1(self.actvn(self.bn_3(out_3)))   # 256 * 64 * 64
        out_5 = self.conv_3(self.actvn(self.bn_4(out_4)))   # 512 * 64 *64
        out_6 = self.down_2(self.actvn(self.bn_5(out_5)))   # 512 * 32 * 32
        out_7 = self.conv_4(self.actvn(self.bn_6(out_6)))   # 1024 * 32 * 32
        out_8 = self.down_3(self.actvn(self.bn_7(out_7)))   # 1024 * 16 * 16
        out_9 = self.conv_5(self.actvn(self.bn_8(out_8)))   # 1024 * 16 * 16
        out_10 = self.down_4(self.actvn(self.bn_9(out_9)))  # 1024 * 8 * 8

        x = self.G_middle_0(out_10,z)                 # 1024 * 8 * 8
        x = self.up(x)                                              # 1024 * 16 * 16
        x = self.G_middle_1(x, z)                      # 1024 * 16 * 16
        x = self.G_middle_2(x, z)                      # 1024 * 16 * 16

        x = self.up(x)                                              # 1024 * 32 * 32
        x = self.up_0(x,z)                            # 512 * 32 * 32
        x = self.up(x)                                              # 512 * 64 * 64
        x = self.up_1(x,z)                            # 256 * 64 * 64
        x = self.up(x)                                              # 256 * 128 * 128
        x = self.up_2(x,z)                            # 128 * 128 * 128
        x = self.up(x)                                              # 128 * 256 * 256
        x = self.up_3(x,z)                            # 64 * 256 * 256

        if self.opt.crop_size == 512 and hasattr(self.opt, 'downsample_first_layer') and self.opt.downsample_first_layer:
            x = self.up(x)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        if hasattr(self.opt, 'segmentation') and self.opt.segmentation:
            # un-normalized log prob
            pass
        else:
            # rescale to [-1, 1]
            x = torch.tanh(x)


        #x = torch.tanh(x)

        return x



# x = torch.cat([x, short_3], dim=1)  



