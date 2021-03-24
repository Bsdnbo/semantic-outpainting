import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
import util.utils as util


class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = NLayerDiscriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = F.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        # input_nc = opt.label_nc + opt.output_nc + 1     # 151 + 3 + 1
        input_nc = opt.label_nc + opt.output_nc     # 151 + 3

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]




class GlobalDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt


        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        # input_nc = opt.label_nc + opt.output_nc + 1     # 151 + 3 + 1
        input_nc = opt.label_nc + opt.output_nc     # 151 + 3

        # calculate the channal

        channels = 3

        self.output_shape = (24, 24)

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


class LocalDiscriminator(BaseNetwork):
    def __init__(self, mask, channels=3):
        super(LocalDiscriminator, self).__init__()
        self.output_shape = (24, 24)

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img, mask):
        img = img * mask
        return self.model(img)


class ContextDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        #self.output_shape = (1,)

        # self.model_ld = LocalDiscriminator(local_input_shape)
        self.model_ld = NLayerDiscriminator(opt)
        self.add_module("local_discriminator", self.model_ld)
        # self.model_gd = GlobalDiscriminator(global_input_shape, arc=arc)
        self.model_gd = NLayerDiscriminator(opt)
        self.add_module("global_discriminator", self.model_gd)

        # TODO: Remove, this stuff gets handled afterwards
        # self.concat1 = layers.Concatenate(dim=-1)
        # self.flatten1 = nn.Flatten()
        # in_features = self.model_ld.output_shape[-1] ** 2 + self.model_gd.output_shape[-1] ** 2
        # self.linear1 = nn.Linear(in_features, 1)
        # self.act1 = nn.Sigmoid()
        

    def forward(self, fake_and_real_global, fake_and_real_local):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        

        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            # input = F.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)
        return result

    def forward(self, x, mask):
        x_ld = self.model_ld(x, mask)
        x_gd = self.model_gd(x)
        # concat = self.concat1([self.flatten1(x_ld), self.flatten1(x_gd)])
        # print('concatenated and flattened discriminator outputs', concat.shape)
        # lin = self.linear1(concat)
        # out = self.act1(lin)
        out = (x_ld + x_gd) / 2
        return out

