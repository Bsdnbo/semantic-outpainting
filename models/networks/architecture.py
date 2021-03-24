import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.normalization import SPADE




class ResnetBlockWithSN_random(nn.Module):
    def __init__(self, fin, fout, opt, stride=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout) or (stride != 1)
        fmiddle = min(fin, fout)
        self.fin = fin
        self.fmiddle = fmiddle
        self.fout = fout

        self.z_lfc0 = nn.Linear(opt.z_dim, opt.z_dim)
        self.z_lfc1 = nn.Linear(opt.z_dim, opt.z_dim)
        
        self.z_fc_0 = nn.Linear(opt.z_dim, 2 * fin)
        self.z_fc_1 = nn.Linear(opt.z_dim, 2 * fmiddle)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1, stride=stride)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1, stride=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False, stride=stride)
            self.z_fc_s = nn.Linear(opt.z_dim, 2 * fin)

        # apply spectral norm if specified
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        if 'syncbatch' in opt.norm_G:
            self.norm_0 = nn.SyncBatchNorm(fin, affine=False)
            self.norm_1 = nn.SyncBatchNorm(fmiddle, affine=False)
            if self.learned_shortcut:
                self.norm_s = nn.SyncBatchNorm(fin, affine=False)
        elif 'instance' in opt.norm_G:
            self.norm_0 = nn.InstanceNorm2d(fin, affine=True)
            self.norm_1 = nn.InstanceNorm2d(fmiddle, affine=True)
            if self.learned_shortcut:
                self.norm_s = nn.InstanceNorm2d(fin, affine=True)
        else:
            self.norm_0 = nn.BatchNorm2d(fin, affine=True)
            self.norm_1 = nn.BatchNorm2d(fmiddle, affine=True)
            if self.learned_shortcut:
                self.norm_s = nn.BatchNorm2d(fin, affine=True)

    def forward(self, x, z):
        # z: batchsize x z_dim
        if self.learned_shortcut:
            m_batchsize, C, width ,height = x.size() 
            x = self.norm_s(x)
            dz = self.z_fc_s(z) # bs x 2fin
            # divide dz to mean and dev
            x = x.view(m_batchsize,-1,width*height) # bs x fin x (wh)

            scale0 = torch.split(dz,self.fin, dim=1)[0].view(m_batchsize, self.fin, 1)
            shift0 = torch.split(dz,self.fin, dim=1)[1].view(m_batchsize, self.fin, 1) # bs x self.fin x1
            scale0 = scale0.expand(m_batchsize, self.fin, width*height)
            shift0 = shift0.expand(m_batchsize, self.fin, width*height) # bs x self.fin x wh
            #print("scale0: " + str(scale0))
            #print("shift0: "+str(shift0))
            x = torch.mul(x, scale0) + shift0
            x = x.view(m_batchsize, C, width ,height)

            x_s = self.conv_s(self.actvn(x))
        else:
            x_s = x
        m_batchsize, C, width ,height = x.size() # bs x self.fin x w x h

        x = self.norm_0(x) # bs x self.fin x w x h

        dz = self.actvn(self.z_lfc0(z))

        dz = self.z_fc_0(dz) # bs x 2self.fin
        # divide dz to mean and dev
        x = x.view(m_batchsize,-1,width*height) # bs x self.fin x (wh)

        scale0 = torch.split(dz,self.fin, dim=1)[0].view(m_batchsize, self.fin, 1)
        shift0 = torch.split(dz,self.fin, dim=1)[1].view(m_batchsize, self.fin, 1) # bs x self.fin x1
        scale0 = scale0.expand(m_batchsize, self.fin, width*height)
        shift0 = shift0.expand(m_batchsize, self.fin, width*height) # bs x self.fin x wh
        x = torch.mul(x, (1+scale0)) + shift0
        x = x.view(m_batchsize, C, width ,height)
        #mean1 = 
        
        dx = self.conv_0(self.actvn(x))

        dx = self.norm_1(dx)

        dz = self.actvn(self.z_lfc1(z))
        dz = self.z_fc_1(dz) # bs x 2self.fmiddle
        # divide dz to mean and dev
        dx = dx.view(m_batchsize,-1,width*height) # bs x self.fmiddle x (wh)

        scale1 = torch.split(dz,self.fmiddle, dim=1)[0].view(m_batchsize, self.fmiddle, 1)
        shift1 = torch.split(dz,self.fmiddle, dim=1)[1].view(m_batchsize, self.fmiddle, 1) # bs x self.fin x1
        scale1 = scale1.expand(m_batchsize, self.fmiddle, width*height)
        shift1 = shift1.expand(m_batchsize, self.fmiddle, width*height) # bs x self.fin x wh
        dx = torch.mul(dx, (1+scale1)) + shift1
        dx = dx.view(m_batchsize, self.fmiddle, width ,height)

        dx = self.conv_1(self.actvn(dx))

        out = x_s + dx

        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# SegoutSPADEResnetBlock
class SegoutSPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        no_spade = hasattr(opt, 'no_spade') and opt.no_spade is True
        self.norm_0 = SPADE(spade_config_str, fin, opt.shortcut_nc, no_spade=no_spade)
        self.norm_1 = SPADE(spade_config_str, fmiddle, opt.shortcut_nc, no_spade=no_spade)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, opt.shortcut_nc, no_spade=no_spade)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    # seg = cat [seg, mask]
    def forward(self, x, seg): 
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.actvn(self.norm_s(x, seg)))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)



class ResnetBlockWithSN_cat(nn.Module):
    def __init__(self, fin, fout, opt, stride=1):
        super().__init__()
        # Attributes
        fmiddle = min(fin, fout)
        fin = fin + opt.shortcut_nc
        self.learned_shortcut = (fin != fout) or (stride != 1)
        

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1, stride=stride)
        self.conv_1 = nn.Conv2d(fmiddle + opt.shortcut_nc, fout, kernel_size=3, padding=1, stride=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False, stride=stride)

        # apply spectral norm if specified
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        if 'syncbatch' in opt.norm_G:
            self.norm_0 = nn.SyncBatchNorm(fin, affine=True)
            self.norm_1 = nn.SyncBatchNorm(fmiddle + opt.shortcut_nc, affine=True)
            if self.learned_shortcut:
                self.norm_s = nn.SyncBatchNorm(fin, affine=True)
        elif 'instance' in opt.norm_G:
            self.norm_0 = nn.InstanceNorm2d(fin, affine=True)
            self.norm_1 = nn.InstanceNorm2d(fmiddle + opt.shortcut_nc, affine=True)
            if self.learned_shortcut:
                self.norm_s = nn.InstanceNorm2d(fin, affine=True)
        else:
            self.norm_0 = nn.BatchNorm2d(fin, affine=True)
            self.norm_1 = nn.BatchNorm2d(fmiddle + opt.shortcut_nc, affine=True)
            if self.learned_shortcut:
                self.norm_s = nn.BatchNorm2d(fin, affine=True)

    def forward(self, x, shortcut_input):

        shortcut_input = F.interpolate(shortcut_input, size=x.size()[2:], mode='nearest') # 8 152 8 8

        if self.learned_shortcut:
            x_s = self.conv_s(self.actvn(self.norm_s(torch.cat([x, shortcut_input], dim=1) )))
        else:
            x_s = x

        dx = self.conv_0(self.actvn(self.norm_0(torch.cat([x, shortcut_input], dim=1) )))
        dx = self.conv_1(self.actvn(self.norm_1(torch.cat([dx, shortcut_input], dim=1) )))

        out = x_s + dx

        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)





# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        no_spade = hasattr(opt, 'no_spade') and opt.no_spade is True
        self.norm_0 = SPADE(spade_config_str, fin, opt.shortcut_nc, no_spade=no_spade)
        self.norm_1 = SPADE(spade_config_str, fmiddle, opt.shortcut_nc, no_spade=no_spade)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, opt.shortcut_nc, no_spade=no_spade)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    # seg = cat [seg, mask]
    def forward(self, x, seg): 
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.actvn(self.norm_s(x, seg)))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class ResnetBlockWithSN(nn.Module):
    def __init__(self, fin, fout, opt, stride=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout) or (stride != 1)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1, stride=stride)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1, stride=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False, stride=stride)

        # apply spectral norm if specified
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        if 'syncbatch' in opt.norm_G:
            self.norm_0 = nn.SyncBatchNorm(fin, affine=True)
            self.norm_1 = nn.SyncBatchNorm(fmiddle, affine=True)
            if self.learned_shortcut:
                self.norm_s = nn.SyncBatchNorm(fin, affine=True)
        elif 'instance' in opt.norm_G:
            self.norm_0 = nn.InstanceNorm2d(fin, affine=True)
            self.norm_1 = nn.InstanceNorm2d(fmiddle, affine=True)
            if self.learned_shortcut:
                self.norm_s = nn.InstanceNorm2d(fin, affine=True)
        else:
            self.norm_0 = nn.BatchNorm2d(fin, affine=True)
            self.norm_1 = nn.BatchNorm2d(fmiddle, affine=True)
            if self.learned_shortcut:
                self.norm_s = nn.BatchNorm2d(fin, affine=True)

    def forward(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(self.actvn(self.norm_s(x)))
        else:
            x_s = x

        dx = self.conv_0(self.actvn(self.norm_0(x)))
        dx = self.conv_1(self.actvn(self.norm_1(dx)))

        out = x_s + dx

        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, model_path, requires_grad=False):
        super().__init__()
        vgg19 = torchvision.models.vgg19(pretrained=False)
        vgg19.load_state_dict(torch.load(model_path))
        vgg_pretrained_features = vgg19.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
