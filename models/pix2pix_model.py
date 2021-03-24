import torch
import models.networks as networks
import util.utils as util
from models.networks.generator import SPADEOutpaintingGenerator, BaseSegOutpaintingGenerator
from models.networks.discriminator import MultiscaleDiscriminator
import numpy as np

class Pix2PixModel(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.opt.use_gpu \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.opt.use_gpu \
            else torch.ByteTensor

        if not hasattr(self.opt, 'segmentation'):
            self.opt.segmentation = False
        if not hasattr(self.opt, 'ce_loss'):
            self.opt.ce_loss = False
        self.netG, self.netD, self.netD_local = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        if len(data) == 7:
            # with segmentation
            img_orig, img_masked, label, label_masked, label_one_hot, label_one_hot_masked, mask = data
            if self.opt.use_gpu:
                img_orig, img_masked, label, label_masked, label_one_hot, label_one_hot_masked, mask = \
                    img_orig.cuda(), img_masked.cuda(), label.cuda(), label_masked.cuda(), label_one_hot.cuda(), label_one_hot_masked.cuda(), mask.cuda()
        else:
            # no segmentation
            img_orig, img_masked, mask = data
            if self.opt.use_gpu:
                img_orig, img_masked, mask = img_orig.cuda(), img_masked.cuda(), mask.cuda()
            label_one_hot = None
            label_one_hot_masked = None
            label = None
        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                img_orig, img_masked, label_one_hot, label_one_hot_masked, mask, label)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(img_orig, img_masked, label_one_hot, label_one_hot_masked, mask)
            return d_loss
        elif mode == 'inference':
            with torch.no_grad():
                if self.opt.outpainting:
                    fake_image = self.netG(img_masked, label_one_hot, mask)
                elif self.opt.segmentation:
                    fake_image = self.netG(img_masked, label_one_hot_masked, mask)
                    fake_image = torch.argmax(fake_image, dim=1)
                else:
                    fake_image = self.netG(label_one_hot)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.contextD:
            util.save_network(self.netD, 'D_local', epoch, self.opt)
    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        if opt.outpainting:
            if hasattr(opt, 'baseSegOutpaintingGenerator') and opt.baseSegOutpaintingGenerator:
                netG = BaseSegOutpaintingGenerator(opt)
            else:
                netG = SPADEOutpaintingGenerator(opt)
        elif opt.segmentation:
            netG = BaseSegOutpaintingGenerator(opt)
        else:
            assert opt.outpainting or opt.segmentation
        netG.print_network()
        netG.init_weights(opt.init_type, opt.init_variance)
        
        if not opt.isTrain:
            netD = None
            netD_local = None
        else:
            netD = MultiscaleDiscriminator(opt)
            netD.print_network()
            netD.init_weights(opt.init_type, opt.init_variance)

            if self.opt.contextD:
                netD_local = MultiscaleDiscriminator(opt)
                netD_local.print_network()
                netD_local.init_weights(opt.init_type, opt.init_variance)
            else:
                netD_local = None

        if self.opt.use_gpu:
            assert (torch.cuda.is_available())
            netG.cuda()
            if netD:
                netD.cuda()
            if netD_local:
                netD_local.cuda()
        if not opt.isTrain or opt.continue_train:
            print("begin to load")
            netG = util.load_network(netG, opt.G_checkpoint_name, opt)
            if opt.isTrain:
                netD = util.load_network(netD, opt.D_checkpoint_name , opt)
                if opt.contextD:
                    netD_local = util.load_network(netD, opt.localD_checkpoint_name, opt)
        return netG, netD, netD_local

    def compute_generator_loss(self, img_orig, img_masked, label_one_hot, label_one_hot_masked, mask, label=None):
        G_losses = {}

        if self.opt.outpainting:
            # Outpainting using full seg map
            img_gen = self.netG(img_masked, label_one_hot, mask)
        elif self.opt.segmentation:
            # Segmentation prediction
            img_gen = self.netG(img_masked, label_one_hot_masked, mask)
        else:
            # SPADE generate image from label
            img_gen = self.netG(label_one_hot)
        label_one_hot_out = None
        if (label_one_hot is not None) and (label_one_hot_masked is not None):
            label_one_hot_out = label_one_hot - label_one_hot_masked
        # Important!
        # For segmentation prediction task
        # label_one_hot -> img_masked
        # img_gen -> seg_map_gen
        # img_orig -> label_one_hot_masked
        if self.opt.segmentation:
            pred_fake, pred_real = self.discriminate(img_masked, img_gen, label_one_hot, mask, label_one_hot_out)
        else:
            if self.opt.contextD:
                pred_fake, pred_real = self.discriminate(label_one_hot, img_gen, img_orig, mask, label_one_hot_out)
            else:
                pred_fake, pred_real = self.discriminate(label_one_hot, img_gen, img_orig, mask, label_one_hot_out)
        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)
        if self.opt.l1_loss:
            G_losses['L1Loss'] = self.opt.lambda_l1 * torch.nn.functional.l1_loss(img_gen, img_orig)
        if self.opt.ce_loss:
            G_losses['CELoss'] = self.opt.lambda_l1 * torch.nn.functional.cross_entropy(img_gen, label)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(img_gen, img_orig) \
                * self.opt.lambda_vgg

        return G_losses, img_gen

    def compute_discriminator_loss(self, img_orig, img_masked, label_one_hot, label_one_hot_masked, mask):
        D_losses = {}
        label_one_hot_out = None
        if (label_one_hot is not None) and (label_one_hot_masked is not None):
            label_one_hot_out = label_one_hot - label_one_hot_masked

        with torch.no_grad():
            if self.opt.outpainting:
                fake_image = self.netG(img_masked, label_one_hot, mask)
            elif self.opt.segmentation:
                fake_image = self.netG(img_masked, label_one_hot_masked, mask)
            else:
                fake_image = self.netG(label_one_hot)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        # Important!
        # For segmentation prediction task
        # label_one_hot -> img_masked
        # img_gen -> seg_map_gen
        # img_orig -> label_one_hot_masked
        if self.opt.segmentation:
            pred_fake, pred_real = self.discriminate(img_masked, fake_image, label_one_hot, mask, label_one_hot_out)
        else:
            pred_fake, pred_real = self.discriminate(label_one_hot, fake_image, img_orig, mask, label_one_hot_out)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, label_one_hot, img_gen, img_orig, mask, label_one_hot_out):
        # Important!  add a param
        # For segmentation prediction task
        # label_one_hot -> img_masked
        # img_gen -> seg_map_gen
        # img_orig -> label_one_hot
        if self.opt.outpainting:
            img_gen_to_d = img_gen * mask + img_orig * (1 - mask)
        elif self.opt.segmentation:
            img_gen_to_d = img_gen * mask + img_orig * (1 - mask)
        else:
            img_gen_to_d = img_gen
        if label_one_hot is not None:
            fake_concat = torch.cat([label_one_hot, img_gen_to_d], dim=1)
            real_concat = torch.cat([label_one_hot, img_orig], dim=1)
        else:
            fake_concat = img_gen_to_d
            real_concat = img_orig

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        discriminator_out = self.netD(fake_and_real)
        if self.opt.contextD:
            img_gen_local = img_gen * mask
            img_orig_local = img_orig * mask
            img_gen_global = img_gen
            img_orig_global = img_orig
            label_one_hot_out
            label_one_hot

            fake_concat_local = torch.cat([label_one_hot_out, img_gen_local], dim=1)
            real_concat_local = torch.cat([label_one_hot_out, img_orig_local], dim=1)

            fake_and_real_global = torch.cat([fake_concat, real_concat], dim=0)
            fake_and_real_local = torch.cat([fake_concat_local, real_concat_local], dim=0)

            discriminator_out_local = self.netD_local(fake_and_real_local)
            discriminator_out =list((np.array(discriminator_out) + np.array(discriminator_out_local))/2)
            for i,c in enumerate(discriminator_out):
                discriminator_out[i]=list(c)
        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real
