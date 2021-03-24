import os
import sys
from argparse import ArgumentParser

import cv2
from torch.utils.data import DataLoader
import numpy as np
from data.ade20k_dataset import ADE20KDataset
from models.pix2pix_model import Pix2PixModel
from scipy.io import loadmat
from multiprocessing.pool import ThreadPool


parser = ArgumentParser()
parser.add_argument('--name', type=str, default='0.5outpainting_512', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--save_image', action='store_true', help='Whether to save images')
parser.add_argument('--show_image', action='store_true', help='Whether to show images')
parser.add_argument('--which_epoch', type=str, default='', help='which epoch to load? set to latest to use latest cached model')
parser.add_argument('--checkpoints_dir', type=str, default='../checkpoints/', help='models are saved here')
parser.add_argument('--use_gpu', action='store_true', default=False)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--load_size', type=int, default=286, help='Scale images to this size. The final image will be cropped to --crop_size.')
parser.add_argument('--crop_size', type=int, default=256, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
parser.add_argument('--label_nc', type=int, default=151, help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
parser.add_argument('--contain_dontcare_label', action='store_true', default=True, help='if the label map contains dontcare label (dontcare=255)')
parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--norm_G', type=str, default='spectralspadesyncbatch3x3', help='instance normalization or batch normalization')
parser.add_argument('--baseSegOutpaintingGenerator', action='store_true', default=False, help='Use cat[img,seg]')
parser.add_argument('--input_masked', action='store_true', default=False, help='Use block cat[img,seg]')


parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
parser.add_argument('--z_dim', type=int, default=256,
                    help="dimension of the latent z vector")
parser.add_argument('--num_upsampling_layers',
                    choices=('normal', 'more', 'most'), default='normal',
                    help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

# for instance-wise features
parser.add_argument('--no_instance', action='store_true', default=True, help='if specified, do *not* add instance map as input')
parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
parser.add_argument('--use_vae', action='store_true', default=False, help='enable training with an image encoder.')
parser.add_argument('--no_spade', action='store_true', default=False)
parser.add_argument('--no_seg', action='store_true', default=False)
parser.add_argument('--dataset', type=str, default='ade20k')
parser.add_argument('--dataroot', type=str, default='')
parser.add_argument('--downsample_first_layer', action='store_true', default=False)
parser.add_argument('--use_seg_predict', action='store_true', default=False)
parser.add_argument('--D_checkpoint_name', type=str, default='latest', help='checkpoint name of discriminator')
parser.add_argument('--G_checkpoint_name', type=str, default='0.5_net_G_image_outpainting.pth', help='checkpoint name of generator')
parser.add_argument('--localD_checkpoint_name', type=str, default='latest', help='checkpoint name of localD')

parser.add_argument('--ratio', type=float, default=0.5, help='# of encoder filters in the first conv layer')


opt = parser.parse_args()

opt.isTrain = False
opt.outpainting = True
opt.l1_loss = True
if opt.no_seg:
    opt.semantic_nc = 3 + 1
    opt.label_nc = 0
else:
    opt.semantic_nc = opt.label_nc + 3 + 1


if opt.save_image:
    opt.save_path = '../results/%s/%s/images/' % (opt.name, opt.which_epoch)
    os.makedirs(opt.save_path, exist_ok=True)
    os.makedirs(opt.save_path + 'seg/', exist_ok=True)
    os.makedirs(opt.save_path + 'gen/', exist_ok=True)
    os.makedirs(opt.save_path + 'gen_new/', exist_ok=True)
    os.makedirs(opt.save_path + 'orig/', exist_ok=True)

opt.max_dataset_size = sys.maxsize
if not opt.use_gpu:
    opt.nThreads = 0
else:
    opt.nThreads = 0 # 32

model = Pix2PixModel(opt)
model.eval()

dataset = ADE20KDataset(opt)
print("dataset [%s] of size %d was created" % (type(dataset).__name__, len(dataset)))
dataloader = DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.nThreads),
    drop_last=opt.isTrain
)

colors = loadmat('../color150.mat')['colors']
colors = np.concatenate([[[0, 0, 0]], colors], axis=0)


def tensor2img(tensor):
    # tensor in shape [c, h, w]
    x = tensor.data.cpu().numpy()
    image_numpy = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(np.uint8)


def label2img(tensor):
    # tensor in shape [h, w]
    x = tensor.data.cpu().numpy()
    return label_to_color_map(x)


def label_to_color_map(x):
    # x in shape (h, w)
    new_x = np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8)
    x = x.astype(np.int32)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            new_x[i, j] = colors[x[i, j]]
    return new_x


def process_and_save_img(img_idx, img_orig, img_masked, label, generated, mask):
    img_orig_ = tensor2img(img_orig)
    img_masked_ = tensor2img(img_masked)
    if label is not None:
        label_ = label2img(label)
    gen_ = tensor2img(generated)
    mask_ = mask.data.cpu().numpy()
    mask_ = mask_[0, :, :, np.newaxis]
    gen_2 = gen_ * mask_ + img_masked_
    gen_2 = gen_2.astype(np.uint8)

    cv2.imwrite(opt.save_path + 'gen/%d-gen.png' % img_idx, gen_)
    cv2.imwrite(opt.save_path + 'gen_new/%d-gen_new.png' % img_idx, gen_2)
    cv2.imwrite(opt.save_path + 'orig/%d-orig.png' % img_idx, img_orig_)
    if label is not None:
        cv2.imwrite(opt.save_path + 'seg/%d-seg.png' % img_idx, label_)


# test
img_idx = 0
for i, data_i in enumerate(dataloader):
    print(i, '/', len(dataloader))
    generated = model(data_i, mode='inference')
    if opt.no_seg:
        img_orig, img_masked, mask = data_i
        label = None
    else:
        img_orig, img_masked, label, label_masked, label_one_hot, label_one_hot_masked, mask = data_i

    for b in range(generated.shape[0]):
        if opt.save_image:
            if opt.no_seg:
                process_and_save_img(img_idx, img_orig[b], img_masked[b], None, generated[b], mask[b])
            else:
                process_and_save_img(img_idx, img_orig[b], img_masked[b], label[b], generated[b], mask[b])
        else:
            img_orig_ = tensor2img(img_orig[b])
            img_masked_ = tensor2img(img_masked[b])
            if label is not None:
                label_ = label2img(label[b])
            gen_ = tensor2img(generated[b])
            mask_ = mask[b].data.cpu().numpy()
            mask_ = mask_[0, :, :, np.newaxis]
            gen_2 = gen_ * mask_ + img_masked_
            gen_2 = gen_2.astype(np.uint8)

            if opt.show_image:
                cv2.imshow('orig', img_orig_)
                if label is not None:
                    cv2.imshow('label', label_)
                cv2.imshow('gen', gen_)
                cv2.imshow('gen_new', gen_2)
                cv2.waitKey(0)
        img_idx += 1

print('Done')