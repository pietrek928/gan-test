import argparse
import gc
import hashlib
import os
import random
from base64 import b64encode
from random import randint

import clip
import cv2
import imageio
import kornia.augmentation as K
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFile, ImageEnhance, ImageOps
from PIL.PngImagePlugin import PngImageFile, PngInfo
from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from dirs import get_path

torch.backends.cudnn.enabled = False
ImageFile.LOAD_TRUNCATED_IMAGES = True
args: argparse.Namespace = None


def noise_gen(shape, octaves=5):
    n, c, h, w = shape
    noise = torch.zeros([n, c, 1, 1])
    max_octaves = min(octaves, math.log(h) / math.log(2), math.log(w) / math.log(2))
    for i in reversed(range(max_octaves)):
        h_cur, w_cur = h // 2 ** i, w // 2 ** i
        noise = F.interpolate(noise, (h_cur, w_cur), mode='bicubic', align_corners=False)
        noise += torch.randn([n, c, h_cur, w_cur]) / 5
    return noise


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


def lerp(a, b, f):
    return (a * (1.0 - f)) + (b * f);


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


# def parse_prompt(prompt):
#    vals = prompt.rsplit(':', 2)
#    vals = vals + ['', '1', '-inf'][len(vals):]
#    return vals[0], float(vals[1]), float(vals[2])

def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 1)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])


def one_sided_clip_loss(input, target, labels=None, logit_scale=100):
    input_normed = F.normalize(input, dim=-1)
    target_normed = F.normalize(target, dim=-1)
    logits = input_normed @ target_normed.T * logit_scale
    if labels is None:
        labels = torch.arange(len(input), device=logits.device)
    return F.cross_entropy(logits, labels)


class EMATensor(nn.Module):
    """implmeneted by Katherine Crowson"""

    def __init__(self, tensor, decay):
        super().__init__()
        self.tensor = nn.Parameter(tensor)
        self.register_buffer('biased', torch.zeros_like(tensor))
        self.register_buffer('average', torch.zeros_like(tensor))
        self.decay = decay
        self.register_buffer('accum', torch.tensor(1.))
        self.update()

    @torch.no_grad()
    def update(self):
        if not self.training:
            raise RuntimeError('update() should only be called during training')

        self.accum *= self.decay
        self.biased.mul_(self.decay)
        self.biased.add_((1 - self.decay) * self.tensor)
        self.average.copy_(self.biased)
        self.average.div_(1 - self.accum)

    def forward(self):
        if self.training:
            return self.tensor
        return self.average


############################################################################################
############################################################################################


class MakeCutoutsCustom(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow, augs):
        super().__init__()
        self.cut_size = cut_size
        tqdm.write(f'cut size: {self.cut_size}')
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=Random_Horizontal_Flip),
            K.RandomSharpness(Random_Sharpness, p=Random_Sharpness_P),
            K.RandomGaussianBlur((Random_Gaussian_Blur), (Random_Gaussian_Blur_W, Random_Gaussian_Blur_W),
                                 p=Random_Gaussian_Blur_P),
            K.RandomGaussianNoise(p=Random_Gaussian_Noise_P),
            K.RandomElasticTransform(
                kernel_size=(Random_Elastic_Transform_Kernel_Size_W, Random_Elastic_Transform_Kernel_Size_H),
                sigma=(Random_Elastic_Transform_Sigma), p=Random_Elastic_Transform_P),
            K.RandomAffine(degrees=Random_Affine_Degrees, translate=Random_Affine_Translate, p=Random_Affine_P,
                           padding_mode='border'),
            K.RandomPerspective(Random_Perspective, p=Random_Perspective_P),
            K.ColorJitter(hue=Color_Jitter_Hue, saturation=Color_Jitter_Saturation, p=Color_Jitter_P), )
        # K.RandomErasing((0.1, 0.7), (0.3, 1/0.4), same_on_batch=True, p=0.2),)

    def set_cut_pow(self, cut_pow):
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        cutouts_full = []
        noise_fac = 0.1

        min_size_width = min(sideX, sideY)
        lower_bound = float(self.cut_size / min_size_width)

        for ii in range(self.cutn):
            # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            randsize = torch.zeros(1, ).normal_(mean=.8, std=.3).clip(lower_bound, 1.)
            size_mult = randsize ** self.cut_pow
            size = int(min_size_width * (
                size_mult.clip(lower_bound, 1.)))  # replace .5 with a result for 224 the default large size is .95
            # size = int(min_size_width*torch.zeros(1,).normal_(mean=.9, std=.3).clip(lower_bound, .95)) # replace .5 with a result for 224 the default large size is .95

            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

        cutouts = torch.cat(cutouts, dim=0)
        cutouts = clamp_with_grad(cutouts, 0, 1)

        # if args.use_augs:
        cutouts = self.augs(cutouts)
        if self.noise_fac:
            facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(0, self.noise_fac)
            cutouts = cutouts + facs * torch.randn_like(cutouts)
        return cutouts


class MakeCutoutsJuu(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow, augs):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            # K.RandomGaussianNoise(mean=0.0, std=0.5, p=0.1),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomSharpness(0.3, p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2, p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            K.RandomGrayscale(p=0.1),
        )
        self.noise_fac = 0.1

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


class MakeCutoutsMoth(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow, augs, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.skip_augs = skip_augs
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomPerspective(distortion_scale=0.4, p=0.7),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.15),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def forward(self, input):
        input = T.Pad(input.shape[2] // 4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(args.cutn):
            if ch > args.cutn - args.cutn // 4:
                cutout = input.clone()
            else:
                size = int(
                    max_size * torch.zeros(1, ).normal_(mean=.8, std=.3).clip(float(self.cut_size / max_size), 1.))
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        return cutouts


class MakeCutoutsAaron(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow, augs):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = augs
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def set_cut_pow(self, cut_pow):
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        cutouts_full = []

        min_size_width = min(sideX, sideY)
        lower_bound = float(self.cut_size / min_size_width)

        for ii in range(self.cutn):
            size = int(min_size_width * torch.zeros(1, ).normal_(mean=.8, std=.3).clip(lower_bound,
                                                                                       1.))  # replace .5 with a result for 224 the default large size is .95

            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

        cutouts = torch.cat(cutouts, dim=0)

        return clamp_with_grad(cutouts, 0, 1)


class MakeCutoutsCumin(nn.Module):
    # from https://colab.research.google.com/drive/1ZAus_gn2RhTZWzOWUpPERNC0Q8OhZRTZ
    def __init__(self, cut_size, cutn, cut_pow, augs):
        super().__init__()
        self.cut_size = cut_size
        tqdm.write(f'cut size: {self.cut_size}')
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
        self.augs = nn.Sequential(
            # K.RandomHorizontalFlip(p=0.5),
            # K.RandomSharpness(0.3,p=0.4),
            # K.RandomGaussianBlur((3,3),(10.5,10.5),p=0.2),
            # K.RandomGaussianNoise(p=0.5),
            # K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
            K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border'),
            K.RandomPerspective(0.7, p=0.7),
            K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
            K.RandomErasing((.1, .4), (.3, 1 / .3), same_on_batch=True, p=0.7), )

    def set_cut_pow(self, cut_pow):
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        cutouts_full = []
        noise_fac = 0.1

        min_size_width = min(sideX, sideY)
        lower_bound = float(self.cut_size / min_size_width)

        for ii in range(self.cutn):
            # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            randsize = torch.zeros(1, ).normal_(mean=.8, std=.3).clip(lower_bound, 1.)
            size_mult = randsize ** self.cut_pow
            size = int(min_size_width * (
                size_mult.clip(lower_bound, 1.)))  # replace .5 with a result for 224 the default large size is .95
            # size = int(min_size_width*torch.zeros(1,).normal_(mean=.9, std=.3).clip(lower_bound, .95)) # replace .5 with a result for 224 the default large size is .95

            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

        cutouts = torch.cat(cutouts, dim=0)
        cutouts = clamp_with_grad(cutouts, 0, 1)

        # if args.use_augs:
        cutouts = self.augs(cutouts)
        if self.noise_fac:
            facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(0, self.noise_fac)
            cutouts = cutouts + facs * torch.randn_like(cutouts)
        return cutouts


class MakeCutoutsHolywater(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow, augs):
        super().__init__()
        self.cut_size = cut_size
        tqdm.write(f'cut size: {self.cut_size}')
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
        self.augs = nn.Sequential(
            # K.RandomGaussianNoise(mean=0.0, std=0.5, p=0.1),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomSharpness(0.3, p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2, p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            K.RandomGrayscale(p=0.1),
        )

    def set_cut_pow(self, cut_pow):
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        cutouts_full = []
        noise_fac = 0.1
        min_size_width = min(sideX, sideY)
        lower_bound = float(self.cut_size / min_size_width)

        for ii in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            randsize = torch.zeros(1, ).normal_(mean=.8, std=.3).clip(lower_bound, 1.)
            size_mult = randsize ** self.cut_pow * ii + size
            size1 = int((min_size_width) * (
                size_mult.clip(lower_bound, 1.)))  # replace .5 with a result for 224 the default large size is .95
            size2 = int((min_size_width) * torch.zeros(1, ).normal_(mean=.9, std=.3).clip(lower_bound,
                                                                                          .95))  # replace .5 with a result for 224 the default large size is .95
            offsetx = torch.randint(0, sideX - size1 + 1, ())
            offsety = torch.randint(0, sideY - size2 + 1, ())
            cutout = input[:, :, offsety:offsety + size2 + ii, offsetx:offsetx + size1 + ii]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

        cutouts = torch.cat(cutouts, dim=0)
        cutouts = clamp_with_grad(cutouts, 0, 1)
        cutouts = self.augs(cutouts)
        facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(0, self.noise_fac)
        cutouts = cutouts + facs * torch.randn_like(cutouts)
        return cutouts


class MakeCutoutsOldHolywater(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow, augs):
        super().__init__()
        self.cut_size = cut_size
        tqdm.write(f'cut size: {self.cut_size}')
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
        self.augs = nn.Sequential(
            # K.RandomHorizontalFlip(p=0.5),
            # K.RandomSharpness(0.3,p=0.4),
            # K.RandomGaussianBlur((3,3),(10.5,10.5),p=0.2),
            # K.RandomGaussianNoise(p=0.5),
            # K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
            K.RandomAffine(degrees=180, translate=0.5, p=0.2, padding_mode='border'),
            K.RandomPerspective(0.6, p=0.9),
            K.ColorJitter(hue=0.03, saturation=0.01, p=0.1),
            K.RandomErasing((.1, .7), (.3, 1 / .4), same_on_batch=True, p=0.2), )

    def set_cut_pow(self, cut_pow):
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        cutouts_full = []
        noise_fac = 0.1

        min_size_width = min(sideX, sideY)
        lower_bound = float(self.cut_size / min_size_width)

        for ii in range(self.cutn):
            # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            randsize = torch.zeros(1, ).normal_(mean=.8, std=.3).clip(lower_bound, 1.)
            size_mult = randsize ** self.cut_pow
            size = int(min_size_width * (
                size_mult.clip(lower_bound, 1.)))  # replace .5 with a result for 224 the default large size is .95
            # size = int(min_size_width*torch.zeros(1,).normal_(mean=.9, std=.3).clip(lower_bound, .95)) # replace .5 with a result for 224 the default large size is .95

            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

        cutouts = torch.cat(cutouts, dim=0)
        cutouts = clamp_with_grad(cutouts, 0, 1)

        # if args.use_augs:
        cutouts = self.augs(cutouts)
        if self.noise_fac:
            facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(0, self.noise_fac)
            cutouts = cutouts + facs * torch.randn_like(cutouts)
        return cutouts


class MakeCutoutsGinger(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow, augs):
        super().__init__()
        self.cut_size = cut_size
        tqdm.write(f'cut size: {self.cut_size}')
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
        self.augs = augs
        '''
        nn.Sequential(
          K.RandomHorizontalFlip(p=0.5),
          K.RandomSharpness(0.3,p=0.4),
          K.RandomGaussianBlur((3,3),(10.5,10.5),p=0.2),
          K.RandomGaussianNoise(p=0.5),
          K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
          K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'), # padding_mode=2
          K.RandomPerspective(0.2,p=0.4, ),
          K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),)
'''

    def set_cut_pow(self, cut_pow):
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        cutouts_full = []
        noise_fac = 0.1

        min_size_width = min(sideX, sideY)
        lower_bound = float(self.cut_size / min_size_width)

        for ii in range(self.cutn):
            # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            randsize = torch.zeros(1, ).normal_(mean=.8, std=.3).clip(lower_bound, 1.)
            size_mult = randsize ** self.cut_pow
            size = int(min_size_width * (
                size_mult.clip(lower_bound, 1.)))  # replace .5 with a result for 224 the default large size is .95
            # size = int(min_size_width*torch.zeros(1,).normal_(mean=.9, std=.3).clip(lower_bound, .95)) # replace .5 with a result for 224 the default large size is .95

            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

        cutouts = torch.cat(cutouts, dim=0)
        cutouts = clamp_with_grad(cutouts, 0, 1)

        # if args.use_augs:
        cutouts = self.augs(cutouts)
        if self.noise_fac:
            facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(0, self.noise_fac)
            cutouts = cutouts + facs * torch.randn_like(cutouts)
        return cutouts


class MakeCutoutsZynth(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow, augs):
        super().__init__()
        self.cut_size = cut_size
        tqdm.write(f'cut size: {self.cut_size}')
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            K.RandomSharpness(0.3, p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2, p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7))

    def set_cut_pow(self, cut_pow):
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        cutouts_full = []
        noise_fac = 0.1

        min_size_width = min(sideX, sideY)
        lower_bound = float(self.cut_size / min_size_width)

        for ii in range(self.cutn):
            # size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            randsize = torch.zeros(1, ).normal_(mean=.8, std=.3).clip(lower_bound, 1.)
            size_mult = randsize ** self.cut_pow
            size = int(min_size_width * (
                size_mult.clip(lower_bound, 1.)))  # replace .5 with a result for 224 the default large size is .95
            # size = int(min_size_width*torch.zeros(1,).normal_(mean=.9, std=.3).clip(lower_bound, .95)) # replace .5 with a result for 224 the default large size is .95

            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

        cutouts = torch.cat(cutouts, dim=0)
        cutouts = clamp_with_grad(cutouts, 0, 1)

        # if args.use_augs:
        cutouts = self.augs(cutouts)
        if self.noise_fac:
            facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(0, self.noise_fac)
            cutouts = cutouts + facs * torch.randn_like(cutouts)
        return cutouts


class MakeCutoutsWyvern(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow, augs):
        super().__init__()
        self.cut_size = cut_size
        tqdm.write(f'cut size: {self.cut_size}')
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))
        self.augs = augs

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        return clamp_with_grad(torch.cat(cutouts, dim=0), 0, 1)


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        print(config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio) ** 0.5), round((area / ratio) ** 0.5)
    return image.resize(size, Image.LANCZOS)


class GaussianBlur2d(nn.Module):
    def __init__(self, sigma, window=0, mode='reflect', value=0):
        super().__init__()
        self.mode = mode
        self.value = value
        if not window:
            window = max(math.ceil((sigma * 6 + 1) / 2) * 2 - 1, 3)
        if sigma:
            kernel = torch.exp(-(torch.arange(window) - window // 2) ** 2 / 2 / sigma ** 2)
            kernel /= kernel.sum()
        else:
            kernel = torch.ones([1])
        self.register_buffer('kernel', kernel)

    def forward(self, input):
        n, c, h, w = input.shape
        input = input.view([n * c, 1, h, w])
        start_pad = (self.kernel.shape[0] - 1) // 2
        end_pad = self.kernel.shape[0] // 2
        input = F.pad(input, (start_pad, end_pad, start_pad, end_pad), self.mode, self.value)
        input = F.conv2d(input, self.kernel[None, None, None, :])
        input = F.conv2d(input, self.kernel[None, None, :, None])
        return input.view([n, c, h, w])


BUF_SIZE = 65536


def get_digest(path, alg=hashlib.sha256):
    hash = alg()
    print(path)
    with open(path, 'rb') as fp:
        while True:
            data = fp.read(BUF_SIZE)
            if not data: break
            hash.update(data)
    return b64encode(hash.digest()).decode('utf-8')


flavordict = {
    "cumin": MakeCutoutsCumin,
    "holywater": MakeCutoutsHolywater,
    "old_holywater": MakeCutoutsOldHolywater,
    "ginger": MakeCutoutsGinger,
    "zynth": MakeCutoutsZynth,
    "wyvern": MakeCutoutsWyvern,
    "aaron": MakeCutoutsAaron,
    "moth": MakeCutoutsMoth,
    "juu": MakeCutoutsJuu,
    "custom": MakeCutoutsCustom
}


@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


def gelu(x):
    return gelu_impl(x)


class MSEDecayLoss(nn.Module):
    def __init__(self, init_weight, mse_decay_rate, mse_epoches, mse_quantize):
        super().__init__()

        self.init_weight = init_weight
        self.has_init_image = False
        self.mse_decay = init_weight / mse_epoches if init_weight else 0
        self.mse_decay_rate = mse_decay_rate
        self.mse_weight = init_weight
        self.mse_epoches = mse_epoches
        self.mse_quantize = mse_quantize

    @torch.no_grad()
    def set_target(self, z_tensor, model):
        z_tensor = z_tensor.detach().clone()
        if self.mse_quantize:
            z_tensor = vector_quantize(z_tensor.movedim(1, 3), model.quantize.embedding.weight).movedim(3,
                                                                                                        1)  # z.average
        self.z_orig = z_tensor

    def forward(self, i, z):
        if self.is_active(i):
            return F.mse_loss(z, self.z_orig) * self.mse_weight / 2
        return 0

    def is_active(self, i):
        if not self.init_weight:
            return False
        if i <= self.mse_decay_rate and not self.has_init_image:
            return False
        return True

    @torch.no_grad()
    def step(self, i):

        if i % self.mse_decay_rate == 0 and i != 0 and i < self.mse_decay_rate * self.mse_epoches:

            if self.mse_weight - self.mse_decay > 0 and self.mse_weight - self.mse_decay >= self.mse_decay:
                self.mse_weight -= self.mse_decay
            else:
                self.mse_weight = 0
            print(f"updated mse weight: {self.mse_weight}")

            return True

        return False


class TVLoss(nn.Module):
    def forward(self, input):
        input = F.pad(input, (0, 1, 0, 1), 'replicate')
        x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
        y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
        diff = x_diff ** 2 + y_diff ** 2 + 1e-8
        return diff.mean(dim=1).sqrt().mean()


class MultiClipLoss(nn.Module):
    def __init__(self, clip_models, text_prompt, cutn, cut_pow=1., clip_weight=1.):
        super().__init__()

        # Load Clip
        self.perceptors = []
        for cm in clip_models:
            c = clip.load(cm[0], jit=False)[0].eval().requires_grad_(False).to(device)
            self.perceptors.append({'res': c.visual.input_resolution, 'perceptor': c, 'weight': cm[1], 'prompts': []})
        self.perceptors.sort(key=lambda e: e['res'], reverse=True)

        # Make Cutouts
        self.max_cut_size = self.perceptors[0]['res']
        # self.make_cuts = flavordict[flavor](self.max_cut_size, cutn, cut_pow)
        # cutouts = flavordict[flavor](self.max_cut_size, cutn, cut_pow=cut_pow, augs=args.augs)

        # Get Prompt Embedings
        # texts = [phrase.strip() for phrase in text_prompt.split("|")]
        # if text_prompt == ['']:
        #  texts = []
        texts = text_prompt
        self.pMs = []
        for prompt in texts:
            txt, weight, stop = parse_prompt(prompt)
            clip_token = clip.tokenize(txt).to(device)
            for p in self.perceptors:
                embed = p['perceptor'].encode_text(clip_token).float()
                embed_normed = F.normalize(embed.unsqueeze(0), dim=2)
                p['prompts'].append({'embed_normed': embed_normed, 'weight': torch.as_tensor(weight, device=device),
                                     'stop': torch.as_tensor(stop, device=device)})

        # Prep Augments
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomSharpness(0.3, p=0.1),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),  # padding_mode=2
            K.RandomPerspective(0.2, p=0.4, ),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            K.RandomGrayscale(p=0.15)
        )
        self.noise_fac = 0.1

        self.clip_weight = clip_weight

    def prepare_cuts(self, img):
        cutouts = self.make_cuts(img)
        cutouts = self.augs(cutouts)
        if self.noise_fac:
            facs = cutouts.new_empty([cutouts.shape[0], 1, 1, 1]).uniform_(0, self.noise_fac)
            cutouts = cutouts + facs * torch.randn_like(cutouts)
        cutouts = self.normalize(cutouts)
        return cutouts

    def forward(self, i, img):
        cutouts = checkpoint(self.prepare_cuts, img)
        loss = []

        current_cuts = cutouts
        currentres = self.max_cut_size
        for p in self.perceptors:
            if currentres != p['res']:
                current_cuts = resample(cutouts, (p['res'], p['res']))
                currentres = p['res']

            iii = p['perceptor'].encode_image(current_cuts).float()
            input_normed = F.normalize(iii.unsqueeze(1), dim=2)
            for prompt in p['prompts']:
                dists = input_normed.sub(prompt['embed_normed']).norm(dim=2).div(2).arcsin().pow(2).mul(2)
                dists = dists * prompt['weight'].sign()
                l = prompt['weight'].abs() * replace_grad(dists, torch.maximum(dists, prompt['stop'])).mean()
                loss.append(l * p['weight'])

        return loss


class ModelHost:
    def __init__(self, args):
        self.args = args
        self.model, self.perceptor = None, None
        self.make_cutouts = None
        self.alt_make_cutouts = None
        self.imageSize = None
        self.prompts = None
        self.opt = None
        self.normalize = None
        self.z, self.z_orig, self.z_min, self.z_max = None, None, None, None
        self.metadata = None
        self.mse_weight = 0
        self.normal_flip_optim = None
        self.usealtprompts = False

    def setup_metadata(self, seed):
        metadata = {k: v for k, v in vars(self.args).items()}
        del metadata['max_iterations']
        del metadata['display_freq']
        metadata['seed'] = seed
        if (metadata['init_image']):
            path = metadata['init_image']
            digest = get_digest(path)
            metadata['init_image'] = (path, digest)
        if (metadata['image_prompts']):
            prompts = []
            for prompt in metadata['image_prompts']:
                path = prompt
                digest = get_digest(path)
                prompts.append((path, digest))
            metadata['image_prompts'] = prompts
        self.metadata = metadata

    def setup_model(self, x):
        i = x
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        if self.args.prompts:
            print('Using prompts:', self.args.prompts)
        if self.args.altprompts:
            print('Using alternate augment set prompts:', self.args.altprompts)
        if self.args.image_prompts:
            print('Using image prompts:', self.args.image_prompts)
        if args.seed is None:
            seed = torch.seed()
        else:
            seed = args.seed
        torch.manual_seed(seed)
        print('Using seed:', seed)

        clear_memory()
        model = load_vqgan_model(get_path(f'{args.vqgan_model}.yaml'), get_path(f'{args.vqgan_model}.ckpt')).to(device)

        active_clips = bool(self.args.clip_model2) + bool(self.args.clip_model3) + bool(self.args.clip_model4) + bool(
            self.args.clip_model5) + bool(self.args.clip_model6)
        if active_clips != 0: clip_weight = round(1 / (active_clips + 1), 2)
        clip_models = [[self.args.clip_model, 1.0]]
        if self.args.clip_model2:
            clip_models = [[self.args.clip_model, clip_weight], [self.args.clip_model2, clip_weight]]
        if self.args.clip_model3:
            clip_models = [[self.args.clip_model, clip_weight], [self.args.clip_model2, clip_weight],
                           [self.args.clip_model3, clip_weight]]
        if self.args.clip_model4:
            clip_models = [[self.args.clip_model, clip_weight], [self.args.clip_model2, clip_weight],
                           [self.args.clip_model3, clip_weight], [self.args.clip_model4, clip_weight]]
        if self.args.clip_model5:
            clip_models = [[self.args.clip_model, clip_weight], [self.args.clip_model2, clip_weight],
                           [self.args.clip_model3, clip_weight], [self.args.clip_model4, clip_weight],
                           [self.args.clip_model5, clip_weight]]
        if self.args.clip_model6:
            clip_models = [[self.args.clip_model, clip_weight], [self.args.clip_model2, clip_weight],
                           [self.args.clip_model3, clip_weight], [self.args.clip_model4, clip_weight],
                           [self.args.clip_model5, clip_weight], [self.args.clip_model6, clip_weight]]
        print(clip_models)

        clip_loss = MultiClipLoss(clip_models, self.args.prompts, cutn=self.args.cutn)

        update_random(self.args.gen_seed, 'image generation')

        # [0].eval().requires_grad_(False)
        perceptor = clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
        # [0].eval().requires_grad_(True)

        cut_size = perceptor.visual.input_resolution

        if self.args.is_gumbel:
            e_dim = model.quantize.embedding_dim
        else:
            e_dim = model.quantize.e_dim

        f = 2 ** (model.decoder.num_resolutions - 1)

        make_cutouts = flavordict[self.args.flavor](cut_size, args.mse_cutn, cut_pow=args.mse_cut_pow, augs=args.augs)

        # make_cutouts = MakeCutouts(cut_size, args.mse_cutn, cut_pow=args.mse_cut_pow,augs=args.augs)
        if args.altprompts:
            self.usealtprompts = True
            self.alt_make_cutouts = flavordict[self.args.flavor](cut_size, args.mse_cutn, cut_pow=args.alt_mse_cut_pow,
                                                                 augs=args.altaugs)
            # self.alt_make_cutouts = MakeCutouts(cut_size, args.mse_cutn, cut_pow=args.alt_mse_cut_pow,augs=args.altaugs)

        if self.args.is_gumbel:
            n_toks = model.quantize.n_embed
        else:
            n_toks = model.quantize.n_e

        toksX, toksY = args.size[0] // f, args.size[1] // f
        sideX, sideY = toksX * f, toksY * f

        if self.args.is_gumbel:
            z_min = model.quantize.embed.weight.min(dim=0).values[None, :, None, None]
            z_max = model.quantize.embed.weight.max(dim=0).values[None, :, None, None]
        else:
            z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
            z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

        from PIL import Image
        import cv2
        # -------

        if self.args.init_image != "":
            img_0 = cv2.imread(self.args.init_image)
            z, *_ = model.encode(TF.to_tensor(img_0).to(device).unsqueeze(0) * 2 - 1)
        elif not os.path.isfile(get_path(f'out/steps/{i:04d}.png')):
            one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
            if self.args.is_gumbel:
                z = one_hot @ model.quantize.embed.weight
            else:
                z = one_hot @ model.quantize.embedding.weight
            z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
        else:
            if self.args.save_all_iterations:
                img_0 = cv2.imread(
                    get_path(f'out/steps/{i:04d}_{self.args.iterations_per_frame}.png')
                )
            else:
                # Hack to prevent colour inversion on every frame
                img_temp = cv2.imread(get_path(f'out/steps/{i}.png'))
                imageio.imwrite(get_path('out/inverted_temp.png'), img_temp)
                img_0 = cv2.imread(get_path('out/inverted_temp.png'))
            center = (1 * img_0.shape[1] // 2, 1 * img_0.shape[0] // 2)
            trans_mat = np.float32(
                [[1, 0, 10],
                 [0, 1, 10]]
            )
            rot_mat = cv2.getRotationMatrix2D(center, 10, 20)

            trans_mat = np.vstack([trans_mat, [0, 0, 1]])
            rot_mat = np.vstack([rot_mat, [0, 0, 1]])
            transformation_matrix = np.matmul(rot_mat, trans_mat)

            img_0 = cv2.warpPerspective(
                img_0,
                transformation_matrix,
                (img_0.shape[1], img_0.shape[0]),
                borderMode=cv2.BORDER_WRAP
            )
            z, *_ = model.encode(TF.to_tensor(img_0).to(device).unsqueeze(0) * 2 - 1)

            def save_output(i, img, suffix='zoomed'):
                filename = \
                    get_path(f"out/steps/{i:04}{'_' + suffix if suffix else ''}.png")
                imageio.imwrite(filename, np.array(img))

            save_output(i, img_0)
        # -------
        if args.init_image:
            pil_image = Image.open(args.init_image).convert('RGB')
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
        else:
            one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
            if self.args.is_gumbel:
                z = one_hot @ model.quantize.embed.weight
            else:
                z = one_hot @ model.quantize.embedding.weight
            z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
        z = EMATensor(z, args.ema_val)

        if args.mse_with_zeros and not args.init_image:
            z_orig = torch.zeros_like(z.tensor)
        else:
            z_orig = z.tensor.clone()
        z.requires_grad_(True)
        # opt = optim.AdamW(z.parameters(), lr=args.mse_step_size, weight_decay=0.00000000)
        if self.normal_flip_optim == True:
            if randint(1, 2) == 1:
                opt = torch.optim.AdamW(z.parameters(), lr=args.step_size, weight_decay=0.00000000)
                # opt = Ranger21(z.parameters(), lr=args.step_size, weight_decay=0.00000000)
            else:
                opt = optim.DiffGrad(z.parameters(), lr=args.step_size, weight_decay=0.00000000)
        else:
            opt = torch.optim.AdamW(z.parameters(), lr=args.step_size, weight_decay=0.00000000)

        self.cur_step_size = args.mse_step_size

        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])

        pMs = []
        altpMs = []

        for prompt in args.prompts:
            txt, weight, stop = parse_prompt(prompt)
            embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
            pMs.append(Prompt(embed, weight, stop).to(device))

        for prompt in args.altprompts:
            txt, weight, stop = parse_prompt(prompt)
            embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
            altpMs.append(Prompt(embed, weight, stop).to(device))

        from PIL import Image
        for prompt in args.image_prompts:
            path, weight, stop = parse_prompt(prompt)
            img = resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
            batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
            embed = perceptor.encode_image(normalize(batch)).float()
            pMs.append(Prompt(embed, weight, stop).to(device))

        for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
            pMs.append(Prompt(embed, weight).to(device))
            if (self.usealtprompts):
                altpMs.append(Prompt(embed, weight).to(device))

        self.model, self.perceptor = model, perceptor
        self.make_cutouts = make_cutouts
        self.imageSize = (sideX, sideY)
        self.prompts = pMs
        self.altprompts = altpMs
        self.opt = opt
        self.normalize = normalize
        self.z, self.z_orig, self.z_min, self.z_max = z, z_orig, z_min, z_max
        self.setup_metadata(seed)
        self.mse_weight = self.args.init_weight

    def synth(self, z):
        if self.args.is_gumbel:
            z_q = vector_quantize(z.movedim(1, 3), self.model.quantize.embed.weight).movedim(3, 1)
        else:
            z_q = vector_quantize(z.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
        return clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

    def add_metadata(self, path, i):
        imfile = PngImageFile(path)
        meta = PngInfo()
        step_meta = {'iterations': i}
        step_meta.update(self.metadata)
        # meta.add_itxt('vqgan-params', json.dumps(step_meta), zip=True)
        imfile.save(path, pnginfo=meta)
        # Hey you. This one's for Glooperpogger#7353 on Discord (Gloop has a gun), they are a nice snek

    @torch.no_grad()
    def checkin(self, i, losses, x):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        if i < args.mse_end:
            tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
        else:
            tqdm.write(f'i: {i - args.mse_end} ({i}), loss: {sum(losses).item():g}, losses: {losses_str}')
        tqdm.write(
            f'cutn: {self.make_cutouts.cutn}, cut_pow: {self.make_cutouts.cut_pow}, step_size: {self.cur_step_size}')
        out = self.synth(self.z.average)
        if i == self.args.max_iterations:
            batchpath = self.unique_index(self.args.folder_name)
            TF.to_pil_image(out[0].cpu()).save(batchpath)
        # TF.to_pil_image(out[0].cpu()).save('progress.png')
        # self.add_metadata('progress.png', i)
        # display.display(display.Image('progress.png'))
        if self.args.png == True:
            TF.to_pil_image(out[0].cpu()).save(get_path('out/png_progress.png'))
            self.add_metadata(get_path('out/png_progress.png'), i)
            TF.to_pil_image(out[0].cpu()).save(get_path('out/progress.png'))
            self.add_metadata(get_path('out/progress.png'), i)

            castle = Image.open(args.init_image).convert('RGB')
            # castle = Image.open('castle.png')
            castle = ImageEnhance.Brightness(castle)
            castle.enhance(100000).save(get_path('out/png_processing/brightness.png'))

            im = Image.open(get_path('out/png_processing/brightness.png'))
            im_invert = ImageOps.invert(im)
            im_invert.save(get_path('out/png_processing/work.png'))

            image = Image.open(get_path('out/png_processing/work.png')).convert('RGB')
            inverted_image = ImageOps.invert(image)
            inverted_image.save(get_path('out/png_processing/last.png'))

            im_rgb = Image.open(get_path('out/progress.png'))
            im_a = Image.open(get_path('out/png_processing/last.png')).convert('L').resize(im_rgb.size)
            im_rgb.putalpha(im_a)

            # im_rgb.save('/content/png_progress.png')
            im_rgb.save(get_path('out/png_processing/progress.png'))
            # display(Image.open('/content/png_progress.png').convert('RGB'))
            cv2.imshow('progress', cv2.imread(get_path('out/png_processing/progress.png'), -1))
            cv2.waitKey(1)

        else:
            TF.to_pil_image(out[0].cpu()).save(get_path('out/progress.png'))
            self.add_metadata(get_path('out/progress.png'), i)
            cv2.imshow('progress', cv2.imread(get_path('out/progress.png'), -1))
            cv2.waitKey(1)

    def unique_index(self, batchpath):
        i = 0
        while i < 10000:
            if os.path.isfile(batchpath + "/" + str(i) + ".png"):
                i = i + 1
            else:
                return batchpath + "/" + str(i) + ".png"

    def ascend_txt(self, i):
        out = self.synth(self.z.tensor)
        iii = self.perceptor.encode_image(self.normalize(self.make_cutouts(out))).float()

        result = []
        if self.args.init_weight and self.mse_weight > 0:
            result.append(F.mse_loss(self.z.tensor, self.z_orig) * self.mse_weight / 2)

        for prompt in self.prompts:
            result.append(prompt(iii))

        if self.usealtprompts:
            iii = self.perceptor.encode_image(self.normalize(self.alt_make_cutouts(out))).float()
            for prompt in self.altprompts:
                result.append(prompt(iii))

        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:, :, :]
        img = np.transpose(img, (1, 2, 0))
        im_path = get_path(f'out/steps/{i}.png')
        imageio.imwrite(im_path, np.array(img))
        self.add_metadata(im_path, i)
        return result

    def train(self, i, x):
        self.opt.zero_grad()
        mse_decay = self.args.mse_decay
        mse_decay_rate = self.args.mse_decay_rate

        lossAll = self.ascend_txt(i)

        if i < args.mse_end and i % args.mse_display_freq == 0:
            self.checkin(i, lossAll, x)
        if i == args.mse_end:
            self.checkin(i, lossAll, x)
        if i > args.mse_end and (i - args.mse_end) % args.display_freq == 0:
            self.checkin(i, lossAll, x)

        loss = sum(lossAll)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            if self.mse_weight > 0 and self.args.init_weight and i > 0 and i % mse_decay_rate == 0:
                if self.args.is_gumbel:
                    self.z_orig = vector_quantize(self.z.average.movedim(1, 3),
                                                  self.model.quantize.embed.weight).movedim(3, 1)
                else:
                    self.z_orig = vector_quantize(self.z.average.movedim(1, 3),
                                                  self.model.quantize.embedding.weight).movedim(3, 1)
                if self.mse_weight - mse_decay > 0:
                    self.mse_weight = self.mse_weight - mse_decay
                    print(f"updated mse weight: {self.mse_weight}")
                else:
                    self.mse_weight = 0
                    self.make_cutouts = flavordict[self.args.flavor](self.perceptor.visual.input_resolution, args.cutn,
                                                                     cut_pow=args.cut_pow, augs=args.augs)
                    if self.usealtprompts:
                        self.alt_make_cutouts = flavordict[self.args.flavor](self.perceptor.visual.input_resolution,
                                                                             args.cutn,
                                                                             cut_pow=args.alt_cut_pow,
                                                                             augs=args.altaugs)
                    self.z = EMATensor(self.z.average, args.ema_val)
                    self.new_step_size = args.step_size
                    self.opt = torch.optim.AdamW(self.z.parameters(), lr=args.step_size, weight_decay=0.00000000)
                    print(f"updated mse weight: {self.mse_weight}")
            if i > args.mse_end:
                if args.step_size != args.final_step_size and args.max_iterations > 0:
                    progress = (i - args.mse_end) / (args.max_iterations)
                    self.cur_step_size = lerp(self.args.step_size, self.args.final_step_size, progress)
                    for g in self.opt.param_groups:
                        g['lr'] = self.cur_step_size
            # self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

    def run(self, x):
        i = 0
        try:
            pbar = tqdm(range(int(args.max_iterations + args.mse_end)))
            while True:
                self.train(i, x)
                if i > 0 and i % args.mse_decay_rate == 0 and self.mse_weight > 0:
                    self.z = EMATensor(self.z.average, args.ema_val)
                    self.opt = torch.optim.AdamW(self.z.parameters(), lr=args.mse_step_size, weight_decay=0.00000000)
                    # self.opt = optim.Adgarad(self.z.parameters(), lr=args.mse_step_size, weight_decay=0.00000000)
                if i >= args.max_iterations + args.mse_end:
                    pbar.close()
                    break
                self.z.update()
                i += 1
                pbar.update()
        except KeyboardInterrupt:
            pass
        return i


def add_noise(img):
    # Getting the dimensions of the image
    row, col = img.shape

    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to white
        img[y_coord][x_coord] = 255

    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        # Pick a random y coordinate
        y_coord = random.randint(0, row - 1)

        # Pick a random x coordinate
        x_coord = random.randint(0, col - 1)

        # Color that pixel to black
        img[y_coord][x_coord] = 0

    return img


import io
import base64


def image_to_data_url(img, ext):
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=ext)
    img_byte_arr = img_byte_arr.getvalue()
    # ext = filename.split('.')[-1]
    prefix = f'data:image/{ext};base64,'
    return prefix + base64.b64encode(img_byte_arr).decode('utf-8')


def update_random(seed, purpose):
    if seed == -1:
        seed = random.seed()
        seed = random.randrange(1, 99999)

    print(f'Using seed {seed} for {purpose}')
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()


import torch
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def rand_perlin_2d(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim=-1) % 1
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0],
                                                                                                              0).repeat_interleave(
        d[1], 1)
    dot = lambda grad, shift: (
            torch.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                        dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d_octaves(desired_shape, octaves=1, persistence=0.5):
    shape = torch.tensor(desired_shape)
    shape = 2 ** torch.ceil(torch.log2(shape))
    shape = shape.type(torch.int)

    max_octaves = int(min(octaves, math.log(shape[0]) / math.log(2), math.log(shape[1]) / math.log(2)))
    res = torch.floor(shape / 2 ** max_octaves).type(torch.int)

    noise = torch.zeros(list(shape))
    frequency = 1
    amplitude = 1
    for _ in range(max_octaves):
        noise += amplitude * rand_perlin_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence

    return noise[:desired_shape[0], :desired_shape[1]]


def rand_perlin_rgb(desired_shape, amp=0.1, octaves=6):
    r = rand_perlin_2d_octaves(desired_shape, octaves)
    g = rand_perlin_2d_octaves(desired_shape, octaves)
    b = rand_perlin_2d_octaves(desired_shape, octaves)
    rgb = (torch.stack((r, g, b)) * amp + 1) * 0.5
    return rgb.unsqueeze(0).clip(0, 1).to(device)


def pyramid_noise_gen(shape, octaves=5, decay=1.):
    n, c, h, w = shape
    noise = torch.zeros([n, c, 1, 1])
    max_octaves = int(min(math.log(h) / math.log(2), math.log(w) / math.log(2)))
    if octaves is not None and 0 < octaves:
        max_octaves = min(octaves, max_octaves)
    for i in reversed(range(max_octaves)):
        h_cur, w_cur = h // 2 ** i, w // 2 ** i
        noise = F.interpolate(noise, (h_cur, w_cur), mode='bicubic', align_corners=False)
        noise += (torch.randn([n, c, h_cur, w_cur]) / max_octaves) * decay ** (max_octaves - (i + 1))
    return noise


def rand_z(model, toksX, toksY):
    e_dim = model.quantize.e_dim
    n_toks = model.quantize.n_e
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
    z = one_hot @ model.quantize.embedding.weight
    z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)

    return z


def make_rand_init(mode, model, perlin_octaves, perlin_weight, pyramid_octaves, pyramid_decay, toksX, toksY, f):
    if mode == 'VQGAN ZRand':
        return rand_z(model, toksX, toksY)
    elif mode == 'Perlin Noise':
        rand_init = rand_perlin_rgb((toksY * f, toksX * f), perlin_weight, perlin_octaves)
        z, *_ = model.encode(rand_init * 2 - 1)
        return z
    elif mode == 'Pyramid Noise':
        rand_init = pyramid_noise_gen((1, 3, toksY * f, toksX * f), pyramid_octaves, pyramid_decay).to(device)
        rand_init = (rand_init * 0.5 + 0.5).clip(0, 1)
        z, *_ = model.encode(rand_init * 2 - 1)
        return z
