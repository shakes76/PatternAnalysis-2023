import PIL
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor
from utils import *

def upscale_image_using_upsample(img_path, upscale_factor=upscale_factor):
    img = PIL.Image.open(img_path).convert('YCbCr')
    y, cb, cr = img.split()

    # Convert Y channel to tensor and upscale using nn.functional.upsample
    input = ToTensor()(y).view(1, 1, y.size[1], y.size[0])
    out = F.upsample(input, scale_factor=upscale_factor, mode='bicubic', align_corners=True)
    out_img_y = ToPILImage()(out[0].detach().cpu())

    # Upscale cb and cr channels using PIL's bicubic interpolation
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)

    # Merge channels back to RGB image
    out_img = PIL.Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

    return out_img

result = upscale_image_using_upsample('./sample_path.jpg')
result.save('./output_upsampled.jpg')
