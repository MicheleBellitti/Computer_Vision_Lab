# binary thresholding
import random
import numpy as np
from skimage import data
from skimage.transform import resize
import torch

im = data.camera()
im = resize(im, (im.shape[0] // 2, im.shape[1] // 2), mode='reflect', preserve_range=True, anti_aliasing=True).astype(
    np.uint8)
im = torch.from_numpy(im)
t = random.randint(0, 255)


def bin_threshold(p):
    if p < t:
        return 0
    else:
        return 255


out = im.clone()
out.apply_(bin_threshold)
