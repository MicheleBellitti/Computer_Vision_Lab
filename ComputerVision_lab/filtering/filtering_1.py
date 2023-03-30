# linear point transformation
import random
import numpy as np
import torch
from skimage import data
from skimage.transform import resize

im = data.coffee()
im = resize(im, (im.shape[0] // 8, im.shape[1] // 8), mode='reflect', preserve_range=True, anti_aliasing=True).astype(
    np.uint8)
im = np.swapaxes(np.swapaxes(im, 0, 2), 1, 2)
im = torch.from_numpy(im)

a = random.uniform(0, 2)
b = random.uniform(-50, 50)


def func(p):
    p = round(p * a + b)
    if p > 255:
        p = 255
    if p < 0:
        p = 0
    return p


out = im.apply_(func)
