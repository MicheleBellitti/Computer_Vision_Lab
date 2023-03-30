import random
import numpy as np
from skimage import data
from skimage.transform import resize
import torch

im = data.camera()
im = resize(im, (im.shape[0] // 2, im.shape[1] // 2), mode='reflect', preserve_range=True, anti_aliasing=True).astype(np.uint8)
im = torch.from_numpy(im)

# compute histogram
histo = torch.histc(im.float(), bins=256)
n_histo = histo / torch.sum(histo)
cumsum = torch.cumsum(n_histo, dim=0)
cummean = torch.cumsum(n_histo * torch.arange(0, 256), dim=0)
thresholds = torch.linspace(0, 1, 256)
q1 = torch.Tensor([sum(n_histo[i]for i in range(0, t+1)) for t in range(0, 256)])
q2 = torch.Tensor([sum(n_histo[i]for i in range(t+1, 256)) for t in range(0, 256)])
mean_1 = cummean / q1
mean_2 = (cummean[-1] - cummean) / q2
var_1 = [sum(((i - mean_1[t]) ** 2) * n_histo[i] / (q1[t]) for i in range(0, t+1)) for t in range(0, 256)]
var_2 = [sum(((i - mean_2[t]) ** 2) * n_histo[i] / (q2[t]) for i in range(t+1, 256)) for t in range(0, 256)]
variance = torch.Tensor([q1[t] * var_1[t] + q2[t] * var_2[t] for t in range(0, 256)])
out = torch.argmin(variance).item()
print(out)
