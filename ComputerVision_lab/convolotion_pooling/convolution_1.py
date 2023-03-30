import random
import torch

n = random.randint(2, 6)
iC = random.randint(2, 6)
oC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 6)
kW = random.randint(2, 6)

input = torch.rand(n, iC, H, W, dtype=torch.float32)
kernel = torch.rand(oC, iC, kH, kW, dtype=torch.float32)
out = torch.zeros((n, oC, H - kH + 1, W - kW + 1))

for i in range(H - kH + 1):
    for j in range(W - kW + 1):
        out[:, :, i, j] = (input[:, :, i:i + kH, j:j + kW].unsqueeze(1) * kernel).sum(dim=[-1, -2, -3])

