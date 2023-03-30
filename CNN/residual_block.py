
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inplanes = inplanes  # input channels
        self.planes = planes  # output channels of convolution c1 and c2
        self.stride = stride  # stride value
        self.conv1 = nn.Conv2d(in_channels=self.inplanes, out_channels=self.planes, kernel_size=(3, 3),
                               stride=self.stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=self.planes, out_channels=self.planes, kernel_size=(3, 3),
                               stride=self.stride, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(self.planes)

    def forward(self, x):
        f = self.conv2(F.relu(self.conv1(x)))
        g = x
        if self.inplanes != self.planes or self.stride > 1:
            g = self.batch_norm(self.conv1(x))
        return F.relu(f + g)

