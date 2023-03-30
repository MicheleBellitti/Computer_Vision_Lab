import random
import torch

n = random.randint(2, 6)
iC = random.randint(2, 6)
H = random.randint(10, 20)
W = random.randint(10, 20)
kH = random.randint(2, 5)
kW = random.randint(2, 5)
s = random.randint(2, 3)

input = torch.rand((n, iC, H, W), dtype=torch.float32)


def max_pool_2d(input_data, kH, kW, s):
    """
    Max pooling function that applies a 2D max pooling over an input tensor.

    Args:
        input (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        kH (int): The size of the sliding window height.
        kW (int): The size of the sliding window width.
        s (int): The stride of the sliding window.

    Returns:
        torch.Tensor: Max-pooled tensor of shape (batch_size, in_channels, out_height, out_width).
    """
    n, iC, H, W = input_data.shape
    out_h = (H - kH) // s + 1
    out_w = (W - kW) // s + 1

    # Initialize the output tensor with zeros.
    output = torch.zeros((n, iC, out_h, out_w), dtype=input_data.dtype)

    for i in range(out_h):
        for j in range(out_w):
            window = input_data[:, :, i * s:i * s + kH, j * s:j * s + kW]
            output[:, :, i, j] = torch.max(window).item()

    return output


out = max_pool_2d(input, kH, kW, s)

print(out)
