import torch
import torch.nn.functional as F
import random
import torch
import torchvision

n = random.randint(1, 3)
C = random.randint(10, 20)
H = random.randint(5, 10)
W = random.randint(5, 10)
oH = random.randint(2, 4)
oW = random.randint(2, 4)
L = random.randint(2, 6)
input = torch.rand(n, C, H, W)
boxes = [torch.zeros(L, 4) for _ in range(n)]
for i in range(n):
    boxes[i][:, 0] = torch.rand(L) * (H - oH)  # y
    boxes[i][:, 1] = torch.rand(L) * (W - oW)  # x
    boxes[i][:, 2] = oH + torch.rand(L) * (H - oH)  # w
    boxes[i][:, 3] = oW + torch.rand(L) * (W - oW)  # h

    boxes[i][:, 2:] += boxes[i][:, :2]
    boxes[i][:, 2] = torch.clamp(boxes[i][:, 2], max=H - 1)
    boxes[i][:, 3] = torch.clamp(boxes[i][:, 3], max=W - 1)
output_size = (oH, oW)


def roi_pooling(input, boxes, output_size):
    n, C, H, W = input.shape
    oH, oW = output_size
    L = len(boxes[0])

    output = torch.zeros((n, L, C, oH, oW), dtype=torch.float32)

    for i in range(n):
        box_coords = boxes[i]
        box_coords = torch.round(box_coords[:])
        for j in range(L):
            y1, x1, y2, x2 = box_coords[j]

            # Calculate the height and width of each ROI
            roi_h = y2 - y1 + 1
            roi_w = x2 - x1 + 1

            # Calculate the height and width of each pooling bin
            bin_h = roi_h / oH
            bin_w = roi_w / oW

            for oh in range(oH):
                for ow in range(oW):
                    # Calculate the coordinates of the pooling bin in the original feature map
                    pool_y1 = torch.floor(y1 + oh * bin_h).to(torch.int)
                    pool_x1 = torch.floor(x1 + ow * bin_w).to(torch.int)
                    pool_y2 = torch.ceil(y1 + (oh + 1) * bin_h).to(torch.int)
                    pool_x2 = torch.ceil(x1 + (ow + 1) * bin_w).to(torch.int)

                    # Perform ROI pooling by taking the maximum value within the bin
                    # print(output[i, c, :, oh, ow].shape)
                    output[i, j, :, oh, ow] = torch.amax(input[i, :, pool_y1:pool_y2, pool_x1:pool_x2], dim=(1, 2))

    return output


out = roi_pooling(input, boxes, output_size)
