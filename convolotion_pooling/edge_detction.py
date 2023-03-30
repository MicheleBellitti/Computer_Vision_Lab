import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data

def visualize_contours(grad_x, grad_y, grad, gray):
    plt.subplot(2,2,1)
    plt.imshow(grad_x,cmap='gray')
    plt.title("Horizontal")
    plt.subplot(2,2,2)
    plt.imshow(grad_y,cmap='gray')
    plt.title("Vertical")
    plt.subplot(2,2,3)
    plt.imshow(grad, cmap='inferno')
    plt.title("Edges")
    plt.subplot(2,2,4)
    plt.imshow(gray)
    plt.title("Original")
    plt.waitforbuttonpress()

gray = data.camera()

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


ddepth = cv2.CV_16S

# Compute the gradient in x and y direction

grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

# Computing magnitude and direction of the gradient
magnitude = np.sqrt(np.power(grad_x, 2) + np.power(grad_y, 2))
direction = np.arctan2(grad_y , grad_x)

# Converting the image back to 8-bit format to visualize edges
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

visualize_contours(grad_x, grad_y, grad, gray)

