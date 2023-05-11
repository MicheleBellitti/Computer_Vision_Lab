import cv2
import numpy as np
from matplotlib import pyplot as plt

im_a = cv2.imread("/Users/miche/PycharmProjects/Computer_Vision_Lab/GeometryTransform/gallery/img_1.jpeg")
im_b = cv2.imread("/Users/miche/PycharmProjects/Computer_Vision_Lab/GeometryTransform/gallery/img_2.jpeg")

# manually selected corresponding points
src_points = np.array([[131, 192], [137, 50], [339, 49], [335, 200]], dtype=np.float32)
dst_points = np.array([[180, 240], [192, 32], [317, 94], [310, 210]], dtype=np.float32)


# Estimate the homography matrix
homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)

# Warp image1 to image2 using the homography matrix
result = cv2.warpPerspective(im_b, homography, (im_a.shape[1], im_a.shape[0]))
result = cv2.addWeighted(im_a, 0.5, result, 0.5, 0)
# Display the result
print(result.shape)
# convert to (W,H,3)


plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.show()
