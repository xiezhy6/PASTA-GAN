import os
import cv2
import numpy as np

canvas = np.ones((256,256,3)) * 128

# cv2.circle(canvas,(128,128),1,(256,0,0),2)

cv2.line(canvas, (20,20), (50,50), (256,0,0), 1)

cv2.imwrite('point.png', canvas)

# cv2.circle(upper_clothes_image_vis, (int(point[0]), int(point[1])), 1, color_list[ii], 2)
