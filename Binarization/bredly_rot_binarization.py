import numpy as np
import cv2
import matplotlib.pyplot as plt
from numba import njit

@njit (cache=True)
def Bradley_threshold(img, s=14, t = 0.12):
    s2 = s//2
    width, height = img.shape
    intImg, res = np.zeros((width, height)), np.zeros((width, height), dtype=np.uint8)
    for i in range(width):
        sumi = 0
        for j in range(height):            
            sumi += img[i, j]
            if i == 0:
                intImg[i, j] = sumi
            else:
                intImg[i, j] = intImg[i-1, j] + sumi


    for i in range(width):
        for j in range(height):
            x1 = i-s2 if i-s2 >= 1 else 1
            x2 = i+s2 if i+s2 < width else width-1
            y1 = j-s2 if j-s2 >= 1 else 1
            y2 = j+s2 if j+s2 < height else height-1

            count = (x2-x1)*(y2-y1)
            sumi = intImg[x2, y2] - intImg[x2, y1-1] - intImg[x1-1, y2] + intImg[x1-1, y1-1]
            if (img[i, j]*count) <= (sumi*(1.0-t)):
                if img[i, j] == 0:
                    res[i, j] = 0
                else:
                    res[i, j] = 255
            else:
                res[i, j] = 0

    return res
