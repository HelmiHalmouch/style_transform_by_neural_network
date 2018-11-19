import numpy as np
import cv2 as cv
img = cv.imread('fig-3640553_960_720.jpg')

#res = cv.resize(img,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)

#OR
height, width = img.shape[:2]
res = cv.resize(img,(2*width, 1*height), interpolation = cv.INTER_CUBIC)
cv.imwrite('resized_image.jpg',res)