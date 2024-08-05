import cv2
import numpy as np

img = cv2.imread('./testimg/meow_400p.jpg', 0)

kernel = np.ones((5, 5), np.uint8)

dilation = cv2.dilate(img, kernel, iterations=2)

erosion = cv2.erode(img, kernel, iterations=2)

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)

closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)

cv2.imshow('Original', img)
cv2.imshow('Dilation', dilation)
cv2.imshow('Erosion', erosion)
cv2.imshow('Open', dilation)
cv2.imshow('Close', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
