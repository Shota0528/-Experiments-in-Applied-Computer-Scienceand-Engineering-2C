#グレースケール変換

import cv2
import numpy as np

img = cv2.imread("./golira.jpg")

#グレースケール変換(BGR->GRAY)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Color", img)
cv2.imshow("GrayScale", gray_img)

cv2.waitKey(0)
print("finish.")