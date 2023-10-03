#画像サイズの変更

import cv2
import numpy as np

img = cv2.imread("./golira.jpg")

height = img.shape[0]
width = img.shape[1]

#変更後のサイズ
scale_h = int(height/2)
scale_w = int(width/2)
#画像サイズの変更
img2 = cv2.resize(img, (scale_w, scale_h))

cv2.imshow("output", img)
cv2.imshow("scale_output", img2)

cv2.waitKey(0)
print("finish.")