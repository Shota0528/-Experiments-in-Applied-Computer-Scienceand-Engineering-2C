#画像の保存

import cv2
import numpy as np

img = cv2.imread("./golira.jpg")

height = img.shape[0]
width = img.shape[1]

#画像変換
img[int(height/2):height, int(width/2):width, :] = (0, 0, 255)
#画像の保存
cv2.imwrite("save_img.jpg", img)

cv2.imshow("output", img)
cv2.waitKey(0)
print("finish.")