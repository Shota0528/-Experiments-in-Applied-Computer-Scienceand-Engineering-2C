import cv2
import numpy as np

#フィルタの定義
f = [[1/9, 1/9, 1/9], 
    [1/9, 1/9, 1/9], 
    [1/9, 1/9, 1/9]]

#画像の読み込み
img = cv2.imread("./signal.jpg", 0)

#フィルタリング処理後の画像
result = img.copy()
for y in range(1, img.shape[0] - 1):
    for x in range(1, img.shape[1] - 1):
        #畳み込み積分
        value = 0
        for d1 in range(-1, 2):
            for d2 in range(-1, 2):
                value += f[d1 + 1][d2 + 1] * img[y - d1][x - d2]
            
            #値の代入
            result[y][x] = int(value)

cv2.imshow("raw", img)
cv2.imshow("result", result)
cv2.waitKey(0)