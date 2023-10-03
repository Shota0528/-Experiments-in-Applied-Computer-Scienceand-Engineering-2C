#画素アクセスと色情報の変換

import cv2
import numpy as np

img = cv2.imread("./golira.jpg")
#画像サイズの確認
print(img.shape)

#縦サイズの取得
height = img.shape[0]
#横サイズの取得
width = img.shape[1]
#確認表示
print(height, width)


"""
#画素のアクセス
pixel = img[50][100]
print(pixel)
#色情報の変更
pixel = (0, 0, 255)
img[50][100] = pixel
"""

""" 
#範囲指定
for y in range(50, 100):
    for x in range(100, 200):
        img[y][x] = (0, 0, 255)
"""

#範囲指定(for無し=>高速化)
img[50:100, 100:200, :] = (0, 0, 255)


#画像の表示
cv2.imshow("output", img)
#キー入力待ち
cv2.waitKey(0)
#終了確認用
print("finish.")