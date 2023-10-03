import cv2
import numpy as np

#画像の読み込み
img1 = cv2.imread("./golira.jpg")
img2 = cv2.imread("./enjin.jpg")
img3 = cv2.imread("./goli.jpg")
#画像の表示
cv2.imshow("golira", img1)
cv2.imshow("enjin", img2)
cv2.imshow("goli", img3)
#キー入力待ち
cv2.waitKey(0)
#終了確認用
print("finish.")