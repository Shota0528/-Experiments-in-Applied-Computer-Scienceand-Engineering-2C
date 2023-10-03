#OpneCVを用いた画像ファイルの読み込みと表示

import cv2
import numpy as np

#画像の読み込み
img = cv2.imread("./golira.jpg")
#画像の表示
cv2.imshow("output", img)
#キー入力待ち
cv2.waitKey(0)
#終了確認用
print("finish.")