#レポート1-1

import cv2
import numpy as np

img = cv2.imread("./golira.jpg")

#四角の描画
cv2.rectangle(img, (25,10), (300,450), (255,0,0), 2)

#文字の範囲
cv2.putText(img, 'Golira', (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), thickness=2)

cv2.imwrite("report1-1.jpg", img)
cv2.imshow("report1-1", img)
cv2.waitKey(0)
print("finish.")