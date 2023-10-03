#丸、四角、文字描画

import cv2
import numpy as np

img = cv2.imread("./golira.jpg")

#四角の描画
cv2.rectangle(img, (50,100), (100,200), (255,0,0), 2)
#円の描画
cv2.circle(img, (200,200), 100, (0,255,0), 2)
#文字の範囲
cv2.putText(img, 'Golira', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), thickness=2)

cv2.imshow("output", img)
cv2.waitKey(0)
print("finish.")