#顔検出とモザイク処理

import cv2
import numpy as np

#モデル情報の読み込み
face_model = cv2.CascadeClassifier("./lib/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while cap.isOpened:
    ret,img = cap.read()
    if ret == False: break
    
    #グレースケール変換
    src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #顔検出
    faces = face_model.detectMultiScale(src_gray)
    #描画
    for x, y, w, h in faces:
        #検出した顔画像の抽出
        p_img = img[y:y+h, x:x+w, :]
        #顔画像の表示
        cv2.imshow("face_img", p_img)
        #顔領域を描画
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow("output", img)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
print("finish.")