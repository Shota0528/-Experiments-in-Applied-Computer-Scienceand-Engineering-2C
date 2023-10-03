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
        
        #以下モザイク処理
        height = p_img.shape[0]
        width = p_img.shape[1]
        scale_size = 20

        scale_h = int(height/scale_size)
        scale_w = int(width/scale_size)
        
        p_img2 = cv2.resize(p_img,(scale_w,scale_h))
        p_img2 = cv2.resize(p_img2,(width,height))
        
        #元の画像にモザイク後の顔を上書き
        img[y:y+h,x:x+w,:] = p_img2
        
        #モザイク後の顔画像の表示
        cv2.imshow("mosaic_face_img", p_img2)
        #モザイク後の顔領域を描画
        cv2.rectangle(p_img2, (x,y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow("output", img)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    if k == ord('c'):
        cv2.imwrite("report1-4.jpg", img)

cap.release()
print("finish.")