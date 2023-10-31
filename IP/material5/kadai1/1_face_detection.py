# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 20:36:44 2018

@author: hashimoto
"""
import cv2
import numpy as np

#モデル情報の読み込み
face_model = cv2.CascadeClassifier("./lib/haarcascade_frontalface_default.xml")
#Generation of camera object
cap = cv2.VideoCapture(0)
#repeat processing
while cap.isOpened():
    #obtaining image data
    ret,img = cap.read()
    #judgement
    if ret == False:
        break
    
    #グレースケール変換
    src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #顔検出
    faces = face_model.detectMultiScale(src_gray)
    #顔領域の座標格納用
    posix1 = posix2 = posiy1 = posiy2 = 0
    
    for x, y, w, h in faces:
        posix1 = x
        posix2 = x+w
        posiy1 = y
        posiy2 = y+h
        
        #検出した顔画像の抽出
        p_img = img.copy()
        p_img = p_img[y:y+h,x:x+w,:]
        #顔画像の表示
        cv2.imshow("face_img",p_img)
        #Opencvの色の並びをBGRからRGBに変換(CNN識別用に変換)
        p_img = cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB)
        #サイズ修正（CNNの入力サイズに合わせる）
        p_img = cv2.resize(p_img,(42,42))
        #-----------------------
        #p_imgに顔画像が格納されている。
        #以降にCNNによる識別コードを書いていく
        #-----------------------
        
        
        
        #顔領域を描画
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #Display
    cv2.imshow("raw",img)
    #break condition
    k = cv2.waitKey(1)
    #qキーでプログラム終了
    if k == ord('q'):
        break
         
cap.release()
cv2.destroyAllWindows()
print("finish")