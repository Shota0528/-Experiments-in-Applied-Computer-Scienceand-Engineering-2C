# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 20:36:44 2018

@author: hashimoto
"""

#----------------------
#モジュールのインポート
#----------------------
import cv2
import numpy as np


cnt = 0
#カメラ操作情報を取得
cap = cv2.VideoCapture(0)
#繰り返し処理
while cap.isOpened():
    #カメラ情報から1フレーム分の画像情報を取得
    ret,img = cap.read()
    #取得失敗時の処理（強制終了）
    if ret == False:   break
     
    
    #ディスプレイ表示
    cv2.imshow("output",img)
    #終了条件("q"キーを押したら終了)
    k = cv2.waitKey(1)
    
    if k == ord('q'):
        break
    
    if k == ord('c'):
        cnt += 1
        save_name = "./save_img/" + "{:0=5}".format(cnt) + ".jpg"
        cv2.imwrite(save_name,img)
        
#動画情報の破棄
cap.release()
cv2.destroyAllWindows()

#終了の確認
print("finish")