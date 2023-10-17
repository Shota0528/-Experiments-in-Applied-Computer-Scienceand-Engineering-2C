# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 19:37:11 2017

@author: hashimoto
"""

#------------------
#モジュールの追加
#------------------
import cv2
import numpy as np


def get_write():
    #描画キャンパスオブジェクトの生成
    campus = np.zeros((256,256),np.uint8)
    #キャンパスを白で塗りつぶす
    campus[:,:] = campus[:,:]
    #画像の表示
    cv2.imshow("campus",campus)
    #マウス情報の取得
    cv2.setMouseCallback("campus", mouse_event,campus)
    #frame
    frame = 0
    #描画時間の生成
    while (True):
        cv2.imshow("campus",campus)
        #1msecキー入力待ち
        k = cv2.waitKey(1) #1msec1待つ
        #入力がESCの場合
        if k == ord('q'):    #'q'ボタンは終了
            #ループを抜ける
            break
        elif k == ord('s'):  #'o'ボタンは認識結果出力
            #frameインクリメント
            frame += 1
            #画像の保存
            cv2.imwrite("./test_img_ori/img"+str(frame)+".bmp",campus)
            campus[:,:] = 0
            
    cv2.destroyAllWindows()


#マウスイベント
def mouse_event(event, x, y, flags, param):
    # 左クリックで座標値を取得
    if flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(param, (x, y), 1, 255, 10)




#手書き入力
get_write()



print("Finish")