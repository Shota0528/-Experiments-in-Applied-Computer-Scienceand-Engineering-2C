#レポート1-2

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

#カメラ画像サイズ、フレームレートを取得
fps = cap.get(cv2.CAP_PROP_FPS)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

#保存する動画ファイルの設定
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
save_name = "./report1-2.mp4"
video = cv2.VideoWriter(save_name,fourcc,fps,(width,height))

#フレーム番号
frame = 0

#繰り返し処理
while cap.isOpened(): 
    #カメラ情報から1フレーム分の画像情報を取得
    ret,img = cap.read()
    
    #文字の描画
    frame = frame + 1 
    cv2.putText(img, str(frame), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), thickness=2)
    
    #取得失敗時の処理(強制終了)
    if ret == False:
        break
    
    #ディスプレイ表示
    cv2.imshow("report1-2",img)
    
    #保存ファイルに書き込み
    video.write(img)
    
    #終了条件("q"キーを押したら終了)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    


cap.release()
video.release()
print("finish.")