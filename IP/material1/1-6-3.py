#動画ファイルとして保存

import cv2
import numpy as np

#カメラ操作情報を取得
cap = cv2.VideoCapture(0)
#カメラ画像サイズ、フレームレートを取得
fps = cap.get(cv2.get(cv2.CAP_PROP_FRAME_HEIGHT))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

#保存する動画ファイルの設定
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
save_name = "./save_movie.mp4"
video = cv2.VideoWriter(save_name, fourcc, fps, (width, height))

#繰り返し処理
while cap.isOpened():
    #カメラ情報から1フレーム分の画像情報を取得
    ret,img = cap.read()
    #取得失敗時の処理(強制終了)
    if ret == False: break
    
    #ディスプレイ表示
    cv2.imshow("output", img)
    
    #保存ファイルに書き込み
    video.write(img)
    
    #終了条件("qキーを押したら終了")
    k = cv2.waitKey(1)
    if k == ord('q'):
        break


#動画情報の破棄
cap.release()
#保存用オブジェクトの破棄
video.release()

print("finish.")