import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from predict_setup import YOLO

gpus = tf.config.list_physical_devices("GPU")
print("selected_gpu =" + str(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
    
if __name__ == "__main__":
    #モード選択："image"=画像ファイルに対する物体検出、"video"=動画ファイル及びWEBカメラに対する物体検出
    #        "video"の場合、video_path = "ファイルのアドレス"→動画ファイル、video_path = 0→webカメラ
    mode            = "video"
    
    #mode = "image"の場合の設定
    image_path      = "./test/street.jpg"
    
    #mode = "video"の場合の設定（動画ファイルならファイルアドレス、webカメらならカメラID(インカメラ:0））
    video_path      = 0
    
    #-------------
    #モデルの読み込み
    #-------------
    yolo = YOLO()
    
    #------------------
    #画像に対する物体検出
    #------------------
    if mode == "image":
        img = image_path
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
        else:
            #物体検出
            result = np.array(yolo.detect_image(image))
            #色情報の変換
            result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
            cv2.imshow("result",result)
            cv2.waitKey(0)
    #-------------------
    #動画ファイル、カメラに対する物体検出
    #-------------------        
    elif mode == "video":
        #カメラオブジェクトの取得
        capture = cv2.VideoCapture(video_path)
        #1フレーム目画像取得
        ref, frame = capture.read()
        if not ref:
            raise ValueError("Camera data can not readed.")

        while(True):
            #各フレームの画像取得
            ref, frame = capture.read()
            if not ref:
                break
            #色情報の変換
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            #形式変換
            frame = Image.fromarray(np.uint8(frame))
            #物体検出
            frame = np.array(yolo.detect_image(frame))
            #色情報の変換
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            #表示
            cv2.imshow("video",frame)
            c= cv2.waitKey(1)
            #終了条件
            if c==ord('q'):
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        cv2.destroyAllWindows()
    else:
        raise AssertionError("Please specify the correct mode: 'image', or 'video'")
    