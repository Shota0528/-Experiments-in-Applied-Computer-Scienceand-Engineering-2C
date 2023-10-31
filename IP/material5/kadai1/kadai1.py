import cv2
import numpy as np
import glob
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
from tensorflow.keras.utils import to_categorical

#--------------------
#学習済みモデルを読み込む
#--------------------
model_gender = load_model('./CNN_model_gender.h5')
model_gender.summary()
model_age = load_model('./CNN_model_age.h5')
model_age.summary()

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
        
        #--------------------
        #テスト画像を読み込み、形式を整える
        #--------------------
        #テストデータを入れるオブジェクト
        test_data = []
        #アペンド
        test_data.append(p_img)
        #型変換
        test_data = np.asarray(test_data)
        #正規化
        test_data = test_data / 255
        
        #-----------------
        #NNによる画像認識
        #-----------------
        #NNによる画像認識
        prediction_gender = model_gender.predict(test_data)
        prediction_age = model_age.predict(test_data)
        #NNの出力層からの出力を確認
        output_gender = prediction_gender[0]
        output_age = prediction_age[0]
        #確率最大のインデックス番号を識別結果として取得
        label_gender = np.argmax(output_gender)
        label_age = np.argmax(output_age)
        
        #文字の描画
        if label_gender == 0:
            cv2.putText(img,'man',(x,y-55),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),thickness=2)
        else:
            cv2.putText(img,'woman',(x,y-55),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),thickness=2)
        
        if label_age == 0:
            cv2.putText(img,'10-19 years old',(x,y-12),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),thickness=2)
        elif label_age == 1:
            cv2.putText(img,'20-29 years old',(x,y-12),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),thickness=2)
        elif label_age == 2:
            cv2.putText(img,'30-39 years old',(x,y-12),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),thickness=2)
        elif label_age == 3:
            cv2.putText(img,'40-49 years old',(x,y-12),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),thickness=2)
        elif label_age == 4:
            cv2.putText(img,'50-59 years old',(x,y-12),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),thickness=2)
        elif label_age == 5:
            cv2.putText(img,'60-110 years old',(x,y-12),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),thickness=2)
        
        #顔領域を描画
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
    #break condition
    #Display
    cv2.imshow("raw",img)
    
    k = cv2.waitKey(1)
    #qキーでプログラム終了
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("finish")