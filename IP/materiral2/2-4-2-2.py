# 学習データの読み込みと構造変換

#モジュールのインポート
import cv2
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

#モデル構造の定義
n_i = int(42 * 42 * 3) #入力層のノード数
n_o = 10 #出力層のノード数

#入力層
inputs = Input(shape=(n_i,))
#中間層
x = Dense(32, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
#出力層
y = Dense(n_o, activation='softmax')(x)

#モデルのインスタンス化
model = Model(inputs=inputs, outputs=y)
#モデルの最適化手法の定義
model.compile(optimizer = 'rmsprop', 
            loss = 'categorical_crossentropy', 
            metrics = ['accuracy'])

#構造の確認
model.summary()
#終了の確認
print("finish")

#読み込むファイルのアドレス指定
f_name = "./goli.jpg"
#1枚の画像を読み込む(target_sizeでサイズの統一)
img = load_img(f_name, target_size = (42, 42))
#PIL形式をnumpy形式に変換(float32型：学習はこの型)
img = img_to_array(img)

#読み込んだ画像をOpenCV, imshowで表示したい場合
#float32型をuint8型に変換
img = np.array(img, dtype='uint8')
#RGBの順番をBGRの順番に変換
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

cv2.imshow("raw", img)
cv2.waitKey(0)