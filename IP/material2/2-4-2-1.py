# モデル構造の定義

#モジュールのインポート
import cv2
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

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
