# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 20:36:44 2018

@author: hashimoto
"""

#----------------------
#モジュールのインポート
#----------------------
import cv2,glob,random
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import MaxPooling2D, Dropout, Activation, Conv2D, BatchNormalization, Flatten, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD

#----------
#モデル構造の定義
#----------
n_i = (256,256,3)     #入力層のノード数(CNNでは、縦、横、3)で設定
n_o = 2           #出力層のノード数=カテゴリ数

#----------
#VGG16モデルを読み込む
#全結合層は不要なため，include_top=False
#拡大縮小後の画像サイズは2分の1に設定
#----------
base_model=VGG16(weights='imagenet',include_top=False,
                 input_tensor=Input(shape=n_i))


#-----------
#全結合層を追加
#-----------
#CNNの出力を取得
x=base_model.output
#平均プーリングにかける
x=GlobalAveragePooling2D()(x)
#全結合層1にかける
x=Dense(512,activation='relu')(x)
x=Dense(1024,activation='relu')(x)
prediction = Dense(n_o,activation='softmax')(x)
model=Model(inputs=base_model.input,outputs=prediction)

#----------
#ファインチューニングする層を指定
#----------
#10層目（プーリング層を含む)までの重みは更新しない
for layer in base_model.layers[:11]:
    layer.trainable=False


#モデルのコンパイル(訓練課程の設定)
model.compile(optimizer=SGD(lr=0.001,momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


def get_data(f_name):
    #入力データを格納する配列
    x_data = []
    #出力データを格納する配列
    y_data = []
    #フォルダリストを取得
    folder_list = glob.glob(f_name + "/*")
    #各ラベルフォルダ内の画像データを読み込む
    for n in range(len(folder_list)):
        #フォルダアドレスを表示(確認用)
        print(folder_list[n])
        #フォルダ名を取得(class-で分割)
        sep = folder_list[n].split('class-')
        label = sep[1]
        #画像ファイルのリストを取得
        img_list = glob.glob(folder_list[n] + "/*")
        #各画像データの読み込み
        for m in range(len(img_list)):
            print(img_list[m])
            #画像の読込(target_sizeでサイズの統一)
            img = img_to_array(load_img(img_list[m], target_size=(256,256)))
            print(m,img_list[m])
            #画像情報のアペンド(入力データの格納)
            x_data.append(img)
            #ラベル番号(フォルダ名を利用)のアペンド(出力データの格納)
            y_data.append([label])
    #データ形式を変更
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    #入力データの正規化
    x_data = x_data / 255
    #出力データのOne-hotベクトル化
    y_data = to_categorical(y_data)
    return x_data, y_data


#--------------------
#学習データの取得
#--------------------
x_train, y_train = get_data("./train_img")

#--------------
#モデルの学習
#--------------
history = model.fit(x_train, y_train, batch_size=50,epochs=100)
#学習モデルの保存
model.save("CNN_model_ave.h5")

