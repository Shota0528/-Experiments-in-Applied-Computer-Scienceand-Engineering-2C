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
from tensorflow.keras.layers import MaxPooling2D, Dropout, Activation, Conv2D, BatchNormalization, Flatten
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50



#----------
#モデル構造の定義
#----------
n_i = (42,42,3)     #入力層のノード数(CNNでは、縦、横、3)で設定
n_o = 4           #出力層のノード数=カテゴリ数
#-----------------
#モデル構造の定義
#-----------------
#ResNet50モデルを読み込む
#全結合層は不要なため，include_top=False
base_model=ResNet50(weights='imagenet',include_top=False,
                 input_tensor=Input(shape=n_i))
#-----------
#全結合層(識別層)を追加
#-----------
#Conv層の出力を取得
x=base_model.output
#一次元配列化
x = Flatten()(x)
#1層目
x=Dense(512,activation='relu')(x)
#2層目
x=Dense(1024,activation='relu')(x)
#ドロップアウト層
x = Dropout(0.5)(x)
#出力層
y = Dense(n_o, activation='softmax')(x)
#モデルのインスタンス化
model = Model(inputs=base_model.input, outputs=y)
#モデルの最適化手法の定義
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#構造の確認
model.summary()


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
            #画像の読込(target_sizeでサイズの統一)
            img = img_to_array(load_img(img_list[m], target_size=(42,42)))
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
history = model.fit(x_train, y_train, batch_size=30,epochs=100)
#学習モデルの保存
model.save("ResNet_model.h5")






