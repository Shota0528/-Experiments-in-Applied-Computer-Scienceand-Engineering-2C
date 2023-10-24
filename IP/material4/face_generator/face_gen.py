# -*- coding: utf-8 -*x-

"""
Created on Thu Aug 23 20:36:44 2018

@author: hashimoto
"""
#----------------------
#モジュールのインポート
#----------------------
import cv2, glob
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt

# 入力する画像サイズ
img_size = [128,128,3]

#----------------------
#学習時に独自で作成した関数等は、読み込めないため、ここで再定義する
#----------------------
K = keras.backend
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean 
    
#学習学習評価用として正解率を求める関数
def rounded_accuracy(y_true, y_pred):
    # 出力を最も近い整数に丸めて二値分類の正解率を求める
    return keras.metrics.binary_accuracy(tf.round(y_true), # 正解ラベル
                                         tf.round(y_pred)) # 予測値

#--------------------
#学習済みモデルを読み込む
#custom_objectsで独自のモジュールを読み込ませる
#--------------------
vae = load_model('face_vae.h5',
                 custom_objects={'Sampling': Sampling,'rounded_accuracy':rounded_accuracy})
vae.summary()

"""
#---------------------
#学習データの取得用関数
#---------------------
def get_data(f_name):
    #入力データを格納する配列
    x_data = []
    #画像ファイルのリストを取得
    img_list = glob.glob(folder_list[n] + "/*")
    #各画像データの読み込み
    for m in range(len(img_list)):
        #画像の読込(target_sizeでサイズの統一)
        img = img_to_array(load_img(img_list[m], target_size=(img_size[0],img_size[1])))
        #画像情報のアペンド(入力データの格納)
        x_data.append(img)
    #データ形式を変更
    x_data = np.asarray(x_data)
    #入力データの正規化
    x_data = x_data / 255
    
    return x_data
"""
#---------------------
#モデルによる画像生成
#<<1枚ずつ画像を生成する場合>>
#---------------------
#テストデータを入れるオブジェクト
test_data = []
#読み込むテストデータを設定
read_file = "./face_img/000033.jpg"
#画像の読み込み
img = img_to_array(load_img(read_file, target_size=(img_size[0],img_size[1])))
#アペンド
test_data.append(img)
#型変換
test_data = np.asarray(test_data)
#正規化
test_data = test_data / 255

#画像生成(配列のゼロ番目の情報のみ取得)
reconstructions = vae.predict(test_data)[0]

#---------------
#OpenCVで表示
#---------------
#Opencvの色の並びに変換(RGBからBGRに変換)
reconstructions = cv2.cvtColor(reconstructions, cv2.COLOR_RGB2BGR)
#サイズが小さいので拡大
reconstructions = cv2.resize(reconstructions,(176,176))
cv2.imshow("output",reconstructions)
cv2.waitKey(0)


#終了の確認
print("finish")

