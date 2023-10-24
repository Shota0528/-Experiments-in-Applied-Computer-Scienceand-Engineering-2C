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
from sklearn.manifold import TSNE
from random import random

# 入力する画像サイズ
img_size = [28,28,3]

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
vae = load_model('vae_model.h5',
                 custom_objects={'Sampling': Sampling,'rounded_accuracy':rounded_accuracy})
vae.summary()


#---------------------
#学習データの取得用関数
#---------------------
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
            img = img_to_array(load_img(img_list[m], target_size=(img_size[0],img_size[1])))
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

#----------------------
#学習データの取得
#---------------------
x_train, y_train = get_data("./train_img")


#---------------------
#潜在ベクトルの取得
#--------------------
#エンコーダ層のみを取得
encoder_vae = vae.get_layer("encoder")
#学習データを入力し、潜在ベクトルを取得
_, _, latent_vec = encoder_vae.predict(x_train)


#------------------------
#グラフ描画と画像生成
#------------------------
#色の定義
colors = ["red", "green", "blue", "orange", "purple", "brown", "fuchsia", "grey", "olive", "lightblue","black"]
#デコーダ層のみを取得
decoder_vae = vae.get_layer("decoder")

#マウスイベントの関数(カーソルのある座標を取得し、その座標の画像を生成)
def onclick(event):
    
    #クリック字の座標をvae座標(潜在ベクトル)にする
    position = (event.xdata,event.ydata)
    #vae_decoderモデルの入力形式を作る
    n_position = []
    n_position.append(position)
    n_position = np.array(n_position)
    #デコーダで画像生成
    gen_img = decoder_vae.predict(n_position)[0]
    #生成画像をOpenCVで表示
    gen_img = cv2.cvtColor(gen_img,cv2.COLOR_RGB2BGR)
    gen_img = cv2.resize(gen_img,(84,84))
    cv2.imshow("Generated_img",gen_img)
    
    

#学習データをプロット
fig = plt.figure()
  
for n in range(len(latent_vec)):
    #カテゴリ番号の取得
    label = np.argmax(y_train[n])
    plt.scatter(latent_vec[n,0],latent_vec[n,1],s = 20, color=colors[label],marker="${}$".format(label),alpha=1.0)

#マウスイベントを追加
cid = fig.canvas.mpl_connect('motion_notify_event',onclick)
plt.show()



#終了の確認
print("finish")

