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
img_size = [256,256,3]

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
        print(label)
        #画像ファイルのリストを取得
        img_list = glob.glob(folder_list[n] + "/*")
        #各画像データの読み込み
        for m in range(len(img_list)):
            #画像の読込(target_sizeでサイズの統一)
            img = img_to_array(load_img(img_list[m], target_size=(img_size[0],img_size[1])))
            
            if label == '1':
                #画像情報のアペンド(入力データの格納)
                x_data.append(img)
                #ラベル番号(フォルダ名を利用)のアペンド(出力データの格納)
                y_data.append(img_list[m])
    #データ形式を変更
    x_data = np.asarray(x_data)
    #y_data = np.asarray(y_data)
    #入力データの正規化
    x_data = x_data / 255
    #出力データのOne-hotベクトル化
    #y_data = to_categorical(y_data)
    return x_data, y_data


#------------------
#全データに対する評価
#------------------
x_train, y_train = get_data("./test_img")

print(x_train.shape)
#画像生成
reconstructions = vae.predict(x_train)

for n in range(len(x_train)):
    #正規化前の値に変換
    recon = reconstructions[n]*255
    #float型からuint型に変換
    recon = np.array(recon, dtype=np.uint8)
    
    #---------------
    #OpenCVで表示
    #---------------
    #Opencvの色の並びに変換(RGBからBGRに変換)
    recon = cv2.cvtColor(recon, cv2.COLOR_RGB2BGR)
    #サイズが小さいので拡大
    recon = cv2.resize(recon,(256,256))
    #表示
    cv2.imshow("output",recon)
    cv2.imwrite("output.png",recon)
    
    #入力画像も表示
    img = cv2.imread(y_train[n])
    img = cv2.resize(img,(256,256))
    cv2.imshow("input",img)
    cv2.imwrite("input.png",img)
    
    #-----------------
    #差分画像を取得
    #-----------------
    #グレースケールに変換
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    recon_gray = cv2.cvtColor(recon,cv2.COLOR_BGR2GRAY)
    #差分画像取得
    img_diff = cv2.absdiff(img_gray, recon_gray)
    #表示
    cv2.imshow("diff",img_diff)
    cv2.imwrite("diff.png",img_diff)
    cv2.waitKey(1)
    
    #----------------
    #差分の標準偏差と最大値を表示
    #----------------
    print(str(np.std(img_diff))+","+str(np.max(img_diff)))



#終了の確認
print("finish")

