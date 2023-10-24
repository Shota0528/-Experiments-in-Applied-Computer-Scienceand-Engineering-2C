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

#---------------------
#モデルによる画像生成
#<<1枚ずつ画像を生成する場合>>
#---------------------
#テストデータを入れるオブジェクト
test_data = []
#読み込むテストデータを設定
read_file = "./test_img/class-2/5050.bmp"
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
reconstructions = cv2.resize(reconstructions,(48,48))
cv2.imshow("output",reconstructions)
cv2.waitKey(0)


"""
#---------------------
#モデルによる画像生成
#<<カテゴリー毎に画像を生成する場合>>
#---------------------
#テストデータの取得(全カテゴリの画像を1枚ずつ取得)
f_name = "./test_img"
#入力データを格納する配列
x_test = []
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
        x_test.append(img)
        break
#データ形式を変更
x_test = np.asarray(x_test)
#入力データの正規化
x_test = x_test / 255

#matplotで表示

def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")

def show_result(model, images=x_test, num_images=5):
    # 学習済みのモデルに入力して生成画像を取得
    reconstructions = model.predict(images[:num_images])
    fig = plt.figure(figsize=(num_images * 1.5, 3))
    for image_index in range(num_images):
        plt.subplot(2,num_images, 1 + image_index)
        plot_image(images[image_index])
        plt.subplot(2,num_images, 1 + num_images + image_index)
        plot_image(reconstructions[image_index])

#num_imagesにカテゴリ数を指定
show_result(vae,images=x_test,num_images=9)
plt.show()
"""


#終了の確認
print("finish")

