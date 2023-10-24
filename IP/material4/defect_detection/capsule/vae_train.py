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


#----------------------
#サンプリング層
#----------------------
K = keras.backend
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean 

#----------------------
#モデル構造の定義
#----------------------
# 潜在変数の次元数
latent_var_size = 64
# 入力する画像サイズ
img_size = [256,256,3]
# 出力画像の補正値
p_x = int(img_size[0]/4)
p_y = int(img_size[1]/4)

#<<エンコーダ層の定義>>
# 入力するテンソルの形状は(バッチサイズ, 28, 28)
inputs = keras.layers.Input(shape=img_size)
# Conv層
x = keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(inputs)
# Conv層
x = keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
# Flatten層
z = keras.layers.Flatten()(x)
# 全結合層:
z = keras.layers.Dense(128, activation="relu")(z)
# 平均値出力層
output_mean = keras.layers.Dense(latent_var_size)(z)
# 分散出力層
output_var = keras.layers.Dense(latent_var_size)(z)
# 潜在変数を作成
latent_var = Sampling()([output_mean, output_var])
# エンコーダーのモデルを作成
variational_encoder = keras.models.Model(inputs=[inputs], outputs=[output_mean, output_var, latent_var],name="encoder")
# エンコーダーのサマリを出力
variational_encoder.summary()

#<<デコーダ層の定義>>
# 入力するテンソルの形状は(バッチサイズ, 10)
decoder_inputs = keras.layers.Input(shape=[latent_var_size])
# 全結合層:  エンコーダでのFlatten前の次元数を指定
x = keras.layers.Dense(p_y * p_x * 64, activation="relu")(decoder_inputs)
#テンソルの形状を変換
x = keras.layers.Reshape((p_y, p_x, 64))(x)
# Deconv層
x = keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
# Deconv層
x = keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
# Conv層
outputs = keras.layers.Conv2D(3, 3, activation="sigmoid", padding="same")(x)
# デコーダーのモデルを作成
variational_decoder = keras.models.Model(inputs=[decoder_inputs], outputs=[outputs],name="decoder")
# デコーダーのサマリを出力
variational_decoder.summary()

#<<変分オートエンコーダー>>
# エンコーダーに画像を入力
_, _, latent_var = variational_encoder(inputs)
# デコーダーに潜在変数を入力
reconstructions = variational_decoder(latent_var)
# 変分オートエンコーダーのモデルを作成
vae = keras.models.Model(inputs=[inputs], outputs=[reconstructions])

#------------------
#損失関数、最適化の定義
#------------------
#画像の層画素数
pixels = img_size[0]*img_size[1]*img_size[2]*1.0
# 潜在ロスの計算
latent_loss = -0.5 * K.sum(1 + output_var - K.exp(output_var) - K.square(output_mean),axis=-1)
# 平均ロスを計算し、再構築ロスと同じスケールになるように784で割る
vae.add_loss(K.mean(latent_loss) / pixels)
#学習学習評価用として正解率を求める関数
def rounded_accuracy(y_true, y_pred):
    # 出力を最も近い整数に丸めて二値分類の正解率を求める
    return keras.metrics.binary_accuracy(tf.round(y_true), # 正解ラベル
                                         tf.round(y_pred)) # 予測値
# 変分オートエンコーダーのモデルをコンパイル
vae.compile(
    loss="binary_crossentropy", # バイナリクロスエントロピー誤差
    optimizer="rmsprop",        # オプティマイザーはRMSprop
    metrics=[rounded_accuracy]) # 二値分類の正解率を求める

# モデルのサマリを出力
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
            if label == '0':
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



#-------------------
#学習開始
#-------------------
history = vae.fit(
    x_train,
    x_train,
    epochs=500,      
    batch_size=50)

#--------------------
#モデルの保存
#--------------------
#学習モデルの保存
vae.save("vae_model.h5")


#終了の確認
print("finish")

