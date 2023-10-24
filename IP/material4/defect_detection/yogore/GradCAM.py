# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:15:49 2017

@author: hashimoto
"""

#----------------------
#追加モジュールの定義
#----------------------
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
import numpy as np
import cv2
import glob

def compute_saliency(model, img_path, layer_name,image_size):
    #-----------------
    #画像を読み込み、CNN入力形式に整える
    #-----------------
    test_img = []
    img = img_to_array(load_img(img_path, target_size=image_size))
    test_img.append(img)
    test_img = np.asarray(test_img)
    #入力データの正規化
    test_img = test_img / 255
    
    #----------------
    #CNNに入力し判定結果を取得
    #----------------
    #画像ファイルをモデルに入力し，その出力を得る．
    predictions = model.predict(test_img)[0]
    #判定結果を取得
    label = np.argmax(predictions)
    print("CNN's Result:" + str(label))
    
    #-----------------
    #CAM処理
    #-----------------
    #GradCam用にモデルをインスタンス化
    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(test_img)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]
    
    # 勾配を計算
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')

    guided_grads = gate_f * gate_r * grads

    # 重みを平均化して、レイヤーの出力に乗じる
    weights = np.mean(guided_grads, axis=(0, 1))
    cam = np.dot(output, weights)

    # 画像を元画像と同じ大きさにスケーリング
    cam = cv2.resize(cam, image_size, cv2.INTER_LINEAR)
    # ReLUの代わり
    cam  = np.maximum(cam, 0)
    # ヒートマップを計算
    heatmap = cam / cam.max()
    
    #-----------------
    #ヒートマップと元画像の合成処理
    #-----------------
    #元画像の読込
    img = cv2.imread(img_path)
    img = cv2.resize(img,image_size)
    #データ形式の変換
    rgb_img = cv2.convertScaleAbs(heatmap*255)
    #gradcamをカラーマップ変換
    color_map = cv2.applyColorMap(rgb_img,cv2.COLORMAP_JET)
    #元画像をgradcamをブレンディング
    blend = cv2.addWeighted(img, 0.5, color_map, 0.5, 0)
    
    return blend


#--------------------------
#学習済みモデルを読み込む
#--------------------------
model = load_model('./CNN_model_ave.h5')
model.summary()

#------------------------
#設定パラメータ
#------------------------
#可視化する画像ファイルを
#5, 7, 31
file = "./test_img/class-1/img_031.bmp"
img = cv2.imread(file)
cv2.imwrite("input-31.bmp",img)
#画像サイズ
image_size = (256,256)
#-----------------------------
#Grad-CAMの生成'conv2d_2'
#block5_conv3
#-----------------------------
output_img = compute_saliency(model, file,'block5_conv3',image_size)
print(img.shape)
output_img = cv2.resize(output_img,(img.shape[1],img.shape[0]))
cv2.imwrite("result-31.bmp",output_img)

   