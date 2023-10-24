# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 20:36:44 2018

@author: hashimoto
"""

#----------------------
#モジュールのインポート
#----------------------
import cv2
import numpy as np
import glob
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import pylab
import itertools

def plot_confusion_matrix(cm, classes, normalize=False,title='confusion matrix',cmap=plt.cm.Oranges):
    plt.figure(figsize=(14,10))
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title,fontsize=25)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=15)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45,fontsize=15)
    plt.yticks(tick_marks,classes,fontsize=15)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]

    th = cm.max() / 2.0
    
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],
                 horizontalalignment="center",fontsize=15,
                 color="white" if cm[i,j] > th else "black")

    plt.tight_layout()
    plt.xlabel('real category',fontsize=25)  
    plt.ylabel('prediction category',fontsize=25)

#-------------------------
#学習データを取得する関数
#-------------------------
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
            img = img_to_array(load_img(img_list[m], target_size=(256,256)))
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
#学習済みモデルを読み込む
#--------------------
model = load_model('./CNN_model_ave.h5')
model.summary()

#--------------------
#テスト画像を読み込み、形式を整える
#--------------------
x_test, y_test = get_data("./test_img")

#-----------------
#NNによる画像認識
#-----------------
#NNによる画像認識
prediction = model.predict(x_test)
#識別結果のカテゴリ番号を取得
y_pred = np.argmax(prediction,axis=1)
#正解のカテゴリ番号を取得
y_real = np.argmax(y_test,axis=1)
#混合行列の算出
cm = confusion_matrix(y_pred,y_real)

#全体の精度確認
scores = model.evaluate(x_test, y_test, verbose=0)
print("test loss = %.4f" % scores[0])
print("test auc = %.4f" % scores[1])
#混合行列
classes = ['0','1']
plot_confusion_matrix(cm, classes=classes)
plt.savefig('./conf_matrix.jpg')
#plt.show()

print("finish.")