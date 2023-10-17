#モジュールのインポート
import cv2,glob,random
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import MaxPooling2D, Dropout, Activation, Conv2D, BatchNormalization, Flatten
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.utils import to_categorical

#モデル構造の定義
n_i = (42, 42, 3)  #入力層のノード数=CNNでは, (縦, 横, 3)で設定
n_o = 10            #出力層のノード数=カテゴリ数

#モデル構造の定義
#入力層
inputs = Input(shape=n_i)

#Conv層(1層目)
x = Conv2D(filters=64, kernel_size=(5,5), padding='same')(inputs)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
#Conv層(2層目)
x = Conv2D(filters=128, kernel_size=(5,5), padding='same')(inputs)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
#Conv層(3層目)
x = Conv2D(filters=256, kernel_size=(5,5), padding='same')(inputs)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
#識別層
x = Flatten()(x)
x = Dense(32, activation='relu')(x)
x = Dense(64, activation='relu')(x)
#ドロップアウト層
x = Dropout(0.5)(x)
#出力層
y = Dense(n_o, activation='softmax')(x)

#モデルのインスタンス化
model = Model(inputs=inputs, outputs=y)
#モデルの最適化手法の定義
model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
#構造の確認
model.summary()

#画像データ群を読み込む関数
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
        label = int(sep[1])
        #画像ファイルのリストを取得
        img_list = glob.glob(folder_list[n] + "/*")
        #各画像データの読み込み
        for m in range(len(img_list)):
            #画像の読込(target_sizeでサイズの統一)
            img = img_to_array(load_img(img_list[m], target_size=(42,42)))
            #1次元配列に変換
            #img = img.flatten()
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

#学習データの取得
x_train, y_train = get_data("./train_img")

#モデルの学習
history = model.fit(x_train, y_train, batch_size=30, epochs=100)
#学習モデルの保存
model.save("train_CNN_pattern3.h5")