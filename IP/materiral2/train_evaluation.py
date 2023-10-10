#モジュールのインポート
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
from tensorflow.keras.utils import to_categorical

#モデル構造の定義
n_i = int(42 * 42 * 3) #入力層のノード数
n_o = 10 #出力層のノード数

#入力層
inputs = Input(shape=(n_i,))

#中間層
x = Dense(32, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)

#出力層
y = Dense(n_o, activation='softmax')(x)

#モデルのインスタンス化
model = Model(inputs=inputs, outputs=y)

#モデルの最適化手法の定義
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#構造の確認
model.summary()
#終了の確認
print("finish")

#学習データを取得する関数
def get_data(f_name):
    #学習データの入力(画像)の入れ物
    train_input = []
    #学習データの出力(カテゴリ番号)の入れ物
    train_output = []

    #フォルダリストを取得
    folder_list = glob.glob(f_name + "/*")
    print(folder_list)

    for c in range(len(folder_list)):
        #アドレス文字列の分割
        dev = folder_list[c].split("class-")
        print(dev)
        #カテゴリ番号の取得
        label = int(dev[1])
        
        #画像ファイルのリストを取得
        img_list = glob.glob(folder_list[c] + "/*")
        
        #各画素データの読み込み
        for n in range(len(img_list)):
            #画像の読み込み(target_sizeでサイズの統一)
            img = img_to_array(load_img(img_list[n], target_size=(42, 42)))
            #1次元配列に変換
            img = img.flatten()
            #画像情報のアペンド(入力データの格納)
            train_input.append(img)
            #カテゴリ番号のアペンド(出力データの格納)
            train_output.append([label])
            
    #データ形式を変更
    train_input = np.asarray(train_input)
    train_output = np.asarray(train_output)
    #入力データの正規化
    train_input = train_input / 255
    #出力データのOne-hotベクトル化
    train_output = to_categorical(train_output)
    return train_input, train_output

#学習データの取得
x_train, y_train = get_data("./train_img")

#テストデータを読み込む
x_test, y_test = get_data("./test_img")

#モデルの学習
history = model.fit(x_train, y_train, batch_size=30, epochs=100, validation_data=(x_test, y_test))
#学習モデルの保存
model.save("NN_model.h5")

#精度確認
scores = model.evaluate(x_test, y_test, verbose=0)
print("test loss = %.4f" % scores[0])
print("test auc = %.4f" % scores[1])
#精度変動の可視化処理
plt.figure(figsize = (14, 10))
plt.plot(history.history['accuracy'], 
        color = 'b', 
        linestyle = '-', 
        linewidth = 3, 
        path_effects = [path_effects.SimpleLineShadow(), path_effects.Normal()])
plt.plot(history.history['val_accuracy'], 
        color = 'r', 
        linestyle = '--', 
        linewidth = 3, 
        path_effects = [path_effects.SimpleLineShadow(), path_effects.Normal()])

plt.tick_params(labelsize=18)
plt.title('epochs-accuracy', fontsize=30,)
plt.ylabel('accuracy', fontsize=25)
plt.xlabel('epoch', fontsize=25)
plt.legend(['train', 'test'], loc='best', fontsize=25)
plt.savefig('./save_NN.jpg')
plt.show()