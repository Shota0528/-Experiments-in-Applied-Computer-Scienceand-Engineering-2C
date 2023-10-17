# 学習データの読み込みと構造変換

#モジュールのインポート
import cv2
import numpy as np
import glob
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

# #モデル構造の定義
# n_i = int(42 * 42 * 3) #入力層のノード数
# n_o = 10 #出力層のノード数

# #入力層
# inputs = Input(shape=(n_i,))
# #中間層
# x = Dense(32, activation='relu')(inputs)
# x = Dense(64, activation='relu')(x)
# #出力層
# y = Dense(n_o, activation='softmax')(x)

# #モデルのインスタンス化
# model = Model(inputs=inputs, outputs=y)
# #モデルの最適化手法の定義
# model.compile(optimizer = 'rmsprop', 
#             loss = 'categorical_crossentropy', 
#             metrics = ['accuracy'])

# #構造の確認
# model.summary()
# #終了の確認
# print("finish")

# #読み込むファイルのアドレス指定
# f_name = "./goli.jpg"
# #1枚の画像を読み込む(target_sizeでサイズの統一)
# img = load_img(f_name, target_size = (42, 42))
# #PIL形式をnumpy形式に変換(float32型：学習はこの型)
# img = img_to_array(img)

# #読み込んだ画像をOpenCV, imshowで表示したい場合
# #float32型をuint8型に変換
# img = np.array(img, dtype='uint8')
# #RGBの順番をBGRの順番に変換
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# cv2.imshow("raw", img)
# cv2.waitKey(0)

# filelist = glob.glob("./train_img/class-1/*")
# print(filelist)

# #画像の入れ物
# train_input = []

# for n in range(len(filelist)):
#     #1枚の画像の読み込む(target_sizeでサイズの統一)
#     img = load_img(filelist[n], target_size = (42, 42))
#     #PIL形式をnumpy形式に変換(float32型 : 学習はこの型)
#     img = img_to_array(img)
    
#     #読み込む画像の格納
#     train_input.append(img)

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