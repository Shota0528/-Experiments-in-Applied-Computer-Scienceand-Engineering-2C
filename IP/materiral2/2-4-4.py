#NNの学習フェーズ

#モジュールのインポート
import cv2
import numpy as np
import glob
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

#学習済みモデルを読み込む
model = load_model('./NN_model.h5')
model.summary()

#テスト画像を読み込み、形式を整える
#テストデータを入れるオブジェクト
test_data = []
#読み込むテストデータを設定
read_file = "./test_img/class-2/5050.bmp"
#画像の読み込み
img = img_to_array(load_img(read_file, target_size=(42, 42)))
#1次元配列に変換
img = img.flatten()
#アペンド
test_data.append(img)
#型変換
test_data = np.asarray(test_data)
#正規化
test_data = test_data / 255

#NNによる画像認識
#NNによる画像認識
prediction = model.predict(test_data)
#NNの出力層からの出力を確認
output = prediction[0]
print(output)
#確率最大のインデックス番号を識別結果として取得
label = np.argmax(output)
#識別結果の表示
print("read_img:" + str(read_file) + "\t" + "result:" + str(label) + "\n")
print("finish.")
