#モジュールのインポート
import cv2
import numpy as np
import glob
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import pylab
import itertools

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

#描画用の関数
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
        plt.text(j,i,cm[i,j], horizontalalignment="center",fontsize=15, color="white" if cm[i,j] > th else "black")
    
    plt.tight_layout()
    plt.xlabel('real category',fontsize=25)
    plt.ylabel('prediction category',fontsize=25)

#テスト画像を読み込み、形式を整える
x_test, y_test = get_data("./test_img")
#学習画像を読み込み、形式を整える
x_train, y_train = get_data("./train_img")

#モデル構造の定義
n_i = int(42 * 42 * 3) #入力層のノード数
n_o = 10 #出力層のノード数

#入力層
inputs = Input(shape=(n_i,))
#中間層
x = Dense(64, activation='relu')(inputs)
x = Dense(128, activation='relu')(x)
x = Dense(256, activation='relu')(x)
#出力層
y = Dense(n_o, activation='softmax')(x)

#モデルのインスタンス化
model = Model(inputs=inputs, outputs=y)
#モデルの最適化手法の定義
model.compile(optimizer = 'rmsprop', 
            loss = 'categorical_crossentropy', 
            metrics = ['accuracy'])

#構造の確認
model.summary()

#モデルの学習
history = model.fit(x_train, y_train, batch_size=50, epochs=200, validation_data=(x_test, y_test))
#学習モデルの保存
model.save("NN_model04.h5")

#NNによる画像認識
#NNによる画像認識
prediction = model.predict(x_test)
#識別結果のカテゴリ番号を取得
y_pred = np.argmax(prediction,axis=1)
#正解のカテゴリ番号を取得
y_real = np.argmax(y_test,axis=1)
#混合行列の算出
cm = confusion_matrix(y_pred,y_real)

classes = ['0','1','2','3','4','5','6','7','8','9']
plot_confusion_matrix(cm, classes=classes)
plt.savefig('./kadai1-pattern4.jpg')
plt.show()

#精度確認
scores = model.evaluate(x_test, y_test, verbose=0)
print("test Loss = %.4f" % scores[0])
print("test auc = %.4f" % scores[1])