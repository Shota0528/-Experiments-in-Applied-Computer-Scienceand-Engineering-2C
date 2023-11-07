import datetime
import os
from functools import partial

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import (EarlyStopping, LearningRateScheduler,
                                        TensorBoard)
from tensorflow.keras.optimizers import SGD, Adam

from nets.yolo import get_train_model, yolo_body
from nets.yolo_training import get_lr_scheduler
from utils.callbacks import LossHistory, ModelCheckpoint, EvalCallback
from utils.dataloader import YoloDatasets
from utils.utils import get_anchors, get_classes, show_config
from utils.utils_fit import fit_one_epoch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == "__main__":
    
    
    
    
    
    #---------------------------------
    #model_dataフォルダに配置するファイル
    #---------------------------------
    #クラスラベルのリスト
    classes_path    = 'model_data/cls_classes.txt'
    #学習済みモデル（ファインチューニング用）
    model_path      = 'model_data/yolov7_weights.h5'
    #yoloアンカー
    anchors_path    = 'model_data/yolo_anchors.txt'
    
    #---------------------------
    #train_dataフォルダに配置するファイル
    #---------------------------
    #学習データ（リスト）の場所
    train_annotation_path   = './train_data/train.txt'
    val_annotation_path     = './train_data/train.txt'
    
    #----------------------------
    #以降、細かな設定（適宜調整）
    #----------------------------
    #画像サイズ
    input_shape = [640, 640]
    #Epoch数
    Epoch = 50
    #バッチサイズ
    batch_size = 2
    #保存先フォルダ名
    save_dir = 'trained_model'
    
    
    
    #以降、細かな設定（固定）
    train_gpu       = [0,]
    eager           = False
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    phi             = 'l'
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    label_smoothing     = 0
    Init_Epoch          = 0
    Freeze_Train        = True
    UnFreeze_Epoch      = 300
    Unfreeze_batch_size = 2
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4
    lr_decay_type       = 'cos'
    save_period         = 10
    eval_flag           = True
    eval_period         = 10
    num_workers         = 1
    
    

    

    #------------
    #GPU情報
    #------------
    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
    ngpus_per_node                      = len(train_gpu)
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if ngpus_per_node > 1 and ngpus_per_node > len(gpus):
        raise ValueError("The number of GPUs specified for training is more than the GPUs on the machine")    
    if ngpus_per_node > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = None
    print('Number of devices: {}'.format(ngpus_per_node))
    
    #-------------------
    #初期モデルの取得
    #-------------------
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)

    if ngpus_per_node > 1:
        with strategy.scope():
            
            model_body  = yolo_body((None, None, 3), anchors_mask, num_classes, phi, weight_decay)
            if model_path != '':
                print('Load weights {}.'.format(model_path))
                model_body.load_weights(model_path, by_name=True, skip_mismatch=True)
            if not eager:
                model = get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask, label_smoothing)
    else:
        model_body  = yolo_body((None, None, 3), anchors_mask, num_classes, phi, weight_decay)
        if model_path != '':
            
            print('Load weights {}.'.format(model_path))
            #ファインチューニングのモデルはここで読み込む
            model_body.load_weights(model_path, by_name=True, skip_mismatch=True)
        if not eager:
            model = get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask, label_smoothing)
    
    #-----------------------
    #学習データの情報を取得
    #-----------------------
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    """
    show_config(
        classes_path = classes_path, anchors_path = anchors_path, anchors_mask = anchors_mask, model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )
    """
    
    
    if True:
        #凍結（重み固定のレイヤーを指定）の設定
        if Freeze_Train:
            freeze_layers = {'n':118, 's': 118, 'm': 167, 'l': 216, 'x': 265}[phi]
            for i in range(freeze_layers): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))
            
        
        #batch_size  = Freeze_batch_size
        
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The number of train data is too small.")

        train_dataloader    = YoloDatasets(train_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, Init_Epoch, UnFreeze_Epoch, \
                                            mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
        val_dataloader      = YoloDatasets(val_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, Init_Epoch, UnFreeze_Epoch, \
                                            mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)

        optimizer = {
            'adam'  : Adam(lr = Init_lr, beta_1 = momentum),
            'sgd'   : SGD(lr = Init_lr, momentum = momentum, nesterov=True)
        }[optimizer_type]
        
        #----------------
        #学習(凍結)
        #----------------
        start_epoch = Init_Epoch
        end_epoch   = Epoch

        if ngpus_per_node > 1:
            with strategy.scope():
                model.compile(optimizer = optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        else:
            model.compile(optimizer = optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        logging         = TensorBoard(log_dir)
        loss_history    = LossHistory(log_dir)
        checkpoint      = ModelCheckpoint(os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), 
                                monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = save_period)
        checkpoint_last = ModelCheckpoint(os.path.join(save_dir, "last_epoch_weights.h5"), 
                                monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)
        checkpoint_best = ModelCheckpoint(os.path.join(save_dir, "best_epoch_weights.h5"), 
                                monitor = 'val_loss', save_weights_only = True, save_best_only = True, period = 1)
        early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
        lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
        eval_callback   = EvalCallback(model_body, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines, log_dir, \
                                        eval_flag=eval_flag, period=eval_period)
        callbacks       = [logging, loss_history, checkpoint, checkpoint_last, checkpoint_best, lr_scheduler, eval_callback]

        if start_epoch < end_epoch:
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
            model.fit(
                x                   = train_dataloader,
                steps_per_epoch     = epoch_step,
                validation_data     = val_dataloader,
                validation_steps    = epoch_step_val,
                epochs              = end_epoch,
                initial_epoch       = start_epoch,
                use_multiprocessing = True if num_workers > 1 else False,
                workers             = num_workers,
                callbacks           = callbacks
            )
        