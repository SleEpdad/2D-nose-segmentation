from __future__ import print_function

import os
import keras.models as models
from skimage.transform import resize
from skimage.io import imsave
import numpy as np

np.random.seed(1337)
import tensorflow as tf
tf.set_random_seed(1337)


from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, AveragePooling2D, ZeroPadding2D,BatchNormalization
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.utils.training_utils import multi_gpu_model
from keras.regularizers import l2
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from metrics import f1score
from keras.metrics import binary_accuracy


from dataset import Dataset

K.set_image_data_format('channels_last')

project_name = '2D_nose_segment'
img_rows = 512
img_cols = 512
smooth = 1


train_log=TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=1, write_graph=False, write_grads=False,
                            write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                            embeddings_metadata=None)

#?
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    bn_inputs = BatchNormalization(axis=-1)(inputs)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(bn_inputs)
    conc11 = concatenate([inputs, conv11], axis=3)

    conv12 = Conv2D(32, (3, 3), activation='relu', padding='same')(conc11)
    conc12 = concatenate([inputs, conv12], axis=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conc12)

    conv21 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conc21 = concatenate([pool1, conv21], axis=3)
    conv22 = Conv2D(64, (3, 3), activation='relu', padding='same')(conc21)
    conc22 = concatenate([pool1, conv22], axis=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conc22)

    conv31 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conc31 = concatenate([pool2, conv31], axis=3)
    conv32 = Conv2D(128, (3, 3), activation='relu', padding='same')(conc31)
    conc32 = concatenate([pool2, conv32], axis=3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conc32)

    conv41 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conc41 = concatenate([pool3, conv41], axis=3)
    conv42 = Conv2D(256, (3, 3), activation='relu', padding='same')(conc41)
    conc42 = concatenate([pool3, conv42], axis=3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conc42)

    conv51 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conc51 = concatenate([pool4, conv51], axis=3)
    conv52 = Conv2D(512, (3, 3), activation='relu', padding='same')(conc51)
    conc52 = concatenate([pool4, conv52], axis=3)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conc52), conc42], axis=3)
    conv61 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conc61 = concatenate([up6, conv61], axis=3)
    conv62 = Conv2D(256, (3, 3), activation='relu', padding='same')(conc61)
    conc62 = concatenate([up6, conv62], axis=3)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conc62), conv32], axis=3)
    conv71 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conc71 = concatenate([up7, conv71], axis=3)
    conv72 = Conv2D(128, (3, 3), activation='relu', padding='same')(conc71)
    conc72 = concatenate([up7, conv72], axis=3)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conc72), conv22], axis=3)
    conv81 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conc81 = concatenate([up8, conv81], axis=3)
    conv82 = Conv2D(64, (3, 3), activation='relu', padding='same')(conc81)
    conc82 = concatenate([up8, conv82], axis=3)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conc82), conv12], axis=3)
    conv91 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conc91 = concatenate([up9, conv91], axis=3)
    conv92 = Conv2D(32, (3, 3), activation='relu', padding='same')(conc91)
    conc92 = concatenate([up9, conv92], axis=3)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conc92)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.summary()
    #plot_model(model, to_file='model.png')

    model.compile(optimizer=Adam(decay=1e-6),
                  loss = 'binary_crossentropy',
                  metrics=['accuracy'])

    return model


def train():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)



    # mean = np.mean(imgs_train)  # mean for data centering
    # std = np.std(imgs_train)  # std for data normalization
    #
    # imgs_train -= mean
    # imgs_train /= std

    # imgs_train = imgs_train.astype(np.uint8)
    mydata = Dataset(512,512)
    imgs_train, imgs_mask_train = mydata.load_train_data()
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_train = imgs_train.astype('float32')
    #imgs_mask_train /= 255.  # scale masks to [0, 1]
    #imgs_train /= 255.  # scale masks to [0, 1]






    #print(imgs_mask_train[10, 11, 100, :, 0])
    #print(imgs_train[10, 11, 100, :, 0])

    # imgs_mask_train = imgs_mask_train.astype(np.uint8)

    # np.set_printoptions(threshold=np.nan)
    # print('-'*30)
    # print(imgs_train[0][30])

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    weight_dir = 'weights'
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    model_checkpoint = ModelCheckpoint(os.path.join(weight_dir, project_name + '.h5'), monitor='val_loss', save_best_only=True)

    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    csv_logger = CSVLogger(os.path.join(log_dir,  project_name + '.txt'), separator=',', append=False)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    # model.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=20, verbose=1, shuffle=True,
    #           validation_split=0.15,
    #           callbacks=[model_checkpoint])

    model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=50, verbose=1, shuffle=True, validation_split=0.10, callbacks=[model_checkpoint, csv_logger,train_log])


    print('-'*30)
    print('Training finished')
    print('-'*30)

def predict():


    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)

    imgs_test = Dataset.load_test_data()
    imgs_test = imgs_test.astype('float32')


    # # imgs_test = imgs_test.astype('float32')
    # imgs_test -= mean
    # imgs_test /= std
    # # imgs_test = imgs_test.astype(np.uint8)


    imgs_test /= 255.  # scale masks to [0, 1]


    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)

    model = get_unet()
    weight_dir = 'weights'
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    model.load_weights(os.path.join(weight_dir, project_name + '.h5'))

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)

    imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)

    npy_mask_dir = 'test_mask_npy'
    if not os.path.exists(npy_mask_dir):
        os.mkdir(npy_mask_dir)

    np.save(os.path.join(npy_mask_dir, project_name + '_mask.npy'), imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)

    imgs_mask_test = Dataset.preprocess_squeeze(imgs_mask_test)
    # imgs_mask_test /= 1.7
    imgs_mask_test = np.around(imgs_mask_test, decimals=0)
    imgs_mask_test = (imgs_mask_test*255.).astype(np.uint8)
    count_visualize = 1
    count_processed = 0
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    pred_dir = os.path.join('preds/', project_name)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, imgs_mask_test.shape[0]):
        for y in range(0, imgs_mask_test.shape[1]):
            if (count_visualize > 1) and (count_visualize < 16):
                imsave(os.path.join(pred_dir, 'pred_' + str(f"{count_processed:03}") + '.png'), imgs_mask_test[x][y])
                count_processed += 1

            count_visualize += 1
            if count_visualize == 17:
                count_visualize = 1
            if (count_processed % 100) == 0:
                print('Done: {0}/{1} test images'.format(count_processed, imgs_mask_test.shape[0]*14))

    print('-'*30)
    print('Prediction finished')
    print('-'*30)


if __name__ == '__main__':
    train()
    #predict()
