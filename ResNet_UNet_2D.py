from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Input, Lambda, MaxPooling2D, BatchNormalization, Activation, add
from keras import backend as K
from keras.layers.merge import concatenate
from keras.initializers import glorot_normal, glorot_uniform, he_normal, he_uniform

from ResNet import identity_block, conv_block
from data import dataProcess
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
init = he_normal(seed=1)


def side_out(x, factor):
    x = Conv2D(1, (1, 1), activation=None, padding='same')(x)

    kernel_size = (2*factor, 2*factor)
    x = Conv2DTranspose(1, kernel_size, strides=factor, padding='same',
                        use_bias=False, activation=None, kernel_initializer=init)(x)
    return x

class myUnet(object):
    def __init__(self, img_rows=120, img_cols=120):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train, imgs_test

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols,1))      # 320, 480, 3
        # Normalization
        # x = Lambda(lambda x: x / 255, name='pre-process')(inputs)
        x = Conv2D(16, (5, 5), strides=(1, 1), padding='same', name='conv1')(inputs)
        x = BatchNormalization(axis=-1, name='bn_conv1')(x)
        x = Activation('relu', name='act1')(x)  # 320, 480, 3
        #
        # Block 1
        c1 = conv_block(x, 3, (8, 8, 32), stage=1, block='a', strides=(1, 1))
        c1 = identity_block(c1, 3, (8, 8, 32), stage=1, block='b')      # 320, 480, 3
        # Block 2
        c2 = conv_block(c1, 3, (16, 16, 64), stage=2, block='a', strides=(2, 2))
        c2 = identity_block(c2, 3, (16, 16, 64), stage=2, block='b')    # 160, 240, 3
        # Block 3
        c3 = conv_block(c2, 3, (32, 32, 128), stage=3, block='a', strides=(2, 2))
        c3 = identity_block(c3, 3, (32, 32, 128), stage=3, block='b')   # 80, 120, 3
        # Block 4
        c4 = conv_block(c3, 3, (64, 64, 256), stage=4, block='a', strides=(2, 2))
        c4 = identity_block(c4, 3, (64, 64, 256), stage=4, block='b')   # 40, 60, 3
        s1=side_out(c4,8)
        # Block 6
        u5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='upconv_6', kernel_initializer=init)(c4)
        u5 = concatenate([u5, c3], name='concat_6')        # 40, 60, 3
        c5 = conv_block(u5, 3, (32, 32, 128), stage=6, block='a', strides=(1, 1))
        c5 = identity_block(c5, 3, (32, 32, 128), stage=6, block='b')
        s2 = side_out(c5, 4)
        # Block 7
        u6 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='upconv_7', kernel_initializer=init)(c5)
        u6 = concatenate([u6, c2], name='concat_7')        # 80, 120, 3
        c6 = conv_block(u6, 3, (16, 16, 64), stage=7, block='a', strides=(1, 1))
        c6 = identity_block(c6, 3, (16, 16, 64), stage=7, block='b')
        s3 = side_out(c6, 2)
        # Block 8
        u7 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', name='upconv_8', kernel_initializer=init)(c6)
        u7 = concatenate([u7, c1], name='concat_8')        # 160, 240, 3
        c7 = conv_block(u7, 3, (8, 8, 32), stage=8, block='a', strides=(1, 1))
        c7 = identity_block(c7, 3, (8, 8, 32), stage=8, block='b')
        s4 = side_out(c7, 1)

        # fuse
        fuse = concatenate(inputs=[s1, s2, s3, s4], axis=-1)
        fuse = Conv2D(1, (1, 1), padding='same', activation=None)(fuse)       # 320 480 1

        # outputs
        o1    = Activation('sigmoid', name='o1')(s1)
        o2    = Activation('sigmoid', name='o2')(s2)
        o3    = Activation('sigmoid', name='o3')(s3)
        o4    = Activation('sigmoid', name='o4')(s4)
        ofuse = Activation('sigmoid', name='ofuse')(fuse)

        #model = Model(inputs=[inputs], outputs=[o1, o2, o3, o4, o5, ofuse])
        model = Model(inputs=[inputs], outputs=ofuse)
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        print('model compile')
        return model
    def train(self):
        print("loading data")

        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.get_unet()
        print("got unet")

        # 保存的是模型和权重,
        model_checkpoint = ModelCheckpoint('unet5.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=10, verbose=1, shuffle=True,
                   callbacks=[model_checkpoint])

        print('predict test data')
        # imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        # np.save('imgs_mask_test.npy', imgs_mask_test)

    def test(self,m,n,q):
        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.get_unet()
        mydata = dataProcess(120,120)
        mydata.create_test_data(m,n,q)
        print("got unet")
        model.load_weights('./unet6.hdf5')
        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save('imgs_mask_test.npy', imgs_mask_test)



if __name__ == '__main__':
    myunet = myUnet()
   # myunet.train()

    myunet.test(4,2,120)
