# coding=utf-8
from __future__ import print_function

import numpy as np
import nibabel as nib
import os
import glob
import cv2

from PIL import ImageFile
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataagument(object):


    """
       用于增强图像的类。

       首先，仔细阅读训练图像和标签，然后将它们合并到下一个过程中。

       其次，用keras预处理来增强图像。

       最后，将图像分割成训练图像和标签。
    """
    def __init__(self):

        """
        Using glob to get all .img_type form path
        """
        self.img_type = "nii"
        self.train_path = "../NoseData/data/train/BBOX"
        self.label_path="../NoseData/data/train/GT"
        self.train_imgs = glob.glob(self.train_path + "/*/*." + self.img_type)
        self.label_imgs = glob.glob(self.label_path + "/*/*." + self.img_type)
        self.train_path = "../NoseData/data/train/BBOX"
        self.label_path = "../NoseData/data/train/GT"
        self.aug_train_path = "../NoseData/data/aug_train"
        self.aug_label_path = "../NoseData/data/aug_label"
        self.slices = len(self.train_imgs)
        self.datagen = ImageDataGenerator(
            rotation_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode='nearest')

    def set_augment(self):

        # 读入单通道的train和label, 分别转换成矩阵, 然后合并train和label的通道,做数据增强
        print("运行 Augmentation")
        """
        Start augmentation.....
        """
        trains = self.train_imgs
        labels = self.label_imgs
        path_train = self.train_path
        path_label = self.label_path
        print(len(trains), len(labels))

        if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
            print("trains can't match labels")
            return 0

        bbox_path = glob.glob(path_train+'/**/*.nii', recursive=True)
        bbox_path.sort()
        mask_path = glob.glob(path_label+'/**/*.nii', recursive=True)
        mask_path.sort()
        array_merge = np.ndarray((512,512,2),dtype=np.uint8)
        for i in range(len(trains)):
            img = nib.load(bbox_path[i])
            array_img = img.get_fdata()
            label = nib.load(mask_path[i])
            array_label = label.get_fdata()
            array_merge[:, :, 0]=array_img
            array_merge[:, :, 1]=array_label

            affine = np.diag([1, 1, 1, 1])

            array_merge_aug = array_merge.reshape(1,512,512,2)  # change shape to (1, 512, 512, 2) for put into function augment
            array_merge_aug = self.augment(array_merge_aug)  # data augmentation
            array_merge_aug = array_merge_aug.reshape(512, 512, 2)
            aug_train = array_merge_aug[:, :, 0]
            aug_label = array_merge_aug[:, :, 1]
            array_train_nib = nib.Nifti1Image(aug_train, affine)
            array_label_nib = nib.Nifti1Image(aug_label, affine)
            path_train_aug = self.aug_train_path  # train path after augmentation
            path_label_aug = self.aug_label_path  # label path after augmentation
            nib.save(array_train_nib, path_train_aug + '/train_aug' + str(i) + '.nii')
            nib.save(array_label_nib, path_label_aug + '/label_aug' + str(i) + '.nii')

    def augment(self, img, batch_size=1, imgnum=2):

        """
        增强一张图像
        """

        print("运行 aument")
        datagen = self.datagen
        i = 0
        for batch in datagen.flow(img,batch_size=batch_size,):
            i += 1
            if i > imgnum:
                break
        return batch

class Dataset(object):

    def __init__(self, out_rows, out_cols):

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.train_path='../NoseData/data/train/BBOX'
        self.label_path='../NoseData/data/train/GT'
        self.aug_merge_path = "../NoseData/data/aug_merge"
        self.aug_train_path = "../NoseData/data/aug_train"
        self.aug_label_path = "../NoseData/aug_label"
        self.test_path = "../NoseData/data/test"
        self.npy_path = "../NoseData/data/npydata"
        self.img_type = 'nii'

    def create_train_data(self):
        count = 920
        imgdatas = np.ndarray((count, self.out_rows, self.out_cols, 1), dtype=np.float32)
        imglabels = np.ndarray((count, self.out_rows, self.out_cols, 1), dtype=np.float32)
        bbox_path = glob.glob(self.train_path + '/*/*.nii', recursive=True)
        bbox_path.sort()
        mask_path = glob.glob(self.label_path + '/*/*.nii', recursive=True)
        mask_path.sort()
        # bbox_path=glob.glob(self.aug_train_path + '/*.nii',recursive=True)
        # bbox_path.sort()
        # mask_path=glob.glob(self.aug_label_path + '/*.nii',recursive=True)
        # mask_path.sort()
        for i in range(count):
            img = nib.load(bbox_path[i])
            img = img.get_fdata()
            label = nib.load(mask_path[i])
            label = label.get_fdata()
            img = self.preprocess(img)
            label = self.preprocess(label)
            imgdatas[i] = img
            imglabels[i] = label
            if i // 1000 == 0:
                print('Done: {0}/{1} images'.format(i, count))
        print('loading done', imgdatas.shape)
        np.save(self.npy_path + '/augimgs_train.npy', imgdatas)  # 将30张训练集和30张label生成npy数据
        np.save(self.npy_path + '/augimgs_mask_train.npy', imglabels)
        print('Saving to .npy train files done.')

    def create_test_data(self):
        count = 33
        imgdatas = np.ndarray((count, self.out_rows, self.out_cols, 1), dtype=np.uint8)
        bbox_path = glob.glob('./data/test/BBOX/**/*.nii', recursive=True)
        bbox_path.sort()
        for i in range(count):
            img = nib.load(bbox_path[i])
            img = img.get_fdata()
            img = self.preprocess(img)
            imgdatas[i] = img
            if i % 1000 == 0:
                print('Done: {0}/{1} images'.format(i, count))
        print('loading done', imgdatas.shape)
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)  # 将30张训练集和30张label生成npy数据
        print('Saving to .npy test files done.')

    def load_train_data(self):
        # 读入训练数据包
        # 括label_mask(npy格式), 归一化(只减去了均值)
        print('load train images...')
        #imgs_train = np.load(self.npy_path + "/augimgs_train.npy")
        #imgs_mask_train = np.load(self.npy_path + "/augimgs_mask_train.npy")
        imgs_train = np.load(self.npy_path + "/train_img.npy")
        imgs_mask_train = np.load(self.npy_path + "/train_label.npy")
        # imgs_train = imgs_train.astype('float32')
        # imgs_mask_train = imgs_mask_train.astype('float32')
        # imgs_train /= 255
        # mean = imgs_train.mean(axis=0)
        # imgs_train -= mean
        # imgs_mask_train /= 255
        # imgs_mask_train[imgs_mask_train > 0.5] = 1
        # imgs_mask_train[imgs_mask_train <= 0.5] = 0
        imgs_mask_train[imgs_mask_train==2]=1
        return imgs_train, imgs_mask_train

    def load_test_data(self):
        print('-' * 30)
        print('load test images...')
        print('-' * 30)
        imgs_test = np.load(self.npy_path + "/train_img.npy")
        imgs_test = imgs_test.astype('float32')
        # imgs_test /= 255
        # mean = imgs_test.mean(axis=0)
        # imgs_test -= mean
        return imgs_test

    def load_predict_data(self):
        print('-' * 30)
        print('load predict images...')
        print('-' * 30)
        imgs_predict=np.load(self.npy_path+'/2D_nose_segment_mask.npy')
        return imgs_predict

    def preprocess(self,imgs):
        imgs = np.expand_dims(imgs, axis=3)
        return imgs


if __name__ == "__main__":

    #aug = Dataagument()
    #aug.set_augment()
    #aug.splitMerge()
    mydata = Dataset(512, 512)
    mydata.create_train_data()
    #mydata.create_test_data()

    # check the shape of imgs_train, imgs_mask_tran and imgs_test

    #imgs_train, imgs_mask_train = mydata.load_train_data()
    #print(imgs_train.shape, imgs_mask_train.shape)
    #imgs_test = mydata.load_test_data()
    #print(imgs_test.shape)

