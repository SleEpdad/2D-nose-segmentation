# coding=utf-8
from __future__ import print_function

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from dataset import Dataset
from pyheatmap.heatmap import HeatMap

def build_heatmap(Dataset):
    count=920
    rows=512
    cols=512

    imgs_predict = Dataset.load_predict_data()
    _,imgs_mask = Dataset.load_train_data()
    imgs_predict = np.squeeze(imgs_predict, axis=3)
    imgs_mask = np.squeeze(imgs_mask, axis=3)
    imgs_predict = imgs_predict * 255
    imgs_mask = imgs_mask * 255
    imgs_predict=imgs_predict[1,:,:]
    imgs_mask=imgs_mask[1,:,:]

    fig=plt.figure()
    ax1=fig.add_subplot(131)
    #imgs_mask=Image.fromarray(imgs_mask[1,:,:])
    im1=ax1.imshow(imgs_predict,cmap=plt.cm.hot)
    plt.colorbar(im1,shrink=0.5)

    ax2 = fig.add_subplot(133)
    im2=ax2.imshow(imgs_mask,cmap=plt.cm.hot)
    plt.colorbar(im2,shrink=0.5)
    plt.show()

    # for i in range(1,2):
    #     img_temp = imgs_predict[i, :, :]
    #     data = []
    #     for j in range(1,rows):
    #         for k in range(1,cols):
    #             if img_temp[j,k]>0.1:
    #                 a = [j,k]
    #                 data.append(a)
    #     hm = HeatMap(data)
    #     hm.heatmap(save_as= './heatmap/'+str(i)+"test.png")

if __name__ == '__main__':
    mydata=Dataset(512,512)
    build_heatmap(mydata)