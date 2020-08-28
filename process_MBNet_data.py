
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob

import pandas as pd

import csv

num_data = 1100

class dataProcess(object):
    def __init__(self, out_rows, out_cols, npy_path="./npydata"):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.npy_path = npy_path

    def label2class(self, label):
        x = np.zeros([self.out_rows, self.out_cols, 5])
        for i in range(self.out_rows):
            for j in range(self.out_cols):
                x[i, j, int(label[i][j])] = 1  
        return x

    def create_train_data(self):
        imgdatas = np.ndarray((num_data, 512,512, 3), dtype=np.uint8)
        imglabels = np.ndarray((num_data, 512,512, 5), dtype=np.uint8)
        with open('data_2020.csv') as csvfile:
            readCSV=csv.reader(csvfile,delimiter=',') 
            i = 0
            for row in readCSV:
                img = load_img(row[0], grayscale=False, target_size=[512,512])
                label = load_img(row[1], grayscale=True, target_size=[512,512])
                img = img_to_array(img)
                label = self.label2class(img_to_array(label))
                imgdatas[i] = img
                imglabels[i] = label
                if i % 100 == 0:
                    print('Done: {0}/{1} images'.format(i, num_data))
                i += 1
                
        print('loading done')
        np.save(self.npy_path + '/lumbar_train.npy', imgdatas)
        np.save(self.npy_path + '/lumbar_mask_train.npy', imglabels)
        print('Saving to .npy files done.')

        
    def load_train_data(self):
        print('load train images...')
        imgs_train = np.load(self.npy_path + "/lumbar_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/lumbar_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        imgs_mask_train /= 255
        return imgs_train, imgs_mask_train
    
if __name__ == "__main__":
    mydata = dataProcess(512,512)
    mydata.create_train_data()

