# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 19:58:48 2019

@author: LuanTran
"""


import cv2
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import io
import os.path as osp
import os
import IOU

from keras.models import *
from keras.layers import *

model = load_model('BiLUnet_lumbar_0211.hdf5') 
#model.load_weights('unet_lumbar.hdf5')



NUM_CLASSES = 5 #Number of classes
Classes = ["BackGround", "Vertebra","Sacral","Hip1","Hip2"] #List of classes

Union = np.float64(np.zeros(len(Classes))) #Sum of union
Intersection =  np.float64(np.zeros(len(Classes))) #Sum of Intersection

test_path = 'G:/BiLUnet_Lumbar/data_lumbar/test/'
label_path = 'G:/BiLUnet_Lumbar/data_lumbar/test_mask_IOU/'
dirs = os.listdir(test_path)
for file in dirs:
    # semantic segmentation
    test = cv2.imread(os.path.join(test_path , file)) 
    imgOri = cv2.resize(test, (512, 512)) 
    imgArr = np.array([imgOri])
    imgArr = imgArr.astype('float32')
    imgArr /= 255
    
    imgs = model.predict(imgArr, batch_size=5, verbose=1)
    
    for i in range(imgs.shape[0]):
        img = np.zeros((imgs.shape[1], imgs.shape[2], 1), dtype=np.uint8)
        for k in range(len(img)):
            for j in range(len(img[k])):  # cv2.imwrite也是BGR顺序
                num = np.argmax(imgs[i][k][j])
                if num == 0:
                    img[k][j] = [0] # 0 
                elif num == 1:
                    img[k][j] = [1]     # 1 Vertebra
                elif num == 2:
                    img[k][j] = [2]     # 2 Sacral
                elif num == 3:
                    img[k][j] = [3]    #3   hip
                elif num == 4:
                    img[k][j] = [4]
                       
    
    image = cv2.resize(img,(test.shape[1], test.shape[0]), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite( "G:/BiLUnet_Lumbar/eval_IOU/" + file[:-4]+'.png',image*50)
    
    
    labelGT= cv2.imread(label_path + file[:-4]+'.png',0)
    cv2.imwrite( "G:/BiLUnet_Lumbar/eval_IOU/" + file[:-4]+'labelGT.png',labelGT*50)
    print(labelGT.shape)
   
    CIOU,CU=IOU.GetIOU(image,labelGT,len(Classes),Classes) #Calculate intersection over union
    Intersection+=CIOU*CU
    Union+=CU

#-----------------------------------------Print results--------------------------------------------------------------------------------------
print("---------------------------Mean Prediction----------------------------------------")
print("---------------------IOU=Intersection Over Inion----------------------------------")
for i in range(len(Classes)):
    if Union[i]>0: print(Classes[i]+"\t"+str(Intersection[i]*100/Union[i]))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

