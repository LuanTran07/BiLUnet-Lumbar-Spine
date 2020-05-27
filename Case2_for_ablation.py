# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 19:57:34 2019

@author: LuanTran
"""

from keras.models import*
from keras.layers import*
from keras.optimizers import*
from keras.callbacks import ModelCheckpoint
import cv2
from data_lumbar import*
import matplotlib.pyplot as plt
import numpy as np

from keras.applications import *
from keras.applications.xception import preprocess_input
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects



from keras import regularizers
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
import cv2


from keras import applications
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from datagenerator import *
from keras.utils.vis_utils import plot_model
import lovasz_loss as lovasz
import keras




def conv_bn_act(inputs, n_filters=64, kernel=(2, 2), strides=1, activation= 'relu'):

    conv = Conv2D(n_filters, kernel_size= kernel, strides = strides, data_format='channels_last')(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation(activation)(conv)
    return conv

def ConvAndAct(x, n_filters, kernel=(1, 1), activation= 'relu', pooling=False):
    poolingLayer = AveragePooling2D(pool_size=(1, 1), padding='same')
    convLayer = Conv2D(filters=n_filters,kernel_size=kernel,strides=1)
    activation = Activation(activation)
    if pooling:
        x = poolingLayer(x)
    x = convLayer(x)
    x = activation(x)
    return x

def ConvAndBatch(x, n_filters=64, kernel=(2, 2), strides=(1, 1), padding='valid', activation= 'relu'):
    filters = n_filters
    conv_ = Conv2D(filters=filters,
                   kernel_size=kernel,
                   strides=strides,
                   padding=padding)
    batch_norm = BatchNormalization()
    activation = Activation(activation)
    x = conv_(x)
    x = batch_norm(x)
    x = activation(x)
    return x

def FeatureFusionModule(input_f, input_s, n_filters):
    concatenate = Concatenate(axis=-1)([input_f, input_s])

    branch0 = ConvAndBatch(concatenate, n_filters=n_filters, kernel=(3, 3), padding='same')
    branch_1 = ConvAndAct(branch0, n_filters=n_filters, pooling=True, activation= 'relu')
    branch_1 = ConvAndAct(branch_1, n_filters=n_filters, pooling=False, activation='sigmoid')

    x = multiply([branch0, branch_1])
    return Add()([branch0, x])



def swish(x):
    return (K.sigmoid(x) * x)

activation_method = swish
        
class myBiLunet(object):
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train, imgs_test

    
    def get_BiLunet(self):
        

            
        inputs = Input((self.img_rows, self.img_cols, 3))     
        
        # If you want to specify input tensor shape, e.g. 256x256 with 3 channels:
        vgg_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
        print([layer.name for layer in vgg_model.layers])
        layers = dict([(layer.name, layer) for layer in vgg_model.layers])
        
        SP1 = conv_bn_act(inputs, 32, strides=2)
        SP2 = conv_bn_act(SP1, 64, strides=2)
        SP3 = conv_bn_act(SP2, 156, strides=2) #156
        
        conv1l = Conv2D(32, 3, activation= activation_method, padding='same', kernel_initializer='he_normal')(inputs)
        conv1l = Conv2D(32, 3, activation= activation_method, padding='same', kernel_initializer='he_normal')(conv1l)
        pool1l = MaxPooling2D(pool_size=(2, 2),strides= 2)(conv1l)

        conv1 = Conv2D(64, 3, activation= activation_method, padding='same', kernel_initializer='he_normal')(pool1l)
        conv1 = Conv2D(64, 3, activation= activation_method, padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2),strides= 2)(conv1)

        conv2 = Conv2D(128, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2),strides= 2)(conv2)   

        conv3 = Conv2D(256, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(conv3)
        #conv3FFM = FeatureFusionModule(conv3, SP3, 4)  
        conv3FFM =concatenate([conv3, SP3], axis=3)
        print(conv3FFM)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        
        conv4 = Conv2D(512, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(conv4) 
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4) 


        conv5 = Conv2D(1024, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation=activation_method, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))      
        merge6 = concatenate([drop4,up6], axis=3)
  
        
        conv6 = Conv2D(512, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(conv6)
        up7 = Conv2D(256, 2, activation=activation_method, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3FFM, up7], axis=3)

        
        conv7 = Conv2D(256, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(conv7)
        up8 = Conv2D(128, 2, activation=activation_method, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))      
        merge8 = concatenate([conv2, up8], axis=3) 
         
        conv8 = Conv2D(128, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(conv8)
        up9 = Conv2D(64, 2, activation=activation_method, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)  

        conv8l = Conv2D(64, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(merge9)
        conv8l = Conv2D(64, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(conv8l)
        up9l = Conv2D(32, 2, activation=activation_method, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8l))

        merge9l = concatenate([conv1l, up9l], axis=3)  

        conv9 = Conv2D(32, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(merge9l)
        conv9 = Conv2D(32, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(conv9) 
        conv9 = Conv2D(5, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(conv9)    
        conv10 = Conv2D(5, 1, activation='softmax')(conv9)
        model = Model(input=inputs, output=conv10)
        
        for layer in model.layers[:19]:
            print('layer:\n',layer)
            layer.trainable = True
        model.load_weights('model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)

        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['acc']) # loss='sparse_categorical_crossentropy',
        model.summary()
        return model

    def train(self):
        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.get_BiLunet()
        print("got BiLunet")
        
        print("number layer myBiLunet")
        print(len(model.layers))
        
        model_checkpoint = ModelCheckpoint('BiLBiLunet_lumbar_NOF.hdf5', monitor='loss', verbose=1, save_best_only=True)
        
        print('Fitting model...')
        
        results = model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=60, verbose=1,
                  validation_split=0.05, shuffle=True, callbacks=[model_checkpoint])
        
        acc = results.history['acc']
        val_acc = results.history['val_acc']
        loss = results.history['loss']
        val_loss = results.history['val_loss']
         
        epochs = range(len(acc))
         
        plt.figure(figsize=(8,5))
        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("acc")
        plt.legend()       
        plt.savefig('train_acc.pdf')

        
        plt.figure(figsize=(8,5))
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
        plt.title('Training and validation loss')
        plt.xlabel("Epochs")
        plt.ylabel("log_loss")
        plt.legend()       
        plt.savefig('train_loss.pdf')


        
        
        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=3, verbose=1)
        np.save('./results/camvid_mask_test.npy', imgs_mask_test)
    
    def save_img(self):
        print("array to image")
        imgs = np.load('./results/lumbar_mask_test.npy')
        piclist = []
        for line in open("./results/lumbar.txt"):
            line = line.strip()
            picname = line.split('/')[-1]
            piclist.append(picname)
        for i in range(imgs.shape[0]):
            path = "./results/" + piclist[i]
            img = np.zeros((imgs.shape[1], imgs.shape[2], 3), dtype=np.uint8)
            for k in range(len(img)):
                for j in range(len(img[k])):  # cv2.imwrite也是BGR顺序
                    num = np.argmax(imgs[i][k][j])
                    if num == 0:
                        img[k][j] = [0, 0, 0] # 0 
                    elif num == 1:
                        img[k][j] = [0, 0, 255]   # 1 Vertebra
                    elif num == 2:
                        img[k][j] = [0, 255, 0]    # 2 Sacral
                    elif num == 3:
                        img[k][j] = [255, 0, 0]    #3   hip
                    elif num == 4:
                        img[k][j] = [150, 0, 0]
            img = cv2.resize(img,(2035, 3408), interpolation=cv2.INTER_CUBIC)
            print(path)
            cv2.imwrite(path, img)
            
if __name__ == '__main__':
    myBiLunet = myBiLunet()
    model = myBiLunet.get_BiLunet()    # model.summary()
    myBiLunet.train()
    myBiLunet.save_img()