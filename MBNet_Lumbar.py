
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


from process_MBNet_data import*

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
from keras.callbacks import ReduceLROnPlateau
from keras import applications
from keras.layers import Input
from keras.utils.vis_utils import plot_model
import keras
from keras.layers.core import Activation, Reshape
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.utils import plot_model


import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
import numpy as ny
import pandas


from sklearn import preprocessing
#conda install graphviz
from keras.utils.vis_utils import plot_model
#from sklearn.externals import joblib
import joblib

mm_scaler = preprocessing.MinMaxScaler()

def swish(x):
    return (K.sigmoid(x) * x)
activation_method = swish

    
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


def semantic_segmentation_branch(inputs):

    SP1 = conv_bn_act(inputs, 32, strides=2)
    SP2 = conv_bn_act(SP1, 64, strides=2)
    SP3 = conv_bn_act(SP2, 156, strides=2) 
        
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
    conv3FFM = FeatureFusionModule(conv3, SP3, 4)  
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
    out_SSB = Conv2D(5, 1, activation='softmax', name="seg_output")(conv9)
    
    return out_SSB


def inspected_values_branch(in_layer):
    
     conv1 = Conv2D(32,3, activation= activation_method, padding='same', kernel_initializer='he_normal')(in_layer)
     pool1M = MaxPooling2D(pool_size=(4, 4),strides= 4)(conv1)
        
     conv2 = Conv2D(128,3, activation= activation_method, padding='same', kernel_initializer='he_normal')(pool1M)           
     pool12M = MaxPooling2D(pool_size=(4, 4),strides= 4)(conv2)
                 
     LayerF = Flatten()(pool12M)  
     D1 = Dense(1024)(LayerF)
     D1 = Activation('relu')(D1)

     D2 = Dense(1024)(D1)
     D2 = Activation('relu')(D2)
     Do = Dense(18)(D2)
     
     out_IVB = Activation('sigmoid', name="number_output")(Do)
     return out_IVB


        
class myMBNet(object):
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()

        
        Y = ny.empty([0,18])
        with open('data_2020.csv') as csvfile:
            readCSV=csv.reader(csvfile,delimiter=',')
             
            for row in readCSV:
                Y = ny.vstack([Y,[row[2], row[3],row[4],
                          row[5],row[6], row[7],row[8],
                          row[9],row[10], row[11],row[12],
                          row[13],row[14], row[15],row[16],row[17],row[18],row[19]]])
        
        Y_train = mm_scaler.fit_transform(Y)
        
        #print(Y_train)
      
        joblib.dump(mm_scaler, 'scaler2.gz')        
        return imgs_train, imgs_mask_train, Y_train
    

    
    def get_MBNet(self):
        
        inputs = Input((self.img_rows, self.img_cols, 3))     
        out_SSB = semantic_segmentation_branch(inputs)
        out_IVB = inspected_values_branch(out_SSB)
        model = Model(inputs= inputs, outputs= [out_IVB, out_SSB])
        
        
        losses = { "number_output": "mse", "seg_output": "categorical_crossentropy"}
        lossWeights = {"number_output": 0.1, "seg_output": 0.9}
        model.compile(optimizer=Adam(lr=1e-4), loss=losses, loss_weights=lossWeights, metrics=["accuracy"])
        
        #model.summary()
        plot_model(model, to_file='model_plot.pdf', show_shapes=True, show_layer_names=True)
        return model

    def train(self):
        print("loading data")
        imgs_train, imgs_mask_train, Y_train = self.load_data()
        print("loading data done")
        model = self.get_MBNet()
        print("got MBNet")    
        print("number layer myMBNet")
        print(len(model.layers))
       
        
        
        model_checkpoint = ModelCheckpoint('MBNET1000_V3.hdf5', monitor= 'val_loss', verbose=1, mode= 'auto', save_best_only= False)
        print('Fitting model...')   
        results = model.fit(imgs_train, {"number_output": Y_train, "seg_output": imgs_mask_train}, validation_split=0.1, shuffle=True, callbacks=[model_checkpoint], epochs=200, batch_size=4, verbose=1)
  
        acc_seg_output = results.history['seg_output_accuracy']
        val_seg_output = results.history['val_seg_output_accuracy']
        acc_number_output = results.history['number_output_accuracy']
        val_number_output = results.history['val_number_output_accuracy']
        
        loss = results.history['loss']
        val_loss = results.history['val_loss']
        loss_seg_output = results.history['seg_output_loss'] 
        val_seg_output_loss = results.history['val_seg_output_loss']
        loss_number_output = results.history['number_output_loss']
        val_number_output_loss = results.history['val_number_output_loss']
         
        epochs = range(len(acc_seg_output))
         
        plt.figure(figsize=(7,5))
        plt.plot(epochs, acc_seg_output, 'b', label='Training acc_seg_output')
        plt.plot(epochs, val_seg_output, 'r', label='Validation acc_seg_output')
        plt.title('Training and validation accuracy seg_output')
        plt.xlabel("Epochs")
        plt.ylabel("acc")
        plt.legend()       
        plt.savefig('train_acc_seg_output.pdf')
        
        
        
        plt.figure(figsize=(7,5))
        plt.plot(epochs, acc_number_output, 'b', label='Training acc_number_output')
        plt.plot(epochs, val_number_output, 'r', label='Validation acc_number_output')
        plt.title('Training and validation accuracy number_output')
        plt.xlabel("Epochs")
        plt.ylabel("acc")
        plt.legend()       
        plt.savefig('train_acc_number_output.pdf')
        
        plt.figure(figsize=(7,5))
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel("Epochs")
        plt.ylabel("log_loss")
        plt.legend()       
        plt.savefig('train_loss.pdf')
        
        
        plt.figure(figsize=(7,5))
        plt.plot(epochs, loss_seg_output, 'b', label='Training loss')
        plt.plot(epochs, val_seg_output_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel("Epochs")
        plt.ylabel("log_loss")
        plt.legend()       
        plt.savefig('train_loss_seg.pdf')


        plt.figure(figsize=(7,5))
        plt.plot(epochs, loss_number_output, 'b', label='Training loss')
        plt.plot(epochs, val_number_output_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel("Epochs")
        plt.ylabel("log_loss")
        plt.legend()       
        plt.savefig('train_loss_number.pdf')
               
if __name__ == '__main__':
    
    myMBNet = myMBNet()
    model = myMBNet.get_MBNet()   
    myMBNet.train()

    
    
    
    