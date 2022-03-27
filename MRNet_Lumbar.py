import tensorflow as tf
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
tf.config.list_physical_devices('GPU')
print (device_lib.list_local_devices())

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.models import*
from tensorflow.keras.layers import*
from tensorflow.keras.optimizers import*
from tensorflow.keras.callbacks import ModelCheckpoint
from process_MRNet_data import*
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import *
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras import backend as K
import keras_radam
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
import numpy as ny
import pandas
from sklearn import preprocessing
import joblib

import tensorflow.python.keras.engine
from model.custom_layers import Conv2dUnit, Conv3x3
from tensorflow.keras import applications
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

mm_scaler = preprocessing.MinMaxScaler()

    
class ConvBlock(object):
    
    def __init__(self, filters, use_dcn=False, stride=2):      
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters

        self.conv1 = Conv2dUnit(filters1, 1, strides=1, padding='valid', use_bias=False, bn=1, act= activation_method )
        self.conv2 = Conv3x3(filters2, stride, use_dcn)
        self.conv3 = Conv2dUnit(filters3, 1, strides=1, padding='valid', use_bias=False, bn=1, act=activation_method)

        self.conv4 = Conv2dUnit(filters3, 1, strides=stride, padding='valid', use_bias=False, bn=1, act=activation_method)
        self.act = ReLU()

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        shortcut = self.conv4(input_tensor)
        x = add([x, shortcut])
        x = self.act(x)
        return x

class IdentityBlock(object):
    def __init__(self, filters, use_dcn=False):
        super(IdentityBlock, self).__init__()
        filters1, filters2, filters3 = filters

        self.conv1 = Conv2dUnit(filters1, 1, strides=1, padding='valid', use_bias=False, bn=1, act= activation_method )
        self.conv2 = Conv3x3(filters2, 1, use_dcn)
        self.conv3 = Conv2dUnit(filters3, 1, strides=1, padding='valid', use_bias=False, bn=1, act=activation_method)

        self.act = ReLU()

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        x = add([x, input_tensor])
        x = self.act(x)
        return x


def segmentation_branch(inputs):

    conv1R = Conv2dUnit(64, 3, strides=1, padding='same', use_bias=False, bn=1, act= activation_method)(inputs)
    maxpool = MaxPooling2D(pool_size=2, strides=2, padding='same')(conv1R)
    print('maxpool', maxpool)
               

    stage2_0 = ConvBlock([64, 64, 256], stride=1)(maxpool)
    stage2_1 = IdentityBlock([64, 64, 256])(stage2_0)
    stage2_2 = IdentityBlock([64, 64, 256])(stage2_1)
    print('stage2_2', stage2_2)

    stage3_0 = ConvBlock([128, 128, 512], use_dcn=False)(stage2_2)
    stage3_1 = IdentityBlock([128, 128, 512], use_dcn=False)(stage3_0)
    stage3_2 = IdentityBlock([128, 128, 512], use_dcn=False)(stage3_1)
    stage3_3 = IdentityBlock([128, 128, 512], use_dcn=False)(stage3_2)
    print('stage3_3', stage3_3)
        
    stage4_0 = ConvBlock([256, 256, 1024], use_dcn=False)(stage3_3)
    stage4_last_layer = IdentityBlock([256, 256, 1024], use_dcn=False)(stage4_0)
    print('stage4_last_layer', stage4_last_layer)

    # stage5
    stage5_0 = ConvBlock([512, 512, 2048], use_dcn=False)(stage4_last_layer)
    stage5_1 = IdentityBlock([512, 512, 2048], use_dcn=False)(stage5_0)
    stage5_2 = IdentityBlock([512, 512, 2048], use_dcn=False)(stage5_1) 
    print('stage5_2', stage5_2)
        
        
    conv6 = Conv2D(512, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(stage5_2)
    conv6 = Conv2D(512, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(conv6)        
    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (conv6)
    merge7 = concatenate([stage4_last_layer, up7], axis=3)
        

    conv7 = Conv2D(256, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(conv7)
    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (conv7)
    print('up8', up8) 
    merge8 = concatenate([stage3_3, up8], axis=3) 
        
         
    conv8 = Conv2D(128, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(conv8)
    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv8)
    print('up9', up9)  
    merge9 = concatenate([stage2_2, up9], axis=3)

        
    conv8l = Conv2D(64, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(merge9)
    conv8l = Conv2D(64, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(conv8l)
    up9l = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (conv8l)
    print('up9l', up9l)
         
    conv10 = Conv2D(32, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(up9l) 
    conv11 = Conv2D(32, 3, activation=activation_method, padding='same', kernel_initializer='he_normal')(conv10) 
    out_SSB = Conv2D(1, 1, activation='softmax', name="seg_output")(conv11)   

    return out_SSB


def inspected_values_branch(in_layer):
    
     conv1 = Conv2D(32,3, activation= activation_method, padding='same', kernel_initializer='he_normal')(in_layer)
     pool1M = MaxPooling2D(pool_size=(5, 5),strides= 4)(conv1)
        
     conv2 = Conv2D(128,3, activation= activation_method, padding='same', kernel_initializer='he_normal')(pool1M)           
     pool12M = MaxPooling2D(pool_size=(5, 5),strides= 4)(conv2)
                 
     LayerF = Flatten()(pool12M)  
     D1 = Dense(1024)(LayerF)
     D1 = Activation('sigmoid')(D1)

     D2 = Dense(1024)(D1)
     D2 = Activation('sigmoid')(D2)
     Do = Dense(18)(D2)  
     out_IVB = Activation('sigmoid', name="number_output")(Do)
     return out_IVB


        
class myMRNet(object):
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
        joblib.dump(mm_scaler, 'scaler2.gz')        
        return imgs_train, imgs_mask_train, Y_train
    

    
    def get_MRNet(self):
        
        inputs = Input((self.img_rows, self.img_cols, 1))     
        out_SSB = segmentation_branch(inputs)
        out_IVB = inspected_values_branch(out_SSB)
        model = Model(inputs= inputs, outputs= [out_IVB, out_SSB])
        
        
        losses = { "number_output": "mse", "seg_output": "binary_crossentropy"}
        lossWeights = {"number_output": 0.1, "seg_output": 0.9}
        
        opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=["acc"])
        
        #model.summary()
        #plot_model(model, to_file='model_plot.pdf', show_shapes=True, show_layer_names=True)
        return model

    def train(self):
        print("loading data")
        imgs_train, imgs_mask_train, Y_train = self.load_data()
        print("loading data done")
        model = self.get_MRNet()
        print("got MRNet")    
        print("number layer myMRNet")
        print(len(model.layers))
       
        
        
        model_checkpoint = ModelCheckpoint('MRNET1000_V4.hdf5', monitor= 'val_loss', verbose=1, mode= 'auto', save_best_only= False)
        print('Fitting model...')   
        results = model.fit(imgs_train, {"number_output": Y_train, "seg_output": imgs_mask_train}, validation_split=0.1, shuffle=True, callbacks=[model_checkpoint], epochs=50, batch_size= 1, verbose=1)
  
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
    
    myMRNet = myMRNet()
    model = myMRNet.get_MRNet()   
    myMRNet.train()

    
    
    
    
