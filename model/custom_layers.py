#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-11 17:12:26
#   Description : keras_solo
#
# ================================================================
#import keras
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import backend as K
import tensorflow.python.keras.engine as Layer



class Conv2dUnit(object):
    def __init__(self, filters, kernels, strides=1, padding='valid', use_bias=False, bn=1, act='relu'):
        super(Conv2dUnit, self).__init__()
        self.conv = layers.Conv2D(filters, kernels,
                   padding=padding,
                   strides=strides,
                   use_bias=use_bias,
                   activation='linear',
                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01))
        self.bn = None
        self.act = None
        if bn:
            self.bn = layers.BatchNormalization()
        if act == 'leaky':
            self.act = layers.advanced_activations.LeakyReLU(alpha=0.1)
        elif act == 'relu':
            self.act = layers.advanced_activations.ReLU()

    def __call__(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x

class Conv3x3(object):
    def __init__(self, filters2, strides, use_dcn):
        super(Conv3x3, self).__init__()
        if use_dcn:
            self.conv2d_unit = None
        else:
            self.conv2d_unit = Conv2dUnit(filters2, 3, strides=strides, padding='same', use_bias=False, bn=0, act=None)
        self.bn = layers.BatchNormalization()
        self.act = layers.advanced_activations.ReLU()

    def __call__(self, x):
        x = self.conv2d_unit(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class InstanceNormalization(Layer):
    """InstanceNormalization，输出图片的格式是(N, h, w, c)
    """
    def __init__(self, epsilon=1e-9, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        super(InstanceNormalization, self).build(input_shape)
        shape = (input_shape[-1], )
        self.gamma = self.add_weight(shape=shape,
                                     initializer='ones',
                                     name='gamma')
        self.beta = self.add_weight(shape=shape,
                                    initializer='zeros',
                                    name='beta')

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        N, h, w, c = K.shape(x)[0], K.shape(x)[1], K.shape(x)[2], K.shape(x)[3]
        x_reshape = K.reshape(x, (N, h*w, c))
        mean = K.mean(x_reshape, axis=1, keepdims=True)
        t = K.square(x_reshape - mean)
        variance = K.mean(t, axis=1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (x_reshape - mean) / std
        outputs = outputs*self.gamma + self.beta
        outputs = K.reshape(outputs, (N, h, w, c))
        return outputs


class GroupNormalization(Layer):
    """GroupNormalization，输出图片的格式是(N, h, w, c)
    """
    def __init__(self, num_groups, epsilon=1e-9, **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.num_groups = num_groups
        self.epsilon = epsilon

    def build(self, input_shape):
        super(GroupNormalization, self).build(input_shape)
        shape = (input_shape[-1], )
        self.gamma = self.add_weight(shape=shape,
                                     initializer='ones',
                                     name='gamma')
        self.beta = self.add_weight(shape=shape,
                                    initializer='zeros',
                                    name='beta')

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        N, h, w, c = K.shape(x)[0], K.shape(x)[1], K.shape(x)[2], K.shape(x)[3]
        x_reshape = K.reshape(x, (N, -1, self.num_groups))   # 把同一group的元素融合到一起。IN是GN的特例，当num_groups为c时。
        mean = K.mean(x_reshape, axis=1, keepdims=True)
        t = K.square(x_reshape - mean)
        variance = K.mean(t, axis=1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (x_reshape - mean) / std
        outputs = K.reshape(outputs, (N, h*w, c))
        outputs = outputs*self.gamma + self.beta
        outputs = K.reshape(outputs, (N, h, w, c))
        return outputs

class Resize(Layer):
    def __init__(self, h, w, method):
        super(Resize, self).__init__()
        self.h = h
        self.w = w
        self.method = method
    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0], self.h, self.w, input_shape[3])
    def call(self, x):
        if self.method == 'BICUBIC':
            m = tf.image.ResizeMethod.BICUBIC
        elif self.method == 'NEAREST_NEIGHBOR':
            m = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        elif self.method == 'BILINEAR':
            m = tf.image.ResizeMethod.BILINEAR
        elif self.method == 'AREA':
            m = tf.image.ResizeMethod.AREA
        a = tf.image.resize_images(x, [self.h, self.w], method=m)
        return a







