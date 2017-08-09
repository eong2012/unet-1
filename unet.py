# Code (c) Sam Russell 2017
# Implementation of U-Net: Convolutional Networks for Biomedical Image Segmentation
# https://arxiv.org/pdf/1505.04597.pdf

import base_trainer
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Input, concatenate, Conv2DTranspose
from keras.layers.core import Activation, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.activations import *
from keras.utils import to_categorical
from keras.datasets import cifar10
import keras
from PIL import Image
import numpy as np


def dice_coef(y_true, y_pred):
  smooth = 1.
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
  return -dice_coef(y_true, y_pred)


class UnetTrainer(base_trainer.BaseTrainer):

  def __init__(self, img_rows = 32, img_cols = 32, img_channels = 3, num_classes = 11):
    self.img_rows = img_rows
    self.img_cols = img_cols
    self.img_channels = img_channels
    self.num_classes = num_classes

  def build_models(self, input_shape):
    #inputs = Input((self.img_rows, self.img_cols, self.img_channels))
    inputs = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    self.model = model

  def load_data(self):
    number_path = "number_masking_data/numbers"
    mask_path = "number_masking_data/masks"
    number_images = []
    mask_images = []
    num_samples = 10

    for number in xrange(self.num_classes):
      for index in xrange(num_samples):
        number_image = Image.open("%s/%s/%s.png" % (number_path, number, index+1)).convert("RGB")
        mask_image = Image.open("%s/%s/%s.png" % (mask_path, number, index+1)).convert("RGB")
        number_image_data = np.asarray(number_image, dtype='float32') / 255.
        mask_image_data = np.asarray(mask_image, dtype='float32')[:,:,:1] / 255.
        if keras.backend.image_data_format() == 'channels_first':
          number_image_data = number_image_data.transpose(2, 0, 1)
          mask_image_data = mask_image_data.transpose(2, 0, 1)
        number_images.append(number_image_data)
        mask_images.append(mask_image_data)

    self.image_data = [np.array(number_images), np.array(mask_images)]

    number_images = []
    for number in xrange(self.num_classes):
      for index in xrange(num_samples, num_samples+5):
        number_image = Image.open("%s/%s/%s.png" % (number_path, number, index+1)).convert("RGB")
        number_image_data = np.asarray(number_image, dtype='float32') / 255.
        if keras.backend.image_data_format() == 'channels_first':
          number_image_data = number_image_data.transpose(2, 0, 1)
        number_images.append(number_image_data)

    self.testing_data = [np.array(number_images), None]

  def load_training_data(self):
    #training_dataframe = pandas.read_csv(self.commandline_args.train)
    #values = training_dataframe.values[:,1:]
    #labels = training_dataframe.values[:,0]
    return self.image_data

  def load_testing_data(self):
    return self.testing_data

if __name__ == "__main__":
  UnetTrainer().run()
