# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
import cv2
from PIL import Image
import argparse
import math
from mylib.keras import dataset
from mylib.process_img import combine_imgs
from mylib.cv2_tools import display_img
from mylib.tools import *


# mnistの場合
def mnist_generator_model():
    model = Sequential()

    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))

    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))

    model.add(Reshape((128, 7, 7), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Convolution2D(1, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    return model


# 論文のLESUN用
def lsun_generator_model():
    model = Sequential()

    model.add(Dense(input_dim=100, output_dim=1024*4*4))
    model.add(Activation('relu'))

    model.add(Dense(1024*4*4))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Reshape((1024, 4, 4), input_shape=(1024*4*4,)))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Convolution2D(512, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Convolution2D(256, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Convolution2D(128, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Convolution2D(3, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))

    return model


def font_generator_model():
    model = Sequential()

    model.add(Dense(input_dim=100, output_dim=512*4*4))
    model.add(BatchNormalization(512*4*4))
    model.add(Activation('relu'))

    model.add(Reshape((512, 4, 4)))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Convolution2D(256, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Convolution2D(128, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Convolution2D(128, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Convolution2D(1, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))

    return model


def discriminator_model():
    model = Sequential()

    model.add(Convolution2D(64, 4, 4, border_mode='same', input_shape=(1, 28, 28)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 4, 4))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def font_discriminator_model():
    model = Sequential()

    model.add(Convolution2D(64, 4, 4, subsample=(2,2), border_mode='same', input_shape=(1, 64, 64)))
    model.add(LeakyReLU(0.2))

    model.add(Convolution2D(128, 4, 4, subsample=(2,2), border_mode='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Convolution2D(256, 4, 4, subsample=(2,2), border_mode='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Convolution2D(512, 4, 4, subsample=(2,2), border_mode='same'))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Convolution2D(1, 4, 4, border_mode='same'))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)

    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[2:]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[0, :, :]
    return image


def train(train_txt_path, batch_size, g_hdf5_path='./generator.hdf5', d_hdf5_path='./discriminator.hdf5'):
    org_imgs, org_labels = dataset.filelist_to_list(train_txt_path)

    discriminator = font_discriminator_model()
    generator = font_generator_model()
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)

    d_opt = SGD(lr=0.0002, momentum=0.9, nesterov=True)
    g_opt = SGD(lr=0.0002, momentum=0.9, nesterov=True)

    generator.compile(loss='binary_crossentropy', optimizer='SGD')
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_opt)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_opt)

    noise = np.zeros((batch_size, 100))

    # 学習開始
    for epoch in range(100):
        print ('epoch = {0}'.format(epoch))
        print ('number of batches = {0}'.format(int(org_imgs.shape[0] / batch_size)))
        for index in range(int(org_imgs.shape[0] / batch_size)):
            for i in range(batch_size):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            batched_org_imgs = org_imgs[index*batch_size : (index+1)*batch_size]
            generated_imgs = generator.predict(noise, verbose=0)
            if index % 20 == 0:
                img = combine_images(generated_imgs)
                img = img*127.5+127.5
                Image.fromarray(img.astype(np.uint8)).save(
                    str(epoch)+"_"+str(index)+".png")
            X = np.concatenate((batched_org_imgs, generated_imgs))
            y = [1] * batch_size + [0] * batch_size
            d_loss = discriminator.train_on_batch(X, y)
            print("batch {0} d_loss: {1}".format(index, d_loss))
            for i in range(batch_size):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(noise, [1] * batch_size)
            discriminator.trainable = True
            print('batch {0} g_loss : {1}'.format(index, g_loss))
            if index % 10 == 9:
                generator.save_weights(g_hdf5_path, True)
                discriminator.save_weights(d_hdf5_path, True)


def debug():
    train_txt_path = '/home/abe/font_dataset/png_6628_64x64/all_1000.txt'
    train(train_txt_path, 128)


if __name__ == '__main__':
    debug()
