# -*- coding: utf-8 -*-
import pickle
import numpy as np
import os
import io
import math
import pylab
from PIL import Image
import cv2

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer.utils import type_check
from chainer import function

import chainer.functions as F
import chainer.links as L

from mylib.chainer import dataset
from mylib.cv2_tools import display_img
from mylib.tools import *
from mylib.process_img.combine_imgs import combine_imgs

def clip_img(x):
	return np.float32(-1 if x<-1 else (1 if x>1 else x))

# # 元論文ImageNet
# class Generator(chainer.Chain):
#     def __init__(self):
#         super(Generator, self).__init__(
#             fc5=L.Linear(100, 512 * 4 * 4),
#             norm5=L.BatchNormalization(512 * 4 * 4),
#             conv4=L.Deconvolution2D(512, 256, ksize=4, stride=2, pad=1),
#             norm4=L.BatchNormalization(256),
#             conv3=L.Deconvolution2D(256, 128, ksize=4, stride=2, pad=1),
#             norm3=L.BatchNormalization(128),
#             conv2=L.Deconvolution2D(128, 64,  ksize=4, stride=2, pad=1),
#             norm2=L.BatchNormalization(64),
#             conv1=L.Deconvolution2D(64, 1, ksize=4, stride=2, pad=1))
#
#     def __call__(self, z, test=False):
#         n_sample = z.data.shape[0]
#         h = F.relu(self.norm5(self.fc5(z), test=test))
#         h = F.reshape(h, (n_sample, 512, 4, 4))
#         h = F.relu(self.norm4(self.conv4(h), test=test))
#         h = F.relu(self.norm3(self.conv3(h), test=test))
#         h = F.relu(self.norm2(self.conv2(h), test=test))
#         x = F.tanh(self.conv1(h))
#         return x

class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            conv1=L.Convolution2D(1,   64,  ksize=4, stride=2, pad=1),
            conv2=L.Convolution2D(64,  128, ksize=4, stride=2, pad=1),
            norm2=L.BatchNormalization(128),
            conv3=L.Convolution2D(128, 256, ksize=4, stride=2, pad=1),
            norm3=L.BatchNormalization(256),
            conv4=L.Convolution2D(256, 512, ksize=4, stride=2, pad=1),
            norm4=L.BatchNormalization(512),
            fc5=L.Linear(512 * 4 * 4, 2))

    def __call__(self, x, test=False):
        n_sample = x.data.shape[0]
        h = F.leaky_relu(self.conv1(x))
        h = F.leaky_relu(self.norm2(self.conv2(h), test=test))
        h = F.leaky_relu(self.norm3(self.conv3(h), test=test))
        h = F.leaky_relu(self.norm4(self.conv4(h), test=test))
        y = self.fc5(h)
        return y

# Keras版MNIST
# class Generator(chainer.Chain):
#     def __init__(self):
#         super(Generator, self).__init__(
#             fc1 = L.Linear(100, 1024),
#             fc2 = L.Linear(1024, 128*16*16),
#             norm2 = L.BatchNormalization(128*16*16),
#             conv3 = L.Deconvolution2D(128, 64, ksize=4, stride=1, pad=1),
#             conv4 = L.Deconvolution2D(64, 1, ksize=4, stride=1, pad=1))
#
#     def __call__(self, z, test=False):
#         h = F.tanh(self.fc1(z))
#         h = F.tanh(self.norm2(self.fc2(h), test=test))
#         h = F.reshape(h, (100, 128, 16, 16))
#         h = F.tanh(self.conv3(F.unpooling_2d(h, ksize=2)))
#         x = F.tanh(self.conv4(F.unpooling_2d(h, ksize=2)))
#         return x
#
# class Discriminator(chainer.Chain):
#     def __init__(self):
#         super(Discriminator, self).__init__(
#             conv1 = L.Convolution2D(1, 64, ksize=4, stride=1, pad=1),
#             conv2 = L.Convolution2D(64, 128, ksize=4, stride=1, pad=1),
#             fc3 = L.Linear(128*16*16, 1024),
#             fc4 = L.Linear(1024, 1))
#
#     def __call__(self, x, test=False):
#         h = F.max_pooling_2d(F.tanh(self.conv1(x)), ksize=2)
#         h = F.max_pooling_2d(F.tanh(self.conv2(h)), ksize=2)
#         h = F.tanh(self.fc3(h))
#         y = F.sigmoid(self.fc4(h))
#         return y

class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
            fc5=L.Linear(100, 512 * 4 * 4),
            norm5=L.BatchNormalization(512 * 4 * 4),
            conv4=L.Deconvolution2D(512, 256, ksize=4, stride=2, pad=1),
            norm4=L.BatchNormalization(256),
            conv3=L.Deconvolution2D(256, 128, ksize=4, stride=2, pad=1),
            norm3=L.BatchNormalization(128),
            conv2=L.Deconvolution2D(128, 64,  ksize=4, stride=2, pad=1),
            norm2=L.BatchNormalization(64),
            conv1=L.Deconvolution2D(64, 1, ksize=4, stride=2, pad=1))

    def __call__(self, z, test=False):
        n_sample = z.data.shape[0]
        h = F.relu(self.norm5(self.fc5(z), test=test))
        h = F.reshape(h, (n_sample, 512, 4, 4))
        h = F.relu(self.norm4(self.conv4(h), test=test))
        h = F.relu(self.norm3(self.conv3(h), test=test))
        h = F.relu(self.norm2(self.conv2(h), test=test))
        x = F.sigmoid(self.conv1(h))
        return x


def train(train_txt_path, dst_dir_path):
    batch_size=100

    org_imgs = dataset.filelist_to_list_for_dcgan(train_txt_path)
    
    xp = cuda.cupy
    cuda.get_device(0).use()

    generator = Generator()
    discriminator = Discriminator()
    generator.to_gpu()
    discriminator.to_gpu()

    # g_opt = optimizers.MomentumSGD(lr=0.0005, momentum=0.9)
    # d_opt = optimizers.MomentumSGD(lr=0.0005, momentum=0.9)
    g_opt = optimizers.Adam(alpha=0.0002, beta1=0.5)
    d_opt = optimizers.Adam(alpha=0.0002, beta1=0.5)
    g_opt.setup(generator)
    d_opt.setup(discriminator)
    g_opt.add_hook(chainer.optimizer.WeightDecay(0.00001))
    d_opt.add_hook(chainer.optimizer.WeightDecay(0.00001))

    for epoch in range(100):
        sum_g_loss = np.float32(0)
        sum_d_loss = np.float32(0)
        print ('epoch: {0}/100'.format(epoch))
        for index in range(int(org_imgs.shape[0]/batch_size)):
            print ('batch: {0}/{1}'.format(index, int(org_imgs.shape[0]/batch_size)))
            z = Variable(xp.random.uniform(-1, 1, (batch_size, 100), dtype=np.float32))
            generated_imgs = generator(z)
            generated_d_score = discriminator(generated_imgs)
            # 0: original, 1: generated
            g_loss = F.softmax_cross_entropy(generated_d_score, Variable(xp.zeros(batch_size, dtype=np.int32)))
            d_loss = F.softmax_cross_entropy(generated_d_score, Variable(xp.ones(batch_size, dtype=np.int32))) 
            batched_org_imgs = np.zeros((batch_size, 1, 64, 64), dtype=np.float32)
            for i, j in enumerate(range(index*batch_size, (index+1)*batch_size)):
                batched_org_imgs[i, :, :, :] = (org_imgs[j] - 128.0)/128.0
            batched_org_imgs = Variable(cuda.to_gpu(batched_org_imgs))
            original_d_score = discriminator(batched_org_imgs)
            d_loss += F.softmax_cross_entropy(original_d_score, Variable(xp.zeros(batch_size, dtype=np.int32)))

            g_opt.zero_grads()
            g_loss.backward()
            g_opt.update()

            d_opt.zero_grads()
            d_loss.backward()
            d_opt.update()

            g_loss_data = g_loss.data.get()
            d_loss_data = d_loss.data.get()

            print('g_loss: {0}, d_loss: {1}'.format(g_loss_data, d_loss_data))

            sum_g_loss += g_loss_data
            sum_d_loss += d_loss_data

            if index % 20 == 0:
                z = Variable(xp.random.uniform(-1, 1, (100, 100), dtype=np.float32))
                generated_imgs = generator(z, test=True)
                generated_imgs = generated_imgs.data.get()
                g_imgs = []
                for g_img in generated_imgs:
                    g_imgs.append(g_img[0])
                combined_img = combine_imgs(g_imgs, 10)
                combined_img = combined_img*127.5 + 127.5
                combined_img = combined_img.astype(np.int32)
                cv2.imwrite('{0}pic/vis_{1}_{2}.png'.format(dst_dir_path, epoch, index), combined_img)


        serializers.save_hdf5("{0}dcgan_model_dis_{1}.hdf5".format(dst_dir_path, epoch),discriminator)
        serializers.save_hdf5("{0}dcgan_model_gen_{1}.hdf5".format(dst_dir_path, epoch),generator)
        serializers.save_hdf5("{0}dcgan_state_dis_{1}.hdf5".format(dst_dir_path, epoch),d_opt)
        serializers.save_hdf5("{0}dcgan_state_gen_{1}.hdf5".format(dst_dir_path, epoch),g_opt)
        print ('epoch end', epoch, sum_d_loss, sum_g_loss)


def debug():
    train_txt_path = '/home/abe/font_dataset/png_6628_64x64/all_1000.txt'
    train(train_txt_path, '/home/abe/dcgan_font/test_output/')


if __name__ == '__main__':
    debug()
