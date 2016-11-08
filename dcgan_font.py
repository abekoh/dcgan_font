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

from chainer import functions as F
from chainer import links as L

from mylib.chainer import dataset
from mylib.cv2_tools import display_img
from mylib.tools import *
from mylib.process_img.combine_imgs import combine_imgs

import models

def clip_img(x):
	return np.float32(-1 if x<-1 else (1 if x>1 else x))

def train(train_txt_path, dst_dir_path):
    epoch_n = 1000
    batch_size=100

    org_imgs = dataset.filelist_to_list_for_dcgan(train_txt_path)

    make_dir(dst_dir_path + 'pic/')

    xp = cuda.cupy
    cuda.get_device(0).use()

    generator = models.Generator()
    discriminator = models.Discriminator()
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

    for epoch_i in range(epoch_n):
        sum_g_loss = np.float32(0)
        sum_d_loss = np.float32(0)
        batch_n = int(org_imgs.shape[0]/batch_size)
        for batch_i in range(batch_n):
            print ('epoch:{0}/{1}, batch:{2}/{3}'.format(epoch_i, epoch_n, batch_i, batch_n))
            z = Variable(xp.random.uniform(-1, 1, (batch_size, 100), dtype=np.float32))
            generated_imgs = generator(z)
            generated_d_score = discriminator(generated_imgs)
            # 0: original, 1: generated
            g_loss = F.softmax_cross_entropy(generated_d_score, Variable(xp.zeros(batch_size, dtype=np.int32)))
            d_loss = F.softmax_cross_entropy(generated_d_score, Variable(xp.ones(batch_size, dtype=np.int32))) 
            batched_org_imgs = np.zeros((batch_size, 1, 64, 64), dtype=np.float32)
            for i, j in enumerate(range(batch_i*batch_size, (batch_i+1)*batch_size)):
                batched_org_imgs[i, :, :, :] = (org_imgs[j] - 127.5)/127.5
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

            print('g_loss:{0}, d_loss:{1}'.format(g_loss_data, d_loss_data))

            sum_g_loss += g_loss_data
            sum_d_loss += d_loss_data

            if batch_i % 200 == 0:
                z = Variable(xp.random.uniform(-1, 1, (100, 100), dtype=np.float32))
                generated_imgs = generator(z, test=True)
                generated_imgs = generated_imgs.data.get()
                g_imgs = []
                for g_img in generated_imgs:
                    g_imgs.append(g_img[0])
                combined_img = combine_imgs(g_imgs, 10)
                combined_img = combined_img*127.5 + 127.5
                combined_img = combined_img.astype(np.int32)
                cv2.imwrite('{0}pic/{1}_{2}.png'.format(dst_dir_path, epoch_i, batch_i), combined_img)
        if epoch_i % 50 == 0 and epoch_i != 0:
            serializers.save_hdf5("{0}dcgan_model_dis_{1}.hdf5".format(dst_dir_path, epoch_i),discriminator)
            serializers.save_hdf5("{0}dcgan_model_gen_{1}.hdf5".format(dst_dir_path, epoch_i),generator)
            serializers.save_hdf5("{0}dcgan_state_dis_{1}.hdf5".format(dst_dir_path, epoch_i),d_opt)
            serializers.save_hdf5("{0}dcgan_state_gen_{1}.hdf5".format(dst_dir_path, epoch_i),g_opt)

def generate(generator_hdf5_path):
    xp = np
    generator = models.Generator()
    serializers.load_hdf5(generator_hdf5_path, generator)
    z = (xp.random.uniform(-1, 1, (100, 100)).astype(np.float32))
    z = Variable(z)
    generated_imgs = generator(z, test=True)
    generated_imgs = generated_imgs.data
    g_imgs = []
    for g_img in generated_imgs:
        g_imgs.append(g_img[0])
    combined_img = combine_imgs(g_imgs, 10)
    combined_img = combined_img*127.5 + 127.5
    combined_img = combined_img.astype(np.int32)
    cv2.imwrite('generated.png', combined_img)
    display_img(combined_img)

def debug():
    train_txt_path = '/home/abe/font_dataset/png_6628_64x64/alph_list/all_A.txt'
    train(train_txt_path, make_date_dir('/home/abe/dcgan_font/output/debug/'))
    # generate('/home/abe/dcgan_font/output/A_likeMNIST/dcgan_model_gen_80.hdf5')

if __name__ == '__main__':
    debug()

