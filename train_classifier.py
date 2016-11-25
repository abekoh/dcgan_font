# -*- coding: utf-8 -*-
import numpy as np
import cv2

import chainer
from chainer import cuda
from chainer import optimizers
from chainer import FunctionSet
from chainer import Variable
from chainer import serializers

from chainer import functions as F
from chainer import links as L

from mylib.chainer import dataset
from mylib import tools

import models


def train(train_txt_path, test_txt_path, dst_dir_path, epoch_n=100, batch_size=128, model=models.Classifier()):
    train_imgs, train_labels = dataset.filelist_to_list(train_txt_path)
    test_imgs, test_labels = dataset.filelist_to_list(test_txt_path)

    xp = cuda.cupy
    cuda.get_device(0).use()

    model.to_gpu()

    optimizer = optimizers.SGD(lr=0.01)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    train_imgs_n = train_imgs.shape[0]
    test_imgs_n = test_imgs.shape[0]

    for epoch_i in range(epoch_n):
        print('epoch:{0}/{1}'.format(epoch_i, epoch_n))
        perm = np.random.permutation(train_imgs_n)
        train_sum_acc = 0
        train_sum_loss = 0
        test_sum_acc = 0
        test_sum_loss = 0
        for batch_i in range(0, train_imgs_n, batch_size):
            optimizer.zero_grads()
            batched_imgs = xp.asarray(train_imgs[perm[batch_i: batch_i + batch_size]])
            batched_labels = xp.asarray(train_labels[perm[batch_i: batch_i + batch_size]])
            batched_imgs_score = model(Variable(batched_imgs))
            loss = F.softmax_cross_entropy(
                batched_imgs_score, Variable(batched_labels))
            acc = F.accuracy(batched_imgs_score, Variable(batched_labels))

            loss.backward()
            optimizer.update()

            train_sum_loss += float(loss.data.get()) * batch_size
            train_sum_acc += float(acc.data.get()) * batch_size

        for batch_i in range(0, test_imgs_n, batch_size):
            batched_imgs = xp.asarray(test_imgs[batch_i: batch_i + batch_size])
            batched_labels = xp.asarray(test_labels[batch_i: batch_i + batch_size])
            batched_imgs_score = model(Variable(batched_imgs))
            loss = F.softmax_cross_entropy(
                batched_imgs_score, Variable(batched_labels))
            acc = F.accuracy(batched_imgs_score, Variable(batched_labels))

            test_sum_loss += float(loss.data.get()) * batch_size
            test_sum_acc += float(acc.data.get()) * batch_size
 
        print('train: loss={0}, accuracy={1}'.format(train_sum_loss / train_imgs_n, train_sum_acc / train_imgs_n))
        print('test:  loss={0}, accuracy={1}'.format(test_sum_loss / test_imgs_n, test_sum_acc / test_imgs_n))

        serializers.save_hdf5('{0}model_{1}.hdf5'.format(dst_dir_path, epoch_i), model)
        serializers.save_hdf5('{0}state_{1}.hdf5'.format(dst_dir_path, epoch_i), optimizer)

def classify(src_png_path, model=models.Classifier(noise=False), hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5'):
    xp = np
    serializers.load_hdf5(hdf5_path, model)
    classifier = L.Classifier(model)
    img = cv2.imread(src_png_path, -1)
    img = img.astype(np.float32)
    img /= 255
    img = img[np.newaxis, np.newaxis, :, :]
    x = Variable(img)
    y = classifier.predictor(x)
    alph_list = tools.make_alphabets()
    for alph, score in zip(alph_list, y.data[0]):
        print (alph, score)
    

def debug():
    # train
    train_txt_path = '/home/abe/font_dataset/png_6628_64x64/train_noise.txt'
    test_txt_path = '/home/abe/font_dataset/png_6628_64x64/test_noise.txt'
    dst_dir_path = tools.make_date_dir('/home/abe/dcgan_font/output_classificator/debug/')
    train(train_txt_path, test_txt_path, dst_dir_path, model=models.Classifier(noise=True))
    # # classify
    # classify('/home/abe/font_dataset/png_6628_64x64/[/3238.png')


if __name__ == '__main__':
    debug()
