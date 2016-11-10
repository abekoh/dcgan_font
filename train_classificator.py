# -*- coding: utf-8 -*-
import numpy as np

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


class LeNet(chainer.Chain):

    def __init__(self):
        super(LeNet, self).__init__(
            conv1=L.Convolution2D(1, 20, 5),
            conv2=L.Convolution2D(20, 50, 5),
            fc3=L.Linear(8450, 500),
            fc4=L.Linear(500, 26))

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2, stride=2)
        h = F.relu(self.fc3(h))
        y = F.softmax(self.fc4(h))
        return y

class AlexNet(chainer.Chain):

    def __init__(self):
        super(AlexNet, self).__init__(
            conv1=L.Convolution2D(1,  96, 8, stride=4), # -> 15
            conv2=L.Convolution2D(96, 256,  5, pad=2), # -> 7
            conv3=L.Convolution2D(256, 384,  3, pad=1), # -> 2
            conv4=L.Convolution2D(384, 384,  3, pad=1), # -> 2
            conv5=L.Convolution2D(384, 256,  3, pad=1), # -> 2
            fc6=L.Linear(256, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 26),
        )

    def __call__(self, x, train=True):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2) # -> 7
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2) # -> 2
        h = F.relu(self.conv3(h)) # -> 2
        h = F.relu(self.conv4(h)) # -> 2
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        # h = F.relu(self.conv5(h)) # -> 2
        h = F.dropout(F.relu(self.fc6(h)), train=train)
        h = F.dropout(F.relu(self.fc7(h)), train=train)
        y = self.fc8(h)
        return y

def train(train_txt_path, test_txt_path, dst_dir_path, epoch_n=100, batch_size=128, model=AlexNet()):
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



def debug():
    train_txt_path = '/home/abe/font_dataset/png_6628_64x64/train.txt'
    test_txt_path = '/home/abe/font_dataset/png_6628_64x64/test.txt'
    dst_dir_path = tools.make_date_dir('/home/abe/dcgan_font/output_classificator/debug/')
    train(train_txt_path, test_txt_path, dst_dir_path)


if __name__ == '__main__':
    debug()
