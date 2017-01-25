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


def train(train_txt_path, test_txt_path, dst_dir_path, is_save_temp=True, last_model_filename='last', epoch_n=100, batch_size=128, model=models.Classifier_AlexNet()):
    '''
    Classifierの学習
    Args:
        train_txt_path:         学習に用いる画像のパスを記載したtxt．
                                1列目は画像パス，2列目はクラスID('A'から順に0,1,2...)
                                ex) /home/hoge/font/A/0.png, 0
                                    /home/hoge/font/B/0.png, 1
                                    /home/hoge/font/C/0.png, 2
                                    /home/hoge/2.png, 0
        test_txt_path:          テストに用いる画像のパスを記載したtxt．
                                フォーマットはtrain_txt_pathと同じ．
        dst_dir_path:           学習済みモデルの出力先．
        epoch_n:                学習回数．
        batch_size:             バッチサイズ．
        model:                  Classifierの学習済みモデルのパス．(models.pyのクラス)
    '''
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

        if is_save_temp and (epoch_i != epoch_n - 1):
            serializers.save_hdf5('{0}model_{1}.hdf5'.format(dst_dir_path, epoch_i), model)
            serializers.save_hdf5('{0}state_{1}.hdf5'.format(dst_dir_path, epoch_i), optimizer)
    serializers.save_hdf5('{0}model_{1}.hdf5'.format(dst_dir_path, last_model_filename), model)
    serializers.save_hdf5('{0}state_{1}.hdf5'.format(dst_dir_path, last_model_filename), optimizer)



def classify(src_png_path, classifier):
    '''
    クラス分別の実行
    Args:
        src_png_path:   分別する画像のパス
        classifier:     Classifierのモデルの構造(models.pyのクラス)
    Return:
        predict_label:  分別されたラベル(クラスID)
    '''
    img = cv2.imread(src_png_path, -1)
    img = img.astype(np.float32)
    img /= 255
    img = img[np.newaxis, np.newaxis, :, :]
    x = Variable(img)
    y = classifier.predictor(x)
    max_score = 0
    for i, score in enumerate(y.data[0]):
        if score > max_score:
            max_score = score
            predict_label = i
    return predict_label


def output_accuracy_rate(img_paths, labels, 
                         model=models.Classifier_AlexNet(), 
                         hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5'):
    '''
    正解率の出力
    Args:
        img_paths:      対象の画像のパス
        labels:         対象の画像の正解ラベル
        model:          Classifierのモデルの構造(models.pyのクラス)
        hdf5_path:      Classifierの学習済みモデルのパス
    '''
    serializers.load_hdf5(hdf5_path, model)
    classifier = L.Classifier(model)
    correct_n = 0
    for img_path, label in zip(img_paths, labels):
        if label == classify(img_path, classifier):
            print(img_path, '正解')
            correct_n += 1
        else:
            print(img_path, '不正解')
    accuracy_rate = float(correct_n) / float(len(img_paths))
    print ('correct_n:', correct_n)
    print (accuracy_rate)

def debug():
    # # train
    # train_txt_path = '/home/abe/font_dataset/png_6628_64x64/train_noise.txt'
    # test_txt_path = '/home/abe/font_dataset/png_6628_64x64/test_noise.txt'
    # dst_dir_path = tools.make_date_dir('/home/abe/dcgan_font/output_classificator/debug/')
    # train(train_txt_path, test_txt_path, dst_dir_path, model=models.Classifier(noise=True))
    # classify
    # print (classify('/home/abe/font_dataset/png_6628_64x64/B/3239.png'))
    # output_accuracy_rate
    path_tmp1 = '/home/abe/dcgan_font/output_storage/forPRMU/CNN_Test/plusclassifier/'
    img_paths, labels = [], []
    for alph in ['A', 'B', 'C', 'D']:
        path_tmp2 = path_tmp1 + alph + '_'
        for i in range(2500):
            img_path = path_tmp2 + str(i) + '.png'
            img_paths.append(img_path)
            labels.append(ord(alph) - 65)
    output_accuracy_rate(img_paths, labels)


if __name__ == '__main__':
    debug()
