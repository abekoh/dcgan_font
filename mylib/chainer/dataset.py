# -*- coding: utf-8 -*-
import cv2
import numpy as np
from chainer import cuda

def filelist_to_list_for_dcgan(filename):
    image_list = []
    # 行数をゲット
    with open(filename, 'r') as file:
        max_count = sum(1 for line in file)
    file = open(filename, 'r')
    for count, readline in enumerate(file):
        if count % 1000 == 0:
            print ('making list... ({0}/{1})'.format(count, max_count))
        readline = readline.rstrip()
        readline_sp = readline.split(' ')
        img = cv2.imread(readline_sp[0], -1)
        if img is None:
            print ('failed to load image')
            break
        img = img.astype(np.float32)
        image_list.append(img)
    file.close()
    alph_num = int(readline_sp[1])
    pic_size = img.shape
    print('converting into numpy format...')
    image_list = np.asarray(image_list)
    print('reshaping...')
    image_list = image_list.reshape((image_list.shape[0], pic_size[0], pic_size[1]))
    print('adding new shape...')
    image_list = image_list[:, np.newaxis, :, :]
    return image_list, alph_num

def filelist_to_list_tuple(filename):
    img_label_list = []
    # 行数をゲット
    with open(filename, 'r') as file:
        max_count = sum(1 for line in file)
    file = open(filename, 'r')
    for count, readline in enumerate(file):
        if count % 1000 == 0:
            print ('making list... ({0}/{1})'.format(count, max_count))
        readline = readline.rstrip()
        readline_sp = readline.split(' ')
        img_cv = cv2.imread(readline_sp[0], -1)
        img = np.asarray(img_cv.flatten().astype(np.float32)/255.0)
        # img = img.reshape((img_cv.shape[0], img_cv.shape[1]))
        label = int(readline_sp[1])
        img_label_list.append((img, label))
    file.close()
    return img_label_list

def filelist_to_list(filename):
    image_list, label_list = [], []
    # 行数をゲット
    with open(filename, 'r') as file:
        max_count = sum(1 for line in file)
    file = open(filename, 'r')
    for count, readline in enumerate(file):
        if count % 1000 == 0:
            print ('making list... ({0}/{1})'.format(count, max_count))
        readline = readline.rstrip()
        readline_sp = readline.split(' ')
        img = cv2.imread(readline_sp[0], -1)
        if img is None:
            print ('failed to load image')
            break
        img = img.astype(np.float32)
        img /= 255
        image_list.append(img)
        label = np.int32(readline_sp[1])
        label_list.append(label)
    file.close()
    pic_size = img.shape
    print('converting into numpy format...')
    image_list = np.asarray(image_list)
    label_list = np.asarray(label_list)
    print('reshaping...')
    image_list = image_list.reshape((image_list.shape[0], pic_size[0], pic_size[1]))
    print('adding new shape...')
    image_list = image_list[:, np.newaxis, :, :]
    return image_list, label_list
