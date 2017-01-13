# -*- coding: utf-8 -*-
import cv2
import os

def display_img(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows

def img_path_to_mat(path):
    img  = cv2.imread(path, -1)
    return img

def read_imgs(filepaths):
    imgs = []
    for path in filepaths:
        img = cv2.imread(path, -1)
        imgs.append(img)
    return imgs

def write_imgs(imgs, filepaths):
    for img, path in zip(imgs, filepaths):
        cv2.imwrite(path, img)

def img_paths_to_mats(dir_path, head='', ext='png'):
    imgs = []
    i = 0
    while(1):
        path = dir_path + head + str(i) + '.' + ext
        if not os.path.isfile(path):
            break
        img = cv2.imread(path, -1)
        imgs.append(img)
        i += 1
    return imgs
