# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import cv2
import argparse

from mylib.cv2_tools import *

def resize_img(img, mag):
    height = img.shape[0]
    width = img.shape[1]
    resized_size = (int(height * mag), int(width * mag))
    resized_img = cv2.resize(img, resized_size, interpolation=cv2.INTER_NEAREST)
    return resized_img

def save_resized_img(src_img_path, mag, dst_img_path):
    src_img = cv2.imread(src_img_path, -1)
    dst_img = resize_img(src_img, mag)
    # display_img(dst_img)
    cv2.imwrite(dst_img_path, dst_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='画像を上下左右にシフトさせる')
    parser.add_argument('src_img_path', action='store', nargs=None, type=str, choices=None, \
            help='入力画像のパス')
    parser.add_argument('mag', action='store', nargs=None, type=float, choices=None, \
            help='倍率')
    parser.add_argument('dst_img_path', action='store', nargs=None, type=str, choices=None, \
            help='出力画像のパス')
    args = parser.parse_args()
    save_resized_img(args.src_img_path, args.mag, args.dst_img_path)
