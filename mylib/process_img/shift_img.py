# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import cv2
import argparse

from mylib.cv2_tools import *


def shift_img(img, direction, distance):
    if direction == 'l':
        shifted_img = np.hstack((img[:, distance:], img[:, 0:distance]))
        return shifted_img
    if direction == 'r':
        end = len(img[0])
        shifted_img = np.hstack((img[:, end - distance:end], img[:, 0:end- distance]))
        return shifted_img
    if direction == 'u':
        shifted_img = np.vstack((img[distance:, :], img[0:distance, :]))
        return shifted_img
    if direction == 'd':
        end = len(img)
        shifted_img = np.vstack((img[end - distance:end,:], img[:end - distance, :]))
        return shifted_img
    print ('Error: 方向が間違っています')


def save_shifted_img(src_img_path, direction, distance, dst_img_path):
    src_img = cv2.imread(src_img_path, -1)
    dst_img = shift_img(src_img, direction, distance)
    # display_img(dst_img)
    cv2.imwrite(dst_img_path, dst_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='画像を上下左右にシフトさせる')
    parser.add_argument('src_img_path', action='store', nargs=None, type=str, choices=None, \
            help='入力画像のパス')
    parser.add_argument('direction', action='store', nargs=None, type=str, choices=None, \
            help='シフトさせる方向')
    parser.add_argument('distance', action='store', nargs=None, type=int, choices=None, \
            help='移動させる距離')
    parser.add_argument('dst_img_path', action='store', nargs=None, type=str, choices=None, \
            help='出力画像のパス')
    args = parser.parse_args()
    save_shifted_img(args.src_img_path, args.direction, args.distance, args.dst_img_path)
