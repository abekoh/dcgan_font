# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import cv2
import argparse
from mylib.tools import *
from mylib.cv2_tools import *

def combine_imgs(imgs, width):
    if len(imgs) % width == 0:
        height = int(len(imgs) / width)
    else:
        height = int(len(imgs) / width + 1)

    img_size = imgs[0].shape

    white_img = np.zeros(img_size, dtype=np.uint8)
    white_img.fill(255)

    for row_i in range(height):
        for col_i in range(width):
            if col_i == 0:
                combined_row_img = imgs.pop(0)
                continue
            if len(imgs) == 0:
                combined_row_img = cv2.hconcat([combined_row_img, white_img])
                continue
            combined_row_img = cv2.hconcat([combined_row_img, imgs.pop(0)])
        if row_i == 0:
            combined_img = combined_row_img
            continue
        combined_img = cv2.vconcat([combined_img, combined_row_img])
    return combined_img

def save_combined_img(src_dir_path, width, dst_img_path):
    src_imgs = img_paths_to_mats(src_dir_path)
    combined_img = combine_imgs(src_imgs, width)
    cv2.imwrite(dst_img_path, combined_img)

def debug():
    imgs = img_paths_to_mats('/home/abe/ga_font/output_debug/layers/5/')
    combined_img = combine_imgs(imgs, 25)
    display_img(combined_img)
    cv2.imwrite('/home/abe/ga_font/output_debug/layers/5.png', combined_img)

def user_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('src_dir_path', action='store', type=str)
    parser.add_argument('width', action='store', type=int)
    parser.add_argument('dst_img_path', action='store', type=str)
    args = parser.parse_args()
    save_combined_img(correct_dir_path(args.src_dir_path), args.width, args.dst_img_path)

if __name__ == '__main__':
    # debug()
    user_input()
