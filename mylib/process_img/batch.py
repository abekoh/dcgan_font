# -*- coding: utf-8 -*-
from __future__ import print_function
import random
from mylib.tools import *
from mylib.cv2_tools import *
import resize_img
import shift_img

def batch(dirpath, findname):
    input_filepaths = get_filepaths(dirpath, findname)
    input_imgs = read_imgs(input_filepaths)

    output_imgs = []
    output_filepath_additions = []
    for input_img in input_imgs:
        output_img = input_img
        # シフト
        r_distance = random.randint(1,20)
        d_distance = random.randint(1,20)
        output_img = shift_img.shift_img(img=output_img,
            direction='r', distance=r_distance)
        output_img = shift_img.shift_img(img=output_img,
            direction='d', distance=d_distance)
        output_filepath_addition = '_shifted(R' + str(r_distance) + 'D' + str(d_distance) + ')'
        output_filepath_additions.append(output_filepath_addition)
        output_imgs.append(output_img)
    
    # output_filepaths = set_filepaths(input_filepaths, '_shifted(4times,max20)')
    output_filepaths = set_special_filepaths(input_filepaths, output_filepath_additions)
    write_imgs(output_imgs, output_filepaths)

def debug(dir_path, findname):
    batch(dir_path, findname)

if __name__ == '__main__':
    batch('../output/results/', '?2?.png')
