# -*- coding:utf-8 -*-
import dcgan_font
from mylib import tools
from mylib import cv2_tools
import models
from mylib.process_img import combine_imgs
import cv2
import random

dcgan_font.train(
    train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_D.txt',
    dst_dir_path=tools.make_date_dir('/home/abe/dcgan_font/output_storage/sorted_random_matrix_D/'),
    generator=models.Generator_ThreeLayers(z_size=50),
    discriminator=models.Discriminator_ThreeLayers(),
    classifier=models.Classifier_AlexNet(class_n=26),
    classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
    classifier_weight=0.01,
    random_matrix_txt='/home/abe/dcgan_font/ramdom_matrix.txt',
    gpu_device=0)
