# -*- coding:utf-8 -*-
import dcgan_font
from mylib import tools
import models

dcgan_font.train(
    train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_A.txt',
    dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/weightClassifier_0.5_A/'),
    generator=models.Generator_ThreeLayers(z_size=50), 
    discriminator=models.Discriminator_ThreeLayers(),
    classifier=models.Classifier_AlexNet(class_n=26),
    classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
    classifier_weight=0.5,
    gpu_device=1)
