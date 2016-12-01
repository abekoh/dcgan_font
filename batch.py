# -*- coding:utf-8 -*-
import dcgan_font
from mylib import tools
import models

dcgan_font.train(
    train_txt_path='/home/abe/font_dataset/png_clustered_1000_64x64/100clust/alph_list/all_A.txt',
    dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/output_1127/100clustered,3layers,zsize50,27class/'),
    generator=models.Generator_ThreeLayers(z_size=50),
    discriminator=models.Discriminator_ThreeLayers(),
    classifier=models.Classifier_AlexNet(class_n=27),
    classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex_27class.hdf5')

dcgan_font.train(
    train_txt_path='/home/abe/font_dataset/png_clustered_1000_64x64/200clust/alph_list/all_A.txt',
    dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/output_1127/200clustered,3layers,zsize50,27class/'),
    generator=models.Generator_ThreeLayers(z_size=50),
    discriminator=models.Discriminator_ThreeLayers(),
    classifier=models.Classifier_AlexNet(class_n=27),
    classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex_27class.hdf5')

dcgan_font.train(
    train_txt_path='/home/abe/font_dataset/png_clustered_1000_64x64/500clust/alph_list/all_A.txt',
    dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/output_1127/500clustered,3layers,zsize50,27class/'),
    generator=models.Generator_ThreeLayers(z_size=50),
    discriminator=models.Discriminator_ThreeLayers(),
    classifier=models.Classifier_AlexNet(class_n=27),
    classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex_27class.hdf5')
