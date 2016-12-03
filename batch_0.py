# -*- coding:utf-8 -*-
import dcgan_font
from mylib import tools
import models

dcgan_font.generate(
        dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/CNN_Test/plusclassifier/'),
        generator=models.Generator_ThreeLayers(z_size=50),
        generator_hdf5_path='/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_A_weight0.01/dcgan_model_gen_9999.hdf5',
        img_name='A',
        img_num=2500,
        img_font_num=1)

dcgan_font.generate(
        dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/CNN_Test/plusclassifier/'),
        generator=models.Generator_ThreeLayers(z_size=50),
        generator_hdf5_path='/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_B_weight0.01/dcgan_model_gen_9999.hdf5',
        img_name='B',
        img_num=2500,
        img_font_num=1)

dcgan_font.generate(
        dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/CNN_Test/plusclassifier/'),
        generator=models.Generator_ThreeLayers(z_size=50),
        generator_hdf5_path='/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_C_weight0.01/dcgan_model_gen_9999.hdf5',
        img_name='C',
        img_num=2500,
        img_font_num=1)

dcgan_font.generate(
        dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/CNN_Test/plusclassifier/'),
        generator=models.Generator_ThreeLayers(z_size=50),
        generator_hdf5_path='/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_D_weight0.01/dcgan_model_gen_9999.hdf5',
        img_name='D',
        img_num=2500,
        img_font_num=1)

