# -*- coding:utf-8 -*-
import dcgan_font
from mylib import tools
import models

for i in range(500, 10000, 500):
    dcgan_font.generate(
        dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/noClassifier_A/generated_proccess/'),
        generator=models.Generator_ThreeLayers(z_size=50),
        generator_hdf5_path='/home/abe/dcgan_font/output_storage/forPRMU/noClassifier_A/dcgan_model_gen_' + str(i) + '.hdf5',
        img_name=str(i),
        img_num=10,
        img_font_num=25)

