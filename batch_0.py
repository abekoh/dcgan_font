# -*- coding:utf-8 -*-
import dcgan_font
from mylib import tools
from mylib import cv2_tools
import models
from mylib.process_img import combine_imgs
import cv2
import random

# for i in range(500, 10000, 500):
#     dcgan_font.generate(
#         dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_A_weight0.05/generated_proccess/'),
#         generator=models.Generator_ThreeLayers(z_size=50),
#         generator_hdf5_path='/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_A_weight0.05/dcgan_model_gen_' + str(i) + '.hdf5',
#         img_name=str(i),
#         img_num=10,
#         img_font_num=25)
#
# dcgan_font.generate(
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_A_weight0.05/generated/'),
#     generator=models.Generator_ThreeLayers(z_size=50),
#     generator_hdf5_path='/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_A_weight0.05/dcgan_model_gen_9999.hdf5',
#     img_num=10,
#     img_font_num=25)
#
# dcgan_font.generate(
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_B_weight0.05/generated/'),
#     generator=models.Generator_ThreeLayers(z_size=50),
#     generator_hdf5_path='/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_B_weight0.05/dcgan_model_gen_9999.hdf5',
#     img_num=10,
#     img_font_num=25)
#
# dcgan_font.generate(
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_C_weight0.05/generated/'),
#     generator=models.Generator_ThreeLayers(z_size=50),
#     generator_hdf5_path='/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_C_weight0.05/dcgan_model_gen_9999.hdf5',
#     img_num=10,
#     img_font_num=25)
#
# dcgan_font.generate(
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_D_weight0.05/generated/'),
#     generator=models.Generator_ThreeLayers(z_size=50),
#     generator_hdf5_path='/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_D_weight0.05/dcgan_model_gen_9999.hdf5',
#     img_num=10,
#     img_font_num=25)

# filepaths = tools.get_filepaths('/home/abe/font_dataset/png_selected_200_64x64/A/', '*.png')
# imgs = []
# for path in filepaths:
#     img = cv2.imread(path, -1)
#     imgs.append(img)
#
#
# combined_img = combine_imgs.combine_imgs(imgs=imgs, width=20)
#
# cv2_tools.display_img(combined_img)
# cv2.imwrite('/home/abe/dcgan_font/output_storage/forPRMU/pics/200selected.png', combined_img)

for j in range(20):
    imgs = []
    for alph in ['A', 'B', 'C', 'D']:
        for i in range(20):
            img = cv2.imread('/home/abe/font_dataset/png_6628_64x64/' + alph + '/' + str(random.randint(0, 6627)) + '.png')
            imgs.append(img)
    
    combined_img = combine_imgs.combine_imgs(imgs=imgs, width=10)
    cv2.imwrite('/home/abe/dcgan_font/output_storage/forPRMU/pics/fontexamples' + str(j) + '.png', combined_img)

