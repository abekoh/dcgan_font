# -*- coding:utf-8 -*-
import dcgan_font
from mylib import tools
import models

# # Deduplication無し
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_6628_64x64/alph_list/all_A.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/noClassifier_noDeduplication_A/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers())

#classifier勾配の変化
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_A.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/weightClassifier_0.005_A/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers(),
#     classifier=models.Classifier_AlexNet(class_n=26),
#     classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
#     classifier_weight=0.005)
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_A.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/weightClassifier_0.05_A/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers(),
#     classifier=models.Classifier_AlexNet(class_n=26),
#     classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
#     classifier_weight=0.05)
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_A.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/weightClassifier_0.1_A/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers(),
#     classifier=models.Classifier_AlexNet(class_n=26),
#     classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
#     classifier_weight=0.1)
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_A.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/weightClassifier_0.05_A/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers(),
#     classifier=models.Classifier_AlexNet(class_n=26),
#     classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
#     classifier_weight=0.05)
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_A.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/weightClassifier_0.001_A/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers(),
#     classifier=models.Classifier_AlexNet(class_n=26),
#     classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
#     classifier_weight=0.001)
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_A.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/weightClassifier_0.0075_A/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers(),
#     classifier=models.Classifier_AlexNet(class_n=26),
#     classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
#     classifier_weight=0.0075)

#classifier無し
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_A.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/noClassifier_A/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers())
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_B.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/noClassifier_B/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers())
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_C.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/noClassifier_C/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers())
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_D.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/noClassifier_D/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers())

#classifier有り
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_A.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_A_weight0.01/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers(),
#     classifier=models.Classifier_AlexNet(class_n=26),
#     classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
#     classifier_weight=0.01)
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_B.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_B_weight0.01/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers(),
#     classifier=models.Classifier_AlexNet(class_n=26),
#     classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
#     classifier_weight=0.01)
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_C.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_C_weight0.01/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers(),
#     classifier=models.Classifier_AlexNet(class_n=26),
#     classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
#     classifier_weight=0.01)
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_D.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_D_weight0.01/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers(),
#     classifier=models.Classifier_AlexNet(class_n=26),
#     classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
#     classifier_weight=0.01)
#
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_A.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_A_weight0.02/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers(),
#     classifier=models.Classifier_AlexNet(class_n=26),
#     classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
#     classifier_weight=0.02)
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_B.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_B_weight0.02/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers(),
#     classifier=models.Classifier_AlexNet(class_n=26),
#     classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
#     classifier_weight=0.02)
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_C.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_C_weight0.02/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers(),
#     classifier=models.Classifier_AlexNet(class_n=26),
#     classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
#     classifier_weight=0.02)
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_D.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_D_weight0.02/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers(),
#     classifier=models.Classifier_AlexNet(class_n=26),
#     classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
#     classifier_weight=0.02)
#
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_A.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_A_weight0.05/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers(),
#     classifier=models.Classifier_AlexNet(class_n=26),
#     classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
#     classifier_weight=0.05)
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_B.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_B_weight0.05/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers(),
#     classifier=models.Classifier_AlexNet(class_n=26),
#     classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
#     classifier_weight=0.05)
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_C.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_C_weight0.05/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers(),
#     classifier=models.Classifier_AlexNet(class_n=26),
#     classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
#     classifier_weight=0.05)
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_D.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_D_weight0.05/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers(),
#     classifier=models.Classifier_AlexNet(class_n=26),
#     classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
#     classifier_weight=0.05)

dcgan_font.train(
    train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_A.txt',
    dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_A_weight0.03/'),
    generator=models.Generator_ThreeLayers(z_size=50), 
    discriminator=models.Discriminator_ThreeLayers(),
    classifier=models.Classifier_AlexNet(class_n=26),
    classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
    classifier_weight=0.03)
dcgan_font.train(
    train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_B.txt',
    dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_B_weight0.03/'),
    generator=models.Generator_ThreeLayers(z_size=50), 
    discriminator=models.Discriminator_ThreeLayers(),
    classifier=models.Classifier_AlexNet(class_n=26),
    classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
    classifier_weight=0.03)
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_C.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_C_weight0.03/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers(),
#     classifier=models.Classifier_AlexNet(class_n=26),
#     classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
#     classifier_weight=0.03)
# dcgan_font.train(
#     train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_D.txt',
#     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/plusClassifier_D_weight0.03/'),
#     generator=models.Generator_ThreeLayers(z_size=50), 
#     discriminator=models.Discriminator_ThreeLayers(),
#     classifier=models.Classifier_AlexNet(class_n=26),
#     classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
#     classifier_weight=0.03)
