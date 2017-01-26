# -*- coding: utf-8 -*-
import random
import dcgan_font
import train_classifier
import models
from mylib import tools

def save_list_include_reject_png_path(dst_dir_path, train_txt_path, test_txt_path, reject_png_dir_path, epoch_i, img_n):
    train_txt_lines, test_txt_lines = [], []
    with open(train_txt_path, 'r') as train_txt_file:
        line = train_txt_file.readline()
        while line:
            train_txt_lines.append(line)
            line = train_txt_file.readline()
    with open(test_txt_path, 'r') as test_txt_file:
        line = test_txt_file.readline()
        while line:
            test_txt_lines.append(line)
            line = test_txt_file.readline()
    for img_i in range(img_n):
        path = reject_png_dir_path + str(img_i) + '.png 26\n'
        if img_i % 10 == 0:
            test_txt_lines.append(path)
        else:   
            train_txt_lines.append(path)
    random.shuffle(train_txt_lines)
    random.shuffle(test_txt_lines)
    with open(dst_dir_path + 'train_' + str(epoch_i) + '.txt', 'w') as new_train_txt_path:
        for line in train_txt_lines:
            new_train_txt_path.write(line)
    with open(dst_dir_path + 'test_' + str(epoch_i) + '.txt', 'w') as new_test_txt_path:
        for line in test_txt_lines:
            new_test_txt_path.write(line)


def train_cdg(dst_dir_path, default_train_all_txt_path, default_test_all_txt_path, train_A_txt_path, epoch_n=10, c_epoch_n=1, gan_epoch_n=2, reject_n=100):
    '''
    Classifierを含めて学習
        まずClassifierを学習し，それに基づきDCGANを学習．
        そのDCGANによる生成画像をn+1クラス目として再度Classifierを学習．
        これを繰り返す．
    Args:
        epoch_n     学習回数．
    '''
    train_all_txt_path=default_train_all_txt_path
    test_all_txt_path=default_test_all_txt_path

    for epoch_i in range(epoch_n):
        train_classifier.train(train_txt_path=train_all_txt_path, test_txt_path=test_all_txt_path, 
                               dst_dir_path=dst_dir_path, is_save_temp=False,
                               last_model_filename='last_' + str(epoch_i), epoch_n=c_epoch_n,
                               model=models.Classifier_AlexNet(class_n=27))
        dcgan_font.train(train_txt_path=train_A_txt_path, dst_dir_path=dst_dir_path, 
                         classifier=models.Classifier_AlexNet(class_n=27), 
                         classifier_hdf5_path=dst_dir_path + 'model_last_' + str(epoch_i) + '.hdf5',
                         classifier_weight=0.01, model_filename=str(epoch_i), is_save_pic=False, 
                         epoch_n=gan_epoch_n, gpu_device=0, save_models_interval=100000)
        dcgan_font.generate(dst_dir_path=tools.make_dir(dst_dir_path + str(epoch_i)),
                            generator_hdf5_path=dst_dir_path + str(epoch_i) + '_model_gen_' + str(gan_epoch_n - 1) + '.hdf5',
                            img_name='', img_num=reject_n, img_font_num=1)
        save_list_include_reject_png_path(dst_dir_path=dst_dir_path,
                                          train_txt_path=train_all_txt_path, 
                                          test_txt_path=test_all_txt_path,
                                          reject_png_dir_path=dst_dir_path + str(epoch_i) + '/',
                                          epoch_i=epoch_i, img_n=reject_n)
        train_all_txt_path = dst_dir_path + 'train_' +  str(epoch_i) + '.txt'
        test_all_txt_path = dst_dir_path + 'test_' +  str(epoch_i) + '.txt'

if __name__ == '__main__':
    train_cdg(tools.make_dir('/home/abe/dcgan_font/cdg/'), '/home/abe/font_dataset/png_6628_64x64/train.txt',
                             '/home/abe/font_dataset/png_6628_64x64/test.txt', 
                             '/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_A.txt',
                             c_epoch_n=10, gan_epoch_n=10000, reject_n=6628)
