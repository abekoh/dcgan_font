# -*- coding: utf-8 -*-
import numpy as np
import cv2
import shutil

import chainer
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer import functions as F

from mylib.chainer import dataset
from mylib.cv2_tools import display_img
from mylib import tools
from mylib.process_img.combine_imgs import combine_imgs
import models


def save_log_file(var_names, var_values, dst_txt_path): 
    with open(dst_txt_path, 'w') as log_file:
        for name, value in zip(var_names, var_values):
            tmp = name + '=' + str(value) + '\n'
            log_file.write(tmp)

def train(train_txt_path, classifier_hdf5_path, dst_dir_path, 
          generator=models.Generator(), discriminator=models.Discriminator(), 
          classifier=models.Classifier(), 
          epoch_n=10000, batch_size=50, pic_interval=200, save_models_interval=250, 
          adam_alpha=0.0002, adam_beta1=0.5, weight_decay=0.00001):
    org_imgs = dataset.filelist_to_list_for_dcgan(train_txt_path)

    log_values = [generator, discriminator, classifier, classifier_hdf5_path, epoch_n, batch_size, adam_alpha, adam_beta1, weight_decay]
    save_log_file(tools.get_vars_names(log_values, locals()), log_values, dst_dir_path + 'log.txt')
    shutil.copy('./models.py', dst_dir_path + 'models.py')
    shutil.copy('./dcgan_font.py', dst_dir_path + 'dcgan_font.py')

    tools.make_dir(dst_dir_path + 'pic/')

    xp = cuda.cupy
    cuda.get_device(0).use()

    generator.to_gpu()
    discriminator.to_gpu()
    classifier.to_gpu()

    serializers.load_hdf5(classifier_hdf5_path, classifier)

    g_opt = optimizers.Adam(alpha=adam_alpha, beta1=adam_beta1)
    d_opt = optimizers.Adam(alpha=adam_alpha, beta1=adam_beta1)
    g_opt.setup(generator)
    d_opt.setup(discriminator)
    g_opt.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    d_opt.add_hook(chainer.optimizer.WeightDecay(weight_decay))

    for epoch_i in range(epoch_n):
        sum_g_loss = np.float32(0)
        sum_d_loss = np.float32(0)
        batch_n = int(org_imgs.shape[0] / batch_size)

        for batch_i in range(batch_n):
            print('epoch:{0}/{1}, batch:{2}/{3}'.format(epoch_i,
                                                        epoch_n, batch_i, batch_n))
            # generated_imgsで学習
            # ※ 0: original, 1: generated
            z = Variable(
                xp.random.uniform(-1, 1, (batch_size, 100), dtype=np.float32))
            generated_imgs = generator(z)
            generated_d_score = discriminator(generated_imgs)
            g_loss = F.softmax_cross_entropy(
                generated_d_score, Variable(xp.zeros(batch_size, dtype=np.int32)))
            d_loss = F.softmax_cross_entropy(
                generated_d_score, Variable(xp.ones(batch_size, dtype=np.int32)))
            generated_c_score = classifier(generated_imgs)
            g_loss += 0.01 * F.softmax_cross_entropy(
                generated_c_score, Variable(xp.zeros(batch_size, dtype=np.int32)))
            acc = F.accuracy(
                generated_c_score, Variable(xp.zeros(batch_size, dtype=np.int32)))
            print(acc.data)

            # original_imgsで学習
            batched_org_imgs = np.zeros(
                (batch_size, 1, 64, 64), dtype=np.float32)
            for i, j in enumerate(range(batch_i * batch_size, (batch_i + 1) * batch_size)):
                batched_org_imgs[i, :, :, :] = (org_imgs[j] - 127.5) / 127.5
            batched_org_imgs = Variable(cuda.to_gpu(batched_org_imgs))
            original_d_score = discriminator(batched_org_imgs)
            d_loss += F.softmax_cross_entropy(original_d_score,
                                              Variable(xp.zeros(batch_size, dtype=np.int32)))

            g_opt.zero_grads()
            g_loss.backward()
            g_opt.update()

            d_opt.zero_grads()
            d_loss.backward()
            d_opt.update()

            g_loss_data = g_loss.data.get()
            d_loss_data = d_loss.data.get()

            print('g_loss:{0}, d_loss:{1}'.format(g_loss_data, d_loss_data))

            sum_g_loss += g_loss_data
            sum_d_loss += d_loss_data

            if batch_i % pic_interval == 0:
                z = Variable(
                    xp.random.uniform(-1, 1, (100, 100), dtype=np.float32))
                generated_imgs = generator(z, test=True)
                generated_imgs = generated_imgs.data.get()
                g_imgs = []
                for g_img in generated_imgs:
                    g_imgs.append(g_img[0])
                combined_img = combine_imgs(g_imgs, 10)
                combined_img = combined_img * 127.5 + 127.5
                combined_img = combined_img.astype(np.int32)
                cv2.imwrite('{0}pic/{1}_{2}.png'.format(dst_dir_path,
                                                        epoch_i, batch_i), combined_img)
        if (epoch_i % save_models_interval == 0 and epoch_i != 0) or epoch_i == epoch_n - 1:
            serializers.save_hdf5("{0}dcgan_model_dis_{1}.hdf5".format(
                dst_dir_path, epoch_i), discriminator)
            serializers.save_hdf5("{0}dcgan_model_gen_{1}.hdf5".format(
                dst_dir_path, epoch_i), generator)
            serializers.save_hdf5("{0}dcgan_state_dis_{1}.hdf5".format(
                dst_dir_path, epoch_i), d_opt)
            serializers.save_hdf5("{0}dcgan_state_gen_{1}.hdf5".format(
                dst_dir_path, epoch_i), g_opt)


def generate_10x10(generator_hdf5_path):
    xp = np
    generator = models.Generator()
    serializers.load_hdf5(generator_hdf5_path, generator)
    z = (xp.random.uniform(-1, 1, (100, 100)).astype(np.float32))
    z = Variable(z)
    generated_imgs = generator(z, test=True)
    generated_imgs = generated_imgs.data
    g_imgs = []
    for g_img in generated_imgs:
        g_imgs.append(g_img[0])
    combined_img = combine_imgs(g_imgs, 10)
    combined_img = combined_img * 127.5 + 127.5
    combined_img = combined_img.astype(np.int32)
    cv2.imwrite('generated.png', combined_img)

def generate_1(generator_hdf5_path, num=10):
    xp = np
    generator = models.Generator()
    serializers.load_hdf5(generator_hdf5_path, generator)
    for i in range(num):
        z = (xp.random.uniform(-1, 1, (1, 100)).astype(np.float32))
        z = Variable(z)
        generated_imgs = generator(z, test=True)
        generated_imgs = generated_imgs.data
        generated_img = generated_imgs[0][0]
        generated_img = generated_img * 127.5 + 127.5
        generated_img = generated_img.astype(np.int32)
        cv2.imwrite('generated/generated_' + str(i) + '.png', generated_img)

def debug():
    train_txt_path = '/home/abe/font_dataset/png_selected_184_64x64/alph_list/all_A.txt'
    classifier_hdf5_path = '/home/abe/dcgan_font/classifier_alex.hdf5'
    train(train_txt_path, classifier_hdf5_path, tools.make_date_dir(
        '/home/abe/dcgan_font/output/debug/'))
    # generate_1('/home/abe/dcgan_font/output/+classifier_0.01/dcgan_model_gen_950.hdf5', 100)


if __name__ == '__main__':
    debug()
