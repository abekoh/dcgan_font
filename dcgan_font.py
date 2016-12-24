# -*- coding: utf-8 -*-
import numpy as np
import cv2
import shutil 
import chainer
import math
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer import functions as F

from mylib.chainer import dataset
from mylib.cv2_tools import display_img
from mylib import tools
from mylib import stopwatch
from mylib.process_img.combine_imgs import combine_imgs
import models


def train(train_txt_path, dst_dir_path, 
          generator, discriminator, classifier=None, classifier_hdf5_path='', 
          classifier_weight=0, gpu_device=0, 
          epoch_n=10000, batch_size=100, pic_interval=200, save_models_interval=500, 
          opt='Adam', sgd_lr=0.0002,
          adam_alpha=0.0002, adam_beta1=0.5, weight_decay=0.00001):

    tools.make_dir(dst_dir_path + 'pic/')

    dst_log_txt_path = dst_dir_path + 'log.txt'
    log_values = [
        train_txt_path, dst_dir_path, generator, discriminator, classifier, classifier_hdf5_path, 
        classifier_weight, epoch_n, batch_size, opt, sgd_lr, adam_alpha, adam_beta1, weight_decay]
    tools.save_vars_log_file(tools.get_vars_names(log_values, locals()), log_values, dst_dir_path + 'log.txt')
    shutil.copy('./models.py', dst_dir_path + 'models.py')
    shutil.copy('./dcgan_font.py', dst_dir_path + 'dcgan_font.py')

    org_imgs, alph_num = dataset.filelist_to_list_for_dcgan(train_txt_path)

    xp = cuda.cupy
    cuda.get_device(gpu_device).use()
    generator.to_gpu(gpu_device)
    discriminator.to_gpu(gpu_device)

    if classifier is not None:
        classifier.to_gpu(gpu_device)
        serializers.load_hdf5(classifier_hdf5_path, classifier)

    if opt == 'SGD':
        print ('use SGD')
        g_opt = optimizers.SGD(lr=sgd_lr)
        d_opt = optimizers.SGD(lr=sgd_lr)
    else:
        g_opt = optimizers.Adam(alpha=adam_alpha, beta1=adam_beta1)
        d_opt = optimizers.Adam(alpha=adam_alpha, beta1=adam_beta1)
    g_opt.setup(generator)
    d_opt.setup(discriminator)
    g_opt.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    d_opt.add_hook(chainer.optimizer.WeightDecay(weight_decay))

    sp = stopwatch.StopWatch()
    sp.start()
    
    for epoch_i in range(epoch_n):
        sum_g_loss = np.float32(0)
        sum_d_loss = np.float32(0)
        batch_n = int(org_imgs.shape[0] / batch_size)

        for batch_i in range(batch_n):
            print ('epoch:{0}/{1}, batch:{2}/{3}'.format(epoch_i,
                                                        epoch_n, batch_i, batch_n))
            # generated_imgsで学習
            # 0: original, 1: generated
            z = Variable(cuda.to_gpu(
                xp.random.uniform(-1, 1, (batch_size, generator.z_size), dtype=np.float32), gpu_device))
            generated_imgs = generator(z)
            generated_d_score = discriminator(generated_imgs)
            g_loss = (1.0 - classifier_weight) * F.softmax_cross_entropy(
                generated_d_score, Variable(xp.zeros(batch_size, dtype=np.int32)))
            d_loss = F.softmax_cross_entropy(
                generated_d_score, Variable(xp.ones(batch_size, dtype=np.int32)))
            if classifier is not None:
                generated_c_score = classifier(generated_imgs)
                g_loss += classifier_weight * F.softmax_cross_entropy(
                    generated_c_score, Variable(xp.ones(batch_size, dtype=np.int32) * alph_num))
                acc = F.accuracy(
                    generated_c_score, Variable(xp.ones(batch_size, dtype=np.int32) * alph_num))
                print ('accuracy_rate:', acc.data.get())

            # original_imgsで学習
            batched_org_imgs = np.zeros(
                (batch_size, 1, 64, 64), dtype=np.float32)
            for i, j in enumerate(range(batch_i * batch_size, (batch_i + 1) * batch_size)):
                batched_org_imgs[i, :, :, :] = (org_imgs[j] - 127.5) / 127.5
            batched_org_imgs = Variable(cuda.to_gpu(batched_org_imgs, gpu_device))
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
                    xp.random.uniform(-1, 1, (100, generator.z_size), dtype=np.float32))
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
            sp.view()
        if (epoch_i % save_models_interval == 0 and epoch_i != 0) or epoch_i == epoch_n - 1:
            serializers.save_hdf5("{0}dcgan_model_dis_{1}.hdf5".format(
                dst_dir_path, epoch_i), discriminator)
            serializers.save_hdf5("{0}dcgan_model_gen_{1}.hdf5".format(
                dst_dir_path, epoch_i), generator)
            serializers.save_hdf5("{0}dcgan_state_dis_{1}.hdf5".format(
                dst_dir_path, epoch_i), d_opt)
            serializers.save_hdf5("{0}dcgan_state_gen_{1}.hdf5".format(
                dst_dir_path, epoch_i), g_opt)
    sp.stop()
    tools.save_text_log_file(sp.format_print(), dst_log_txt_path)


def generate(dst_dir_path,
             generator, generator_hdf5_path,
             img_name='generated', 
             img_num=10, img_font_num=100):
    print ('generate fonts at:', dst_dir_path)
    xp = np
    serializers.load_hdf5(generator_hdf5_path, generator)
    for img_i in range(img_num):
        print ('img: {0}/{1}'.format(img_i, img_num))
        z = (xp.random.uniform(-1, 1, (img_font_num, generator.z_size)).astype(np.float32))
        z = Variable(z)
        generated_imgs = generator(z, test=True)
        generated_imgs = generated_imgs.data
        g_imgs = []
        for g_img in generated_imgs:
            g_imgs.append(g_img[0])
        combined_img = combine_imgs(g_imgs, int(math.sqrt(img_font_num)))
        combined_img = combined_img * 127.5 + 127.5
        combined_img = combined_img.astype(np.int32)
        cv2.imwrite(dst_dir_path + img_name + '_' + str(img_i) + '.png', combined_img)


def debug():
    train(
        train_txt_path='/home/abe/font_dataset/png_selected_200_64x64/alph_list/all_A.txt',
        dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/nopooling/addC_A/'),
        generator=models.Generator_ThreeLayers_NoPooling(z_size=50),
        discriminator=models.Discriminator_ThreeLayers_NoPooling(),
        classifier=models.Classifier_AlexNet(class_n=26),
        classifier_hdf5_path='/home/abe/dcgan_font/classificator_alex.hdf5',
        gpu_device=1)

    # generate(
    #     dst_dir_path=tools.make_dir('/home/abe/dcgan_font/output_storage/forPRMU/noClassifier_A/generated_ones/'),
    #     generator=models.Generator_ThreeLayers(z_size=50),
    #     generator_hdf5_path='/home/abe/dcgan_font/output_storage/forPRMU/noClassifier_A/dcgan_model_gen_9999.hdf5',
    #     img_name='noclassifier_A',
    #     img_num=1000,
    #     img_font_num=1)


if __name__ == '__main__':
    debug()


