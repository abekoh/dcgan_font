# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
from keras.optimizers import SGD
from mylib.keras.networks.lenet import LeNet
from mylib.keras import dataset

def train_model(train_txt_path, test_txt_path):
    train_imgs, train_labels = dataset.filelist_to_list(train_txt_path)

    print('compiling model...')
    opt = SGD(lr=0.01)
    model = LeNet.build(width=64, height=64, depth=1, classes=26, weight_path=None)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    print('training...')
    model.fit(train_imgs, train_labels, batch_size=128, nb_epoch=2, verbose=1)

    if len(test_txt_path) > 0:
        print('testing...')
        test_img, test_labels = dataset.filelist_to_list(test_txt_path)
        (loss, accuracy) = model.evaluate(test_img, test_labels, batch_size=128, verbose=1)
        print('accuracy: {:.2f}%'.format(accuracy * 100))

    return model

def save_model(model, output_hdf_path):
    print('saving...')
    model.save_weights(output_hdf_path, overwrite=True)

def user_input():
    ap = argparse.ArgumentParser()
    ap.add_argument('train_txt_path', type=str, help='train.txt')
    ap.add_argument('-T', '--test-txt-path', type=str, default='', help='test.txt')
    ap.add_argument('output_hdf_path', type=str, help='出力ファイル')
    args = ap.parse_args()

    model = train_model(args.train_txt_path, args.test_txt_path)
    save_model(model, args.output_hdf_path)

def debug():
    model = train_model('/home/abe/font_dataset/png_6628_64x64/train.txt', \
                        '/home/abe/font_dataset/png_6628_64x64/test.txt')
    save_model(model, './trained.hdf5')

if __name__ == '__main__':
    # user_input()
    debug()
