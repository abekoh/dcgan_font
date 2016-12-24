# -*- coding: utf-8 -*-
import numpy as np

def make_ramdom_matrix(dst_txt_path, row_n=100, col_n=50):
    with open(dst_txt_path, 'w') as dst_file:
        for row_i in range(row_n):
            if row_i % 100 == 0:
                print ('{0} / {1}'.format(row_i, row_n))
            if row_i != 0:
                dst_file.write('\n')
            for col_i in range(col_n):
                if col_i != 0:
                    dst_file.write(',')
                dst_file.write(str(np.random.uniform(-1, 1)))

def debug():
    make_ramdom_matrix('/home/abe/dcgan_font/ramdom_matrix.txt', row_n=2000000)

if __name__ == '__main__':
    debug()
