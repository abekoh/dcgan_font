# -*- coding: utf-8 -*-
import os
import glob
from datetime import datetime


def make_dir(path):
    '''
    ディレクトリの存在の確認し，
    存在しなければディレクトリ作成
    -pオプションでまとめてディレクトリ作成
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def correct_dir_path(path):
    '''
    ディレクトリ名を正しく直す
    'hoge/hoge' => 'hoge/hoge/'
    '''
    path_sp = path.split('/')
    if path_sp[-1] == '':
        return path
    return path + '/'


def a2n(alph):
    '''
    アルファベット大文字を数字に
    A => 0, B => 1, ..., Z => 26
    '''
    return ord(alph) - 65


def list2path(ls):
    '''
    リストの要素をパスの形式にして返す
    ['foo', 'bar'] => 'foo/bar/'
    '''
    st = ''
    for l in ls:
        st += str(l) + '/'
    return st


def make_num_dir(path):
    '''
    連番ディレクトリ作成
    作成したディレクトリのパスを返す
    if 'foo/bar0/'が存在しない
      'foo/bar/' => 'foo/bar0/'を作成，パスを返す
    if 'foo/bar0/'が存在
      'foo/bar' => 'foo/bar1'を作成，パスを返す
    '''
    cor_path = correct_dir_path(path)
    cor_path_sp = cor_path.split('/')
    dir_name = cor_path_sp[-2]
    del cor_path_sp[-2:]
    i = -1
    while(True):
        i += 1
        num_path_list = []
        num_path_list = cor_path_sp[:]
        num_path_list.append(dir_name + str(i))
        num_path = list2path(num_path_list)
        if not os.path.exists(num_path):
            os.makedirs(num_path)
            break
    return num_path


def make_date_dir(path):
    cor_path = correct_dir_path(path)
    cor_path_sp = cor_path.split('/')
    dir_name = cor_path_sp[-2]
    del cor_path_sp[-2:]
    date_path_list = cor_path_sp[:]
    now_str = datetime.now().strftime('%Y%m%d%H%M%S')
    date_path_list.append(dir_name + '_' + now_str)
    date_path = list2path(date_path_list)
    os.makedirs(date_path)
    return date_path


def make_alphabets():
    '''
    アルファベット大文字の配列を作成
    => [A, B, ..., Z]
    '''
    alphs = [chr(i + 65) for i in range(26)]
    return alphs


def get_filepaths(dirpath, findname):
    '''
    パスのディレクトリ内のファイルのパスを取得
    findnameでワイルドカード検索可能
    'foo/bar/', '*.png' => 'foo/bar/'内のpng画像のパスのリストを返す
    '''
    cor_dirpath = correct_dir_path(dirpath)
    filepaths = glob.glob(cor_dirpath + findname)
    return filepaths


def set_filepaths(filepaths, addition):
    '''
    ファイル名リストにadditionを追加して返す
    'foo.png', '_bar' => 'foo_bar.png'
    '''
    added_filepaths = []
    for path in filepaths:
        path_sp = path.split('.')
        ext = '.' + path_sp[-1]
        added_path = path.rstrip(ext)
        added_path += addition
        added_path += ext
        added_filepaths.append(added_path)
    return added_filepaths


def set_special_filepaths(filepaths, additions):
    '''
    ファイル名にadditionsリストの順番通り追加して返す
    ['foo.png', 'bar.png'], ['_A', '_B']
      => ['foo_A.png', 'bar_B.png']
    '''
    added_filepaths = []
    for (path, addition) in zip(filepaths, additions):
        path_sp = path.split('.')
        ext = '.' + path_sp[-1]
        added_path = path.rstrip(ext)
        added_path += addition
        added_path += ext
        added_filepaths.append(added_path)
    return added_filepaths


def get_vars_name(var, symboltable, error=None):
    '''
    変数の名前を取得
    http://qiita.com/yuu116atlab/items/154ac83c03de233bf9fb
    '''
    for k, v in symboltable.items():
        if id(v) == id(var):
            return k
    else:
        if error == 'exception':
            raise ValueError('Undefined functions is mixed in subspace?')
        else:
            return error


def get_vars_names(vars, symboltable=locals()):
    '''
    変数(複数)の名前を取得
    http://qiita.com/yuu116atlab/items/154ac83c03de233bf9fb
    '''
    return [get_vars_name(var, symboltable) for var in vars]


def save_vars_log_file(var_names, var_values, dst_txt_path): 
    with open(dst_txt_path, 'w') as log_file:
        for name, value in zip(var_names, var_values):
            tmp = name + '=' + str(value) + '\n'
            log_file.write(tmp)


def save_text_log_file(text, dst_txt_path):
    with open(dst_txt_path, 'a') as log_file:
        log_file.write(text)

