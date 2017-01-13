# dcgan_font
DCGANを用いてフォントを生成する試み．

# 特徴
DCGANに文字を分類するCNNを組み込むことで，
従来のより文字らしいフォントが生成されるようになる．
<img src='https://github.com/abekoh/dcgan_font/blob/master/slide/slide_1.png' height='240px'>
<img src='https://github.com/abekoh/dcgan_font/blob/master/slide/slide_2.png' height='240px'>
<img src='https://github.com/abekoh/dcgan_font/blob/master/slide/slide_3.png' height='240px'>
<img src='https://github.com/abekoh/dcgan_font/blob/master/slide/slide_4.png' height='240px'>

## 実験環境
OS，GPU，ライブラリ，フレームワークなど
* Ubuntu 16.04
* GeForce GTX 1080 * 2
* CUDA 8.0
* CuDNN 5.1
* Python 3.5.2
* chainer 1.17.0
* Numpy 1.11.2
* OpenCV 3.1.0
* h5py 2.6.0

## とりあえずAのフォントを生成してみる
`python dcgan_font.py --mode generate`

ディレクトリ`output`に生成画像が出力されます．

## 使い方
### 学習

GPU必要．

`python dcgan_font.py --mode train --traintxt <学習する画像のパス，ラベルのリスト> --dst <出力先ディレクトリ>`

学習に用いる画像のパスを記載したtxtを入力とする．

1列目は画像パス，2列目はクラスID(Aから順に0,1,2...)
Aのフォントを生成する場合は全部0となる．
```txt:train.txt
/home/hoge/font/A/0.png, 0
/home/hoge/font/A/1.png, 0
/home/hoge/font/A/2.png, 0
```
使用する画像は64x64の2値画像．

### 画像生成

GPU不要．

`python dcgan_font.py --mode generate --trainedg <Generatorの学習済みモデルのパス> --dst <出力先ディレクトリ>`

サンプルとして`trained_model`の`generator_A.hdf5`を使用可能．

## 結果
<img src='https://github.com/abekoh/dcgan_font/blob/develop/example/addClassifier_A.png'>
<img src='https://github.com/abekoh/dcgan_font/blob/develop/example/addClassifier_B.png'>
<img src='https://github.com/abekoh/dcgan_font/blob/develop/example/addClassifier_C.png'>
<img src='https://github.com/abekoh/dcgan_font/blob/develop/example/addClassifier_D.png'>

## 参考文献
* Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks 

  https://arxiv.org/abs/1511.06434

* chainer-DCGAN

  https://github.com/mattya/chainer-DCGAN
