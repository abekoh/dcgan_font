# -*- coding: utf-8 -*-
import chainer
from chainer import functions as F 
from chainer import links as L
'''
Generator, Discriminator, Classifierの定義
'''

# 元論文ImageNet
class Generator_ForImageNet(chainer.Chain):
    def __init__(self):
        super(Generator_ForImageNet, self).__init__(
            fc5=L.Linear(100, 512 * 4 * 4),
            norm5=L.BatchNormalization(512 * 4 * 4),
            conv4=L.Deconvolution2D(512, 256, ksize=4, stride=2, pad=1),
            norm4=L.BatchNormalization(256),
            conv3=L.Deconvolution2D(256, 128, ksize=4, stride=2, pad=1),
            norm3=L.BatchNormalization(128),
            conv2=L.Deconvolution2D(128, 64,  ksize=4, stride=2, pad=1),
            norm2=L.BatchNormalization(64),
            conv1=L.Deconvolution2D(64, 1, ksize=4, stride=2, pad=1))

    def __call__(self, z, test=False):
        n_sample = z.data.shape[0]
        h = F.relu(self.norm5(self.fc5(z), test=test))
        h = F.reshape(h, (n_sample, 512, 4, 4))
        h = F.relu(self.norm4(self.conv4(h), test=test))
        h = F.relu(self.norm3(self.conv3(h), test=test))
        h = F.relu(self.norm2(self.conv2(h), test=test))
        x = F.tanh(self.conv1(h))
        return x

class Discriminator_ForImageNet(chainer.Chain):
    def __init__(self):
        super(Discriminator_ForImageNet, self).__init__(
            conv1=L.Convolution2D(1,   64,  ksize=4, stride=2, pad=1),
            conv2=L.Convolution2D(64,  128, ksize=4, stride=2, pad=1),
            norm2=L.BatchNormalization(128),
            conv3=L.Convolution2D(128, 256, ksize=4, stride=2, pad=1),
            norm3=L.BatchNormalization(256),
            conv4=L.Convolution2D(256, 512, ksize=4, stride=2, pad=1),
            norm4=L.BatchNormalization(512),
            fc5=L.Linear(512 * 4 * 4, 2))

    def __call__(self, x, test=False):
        n_sample = x.data.shape[0]
        h = F.leaky_relu(self.conv1(x))
        h = F.leaky_relu(self.norm2(self.conv2(h), test=test))
        h = F.leaky_relu(self.norm3(self.conv3(h), test=test))
        h = F.leaky_relu(self.norm4(self.conv4(h), test=test))
        y = self.fc5(h)
        return y

class Generator_ManyLayers(chainer.Chain):
    def __init__(self):
        super(Generator_ManyLayers, self).__init__(
            fc8=L.Linear(100, 1024),
            fc7=L.Linear(1024, 1024 * 2 * 2),
            norm6=L.BatchNormalization(1024 * 2 * 2),
            conv5=L.Deconvolution2D(1024, 512, ksize=4, stride=1, pad=1),
            norm5=L.BatchNormalization(512),
            conv4=L.Deconvolution2D(512, 256, ksize=4, stride=1, pad=1),
            norm4=L.BatchNormalization(256),
            conv3=L.Deconvolution2D(256, 128, ksize=4, stride=1, pad=1),
            norm3=L.BatchNormalization(128),
            conv2=L.Deconvolution2D(128, 64,  ksize=4, stride=1, pad=1),
            norm2=L.BatchNormalization(64),
            conv1=L.Deconvolution2D(64, 1, ksize=4, stride=1, pad=1))

    def __call__(self, z, test=False):
        h = F.tanh(self.fc8(z))
        h = F.tanh(self.norm6(self.fc7(h), test=test))
        h = F.reshape(h, (100, 1024, 2, 2))
        h = F.tanh(self.norm5(self.conv5(F.unpooling_2d(h, ksize=2)), test=test))
        h = F.tanh(self.norm4(self.conv4(F.unpooling_2d(h, ksize=2)), test=test))
        h = F.tanh(self.norm3(self.conv3(F.unpooling_2d(h, ksize=2)), test=test))
        h = F.tanh(self.norm2(self.conv2(F.unpooling_2d(h, ksize=2)), test=test))
        x = F.tanh(self.conv1(F.unpooling_2d(h, ksize=2)))
        return x

class Discriminator_ManyLayers(chainer.Chain):
    def __init__(self):
        super(Discriminator_ManyLayers, self).__init__(
            conv1=L.Convolution2D(1, 64, ksize=4, stride=1, pad=1),
            norm1=L.BatchNormalization(64),
            conv2=L.Convolution2D(64, 128, ksize=4, stride=1, pad=1),
            norm2=L.BatchNormalization(128),
            conv3=L.Convolution2D(128, 256, ksize=4, stride=1, pad=1),
            norm3=L.BatchNormalization(256),
            conv4=L.Convolution2D(256, 512, ksize=4, stride=1, pad=1),
            norm4=L.BatchNormalization(512),
            conv5=L.Convolution2D(512, 1024, ksize=4, stride=1, pad=1),
            norm5=L.BatchNormalization(1024),
            fc5=L.Linear(1024*2*2, 2))

    def __call__(self, x, test=False):
        h = F.max_pooling_2d(F.tanh(self.norm1(self.conv1(x), test=test)), ksize=2)
        h = F.max_pooling_2d(F.tanh(self.norm2(self.conv2(h), test=test)), ksize=2)
        h = F.max_pooling_2d(F.tanh(self.norm3(self.conv3(h), test=test)), ksize=2)
        h = F.max_pooling_2d(F.tanh(self.norm4(self.conv4(h), test=test)), ksize=2)
        h = F.max_pooling_2d(F.tanh(self.norm5(self.conv5(h), test=test)), ksize=2)
        y = F.sigmoid(self.fc5(h))
        return y

class Generator_FourLayers(chainer.Chain):
    def __init__(self, z_size=100):
        self.z_size = z_size
        super(Generator_FourLayers, self).__init__(
            fc1 = L.Linear(self.z_size, 1024),
            fc2 = L.Linear(1024, 512*4*4),
            norm2 = L.BatchNormalization(512*4*4),
            conv3 = L.Deconvolution2D(512, 256, ksize=4, stride=1, pad=1),
            conv4 = L.Deconvolution2D(256, 128, ksize=4, stride=1, pad=1),
            conv5 = L.Deconvolution2D(128, 64, ksize=4, stride=1, pad=1),
            conv6 = L.Deconvolution2D(64, 1, ksize=4, stride=1, pad=1))

    def __call__(self, z, test=False):
        h = F.tanh(self.fc1(z))
        h = F.tanh(self.norm2(self.fc2(h), test=test))
        h = F.reshape(h, (self.z_size, 512, 4, 4))
        h = F.tanh(self.conv3(F.unpooling_2d(h, ksize=2)))
        h = F.tanh(self.conv4(F.unpooling_2d(h, ksize=2)))
        h = F.tanh(self.conv5(F.unpooling_2d(h, ksize=2)))
        x = F.tanh(self.conv6(F.unpooling_2d(h, ksize=2)))
        return x

class Discriminator_FourLayers(chainer.Chain):
    def __init__(self):
        super(Discriminator_FourLayers, self).__init__(
            conv1 = L.Convolution2D(1, 64, ksize=4, stride=1, pad=1),
            conv2 = L.Convolution2D(64, 128, ksize=4, stride=1, pad=1),
            conv3 = L.Convolution2D(128, 256, ksize=4, stride=1, pad=1),
            conv4 = L.Convolution2D(256, 512, ksize=4, stride=1, pad=1),
            fc5 = L.Linear(512*4*4, 1024),
            fc6 = L.Linear(1024, 2))

    def __call__(self, x, test=False):
        h = F.max_pooling_2d(F.tanh(self.conv1(x)), ksize=2)
        h = F.max_pooling_2d(F.tanh(self.conv2(h)), ksize=2)
        h = F.max_pooling_2d(F.tanh(self.conv3(h)), ksize=2)
        h = F.max_pooling_2d(F.tanh(self.conv4(h)), ksize=2)
        h = F.tanh(self.fc5(h))
        y = F.sigmoid(self.fc6(h))
        return y

class Generator_TwoLayers(chainer.Chain):
    def __init__(self, z_size=100):
        self.z_size = z_size
        super(Generator_TwoLayers, self).__init__(
            fc1 = L.Linear(self.z_size, 1024),
            fc2 = L.Linear(1024, 128*16*16),
            norm2 = L.BatchNormalization(128*16*16),
            conv3 = L.Deconvolution2D(128, 64, ksize=4, stride=1, pad=1),
            conv4 = L.Deconvolution2D(64, 1, ksize=4, stride=1, pad=1))

    def __call__(self, z, test=False):
        h = F.tanh(self.fc1(z))
        h = F.tanh(self.norm2(self.fc2(h), test=test))
        h = F.reshape(h, (100, 128, 16, 16))
        h = F.tanh(self.conv3(F.unpooling_2d(h, ksize=2)))
        x = F.tanh(self.conv4(F.unpooling_2d(h, ksize=2)))
        return x

class Discriminator_TwoLayers(chainer.Chain):
    def __init__(self):
        super(Discriminator_TwoLayers, self).__init__(
            conv1 = L.Convolution2D(1, 64, ksize=4, stride=1, pad=1),
            conv2 = L.Convolution2D(64, 128, ksize=4, stride=1, pad=1),
            fc3 = L.Linear(128*16*16, 1024),
            fc4 = L.Linear(1024, 2))

    def __call__(self, x, test=False):
        h = F.max_pooling_2d(F.tanh(self.conv1(x)), ksize=2)
        h = F.max_pooling_2d(F.tanh(self.conv2(h)), ksize=2)
        h = F.tanh(self.fc3(h))
        y = F.sigmoid(self.fc4(h))
        return y

class Generator_ThreeLayers(chainer.Chain):
    def __init__(self, z_size=100):
        self.z_size = z_size
        super(Generator_ThreeLayers, self).__init__(
            fc1 = L.Linear(self.z_size, 1024),
            fc2 = L.Linear(1024, 256*8*8),
            norm2 = L.BatchNormalization(256*8*8),
            conv3 = L.Deconvolution2D(256, 128, ksize=4, stride=1, pad=1),
            conv4 = L.Deconvolution2D(128, 64, ksize=4, stride=1, pad=1),
            conv5 = L.Deconvolution2D(64, 1, ksize=4, stride=1, pad=1))

    def __call__(self, z, test=False):
        n_sample = z.data.shape[0]
        h = F.tanh(self.fc1(z))
        h = F.tanh(self.norm2(self.fc2(h), test=test))
        h = F.reshape(h, (n_sample, 256, 8, 8))
        h = F.tanh(self.conv3(F.unpooling_2d(h, ksize=2)))
        h = F.tanh(self.conv4(F.unpooling_2d(h, ksize=2)))
        x = F.tanh(self.conv5(F.unpooling_2d(h, ksize=2)))
        return x

class Discriminator_ThreeLayers(chainer.Chain):
    def __init__(self):
        super(Discriminator_ThreeLayers, self).__init__(
            conv1 = L.Convolution2D(1, 64, ksize=4, stride=1, pad=1), # -> 32*32
            conv2 = L.Convolution2D(64, 128, ksize=4, stride=1, pad=1), # -> 16*16
            conv3 = L.Convolution2D(128, 256, ksize=4, stride=1, pad=1), # -> 8*8
            fc4 = L.Linear(256*8*8, 1024),
            fc5 = L.Linear(1024, 2))

    def __call__(self, x, test=False):
        h = F.max_pooling_2d(F.tanh(self.conv1(x)), ksize=2)
        h = F.max_pooling_2d(F.tanh(self.conv2(h)), ksize=2)
        h = F.max_pooling_2d(F.tanh(self.conv3(h)), ksize=2)
        h = F.tanh(self.fc4(h))
        y = F.sigmoid(self.fc5(h))
        return y


class Generator_ThreeLayers_MultiGPU(chainer.Chain):
    def __init__(self, z_size=100):
        self.z_size = z_size
        super(Generator_ThreeLayers_MultiGPU, self).__init__(
            generator_gpu0 = Generator_ThreeLayers(self.z_size).to_gpu(0),
            generator_gpu1 = Generator_ThreeLayers(self.z_size).to_gpu(1),
            )

    def __call__(self, z, test=False):
        z0 = self.generator_gpu0(z)
        z1 = self.generator_gpu1(F.copy(z, 1))
        x = z0 + F.copy(z1, 0)
        return x

class Discriminator_ThreeLayers_MultiGPU(chainer.Chain):
    def __init__(self):
        super(Discriminator_ThreeLayers_MultiGPU, self).__init__(
            discriminator_gpu0 = Discriminator_ThreeLayers().to_gpu(0),
            discriminator_gpu1 = Discriminator_ThreeLayers().to_gpu(1),
            )

    def __call__(self, x, test=False):
        x0 = self.discriminator_gpu0(x)
        x1 = self.discriminator_gpu1(F.copy(x, 1))
        y = x0 + F.copy(x1, 0)
        return y


class Generator_ThreeLayers_NoPooling(chainer.Chain):
    def __init__(self, z_size=100):
        self.z_size = z_size
        super(Generator_ThreeLayers_NoPooling, self).__init__(
            fc1 = L.Linear(self.z_size, 1024),
            fc2 = L.Linear(1024, 256*8*8),
            norm2 = L.BatchNormalization(256*8*8),
            conv3 = L.Deconvolution2D(256, 128, ksize=4, stride=2, pad=1),
            conv4 = L.Deconvolution2D(128, 64, ksize=4, stride=2, pad=1),
            conv5 = L.Deconvolution2D(64, 1, ksize=4, stride=2, pad=1))

    def __call__(self, z, test=False):
        n_sample = z.data.shape[0]
        h = F.tanh(self.fc1(z))
        h = F.tanh(self.norm2(self.fc2(h), test=test))
        h = F.reshape(h, (n_sample, 256, 8, 8))
        h = F.tanh(self.conv3(h))
        h = F.tanh(self.conv4(h))
        x = F.tanh(self.conv5(h))
        return x


class Discriminator_ThreeLayers_NoPooling(chainer.Chain):
    def __init__(self):
        super(Discriminator_ThreeLayers_NoPooling, self).__init__(
            conv1 = L.Convolution2D(1, 64, ksize=4, stride=2, pad=1), # -> 32*32
            conv2 = L.Convolution2D(64, 128, ksize=4, stride=2, pad=1), # -> 16*16
            conv3 = L.Convolution2D(128, 256, ksize=4, stride=2, pad=1), # -> 8*8
            fc4 = L.Linear(256*8*8, 1024),
            fc5 = L.Linear(1024, 2))

    def __call__(self, x, test=False):
        h = F.tanh(self.conv1(x))
        h = F.tanh(self.conv2(h))
        h = F.tanh(self.conv3(h))
        h = F.tanh(self.fc4(h))
        y = F.sigmoid(self.fc5(h))
        return y


class Generator_ThreeLayers_Fixed(chainer.Chain):
    def __init__(self, z_size=100):
        self.z_size = z_size
        super(Generator_ThreeLayers_Fixed, self).__init__(
            fc1 = L.Linear(self.z_size, 256*8*8),
            conv2 = L.Deconvolution2D(256, 128, ksize=4, stride=2, pad=1),
            conv3 = L.Deconvolution2D(128, 64, ksize=4, stride=2, pad=1),
            conv4 = L.Deconvolution2D(64, 1, ksize=4, stride=2, pad=1))

    def __call__(self, z, test=False):
        n_sample = z.data.shape[0]
        h = F.tanh(self.fc1(z))
        h = F.reshape(h, (n_sample, 256, 8, 8))
        h = F.tanh(self.conv2(h))
        h = F.tanh(self.conv3(h))
        x = F.tanh(self.conv4(h))
        return x


class Discriminator_ThreeLayers_Fixed(chainer.Chain):
    def __init__(self):
        super(Discriminator_ThreeLayers_Fixed, self).__init__(
            conv1 = L.Convolution2D(1, 64, ksize=4, stride=2, pad=1),
            conv2 = L.Convolution2D(64, 128, ksize=4, stride=2, pad=1),
            conv3 = L.Convolution2D(128, 256, ksize=4, stride=2, pad=1),
            fc4 = L.Linear(256*8*8, 2))

    def __call__(self, x, test=False):
        h = F.tanh(self.conv1(x))
        h = F.tanh(self.conv2(h))
        h = F.tanh(self.conv3(h))
        y = F.sigmoid(self.fc4(h))
        return y


class Classifier_AlexNet(chainer.Chain):
    '''
    AlexNetを参考に
    '''
    def __init__(self, class_n=26):
        self.class_n = class_n
        super(Classifier_AlexNet, self).__init__(
            conv1=L.Convolution2D(1,  96, 8, stride=4),
            conv2=L.Convolution2D(96, 256,  5, pad=2),
            conv3=L.Convolution2D(256, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 384,  3, pad=1),
            conv5=L.Convolution2D(384, 256,  3, pad=1),
            fc6=L.Linear(256, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, self.class_n))

    def __call__(self, x, train=True):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=train)
        h = F.dropout(F.relu(self.fc7(h)), train=train)
        y = self.fc8(h)
        return y


class Classifier_AlexNet_MultiGPU(chainer.Chain):
    def __init__(self, class_n=26):
        self.class_n = class_n
        super(Classifier_AlexNet_MultiGPU, self).__init__(
            classifier_gpu0 = Classifier_AlexNet(self.class_n).to_gpu(0),
            classifier_gpu1 = Classifier_AlexNet(self.class_n).to_gpu(1),
            )

    def __call__(self, x, train=True):
        x0 = self.classifier_gpu0(x)
        x1 = self.classifier_gpu1(F.copy(x, 1))
        y = x0 + F.copy(x1, 0)
        return y


