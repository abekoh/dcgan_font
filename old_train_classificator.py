import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import Variable

from mylib.chainer.dataset import filelist_to_list


class Simple(chainer.Chain):

    def __init__(self):
        super(Simple, self).__init__(
            l1=L.Linear(None, 1000),
            l2=L.Linear(None, 1000),
            l3=L.Linear(None, 26))

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y


class LeNet(chainer.Chain):

    def __init__(self):
        super(LeNet, self).__init__(
            conv1=L.Convolution2D(1, 20, 5),
            conv2=L.Convolution2D(20, 50, 5),
            fc3=L.Linear(800, 500),
            fc4=L.Linear(500, 26))

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2, stride=2)
        h = F.relu(self.fc3(h))
        y = F.softmax(self.fc4(h))
        return y


def train_model(train_list_txt, test_list_txt):
    model = L.Classifier(Simple())

    chainer.cuda.get_device(0).use()
    model.to_gpu()

    # optimizer = chainer.optimizers.SGD(lr=0.01)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train = filelist_to_list(train_list_txt)
    test = filelist_to_list(test_list_txt)
    # train, test = chainer.datasets.get_mnist()

    train_iter = chainer.iterators.SerialIterator(train, 100)
    test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=0)
    trainer = training.Trainer(updater, (10, 'epoch'), out='result')

    trainer.extend(extensions.Evaluator(test_iter, model, device=0))

    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))

    trainer.extend(extensions.LogReport())

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))

    trainer.extend(extensions.ProgressBar())

    # if args.resume:
    #     # Resume from a snapshot
    #     chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()



def debug():
    train_list_txt = '/home/abe/font_dataset/png_6628_64x64/train_1000.txt'
    test_list_txt = '/home/abe/font_dataset/png_6628_64x64/test_1000.txt'
    train_model(train_list_txt, test_list_txt)

if __name__ == '__main__':
    debug()
