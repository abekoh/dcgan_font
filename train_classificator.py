import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from mylib.keras.dataset import filelist_to_list

class LeNet(chainer.Chain)
    def __init__(self):
        super(LeNet, self).__init__(
            conv1 = L.Convolution2D(1, 20, ksize=5, j

def train_model(train_list_txt, test_list_txt):
    model = L.Classifier(LeNet())

    chainer.cuda.get_device(0).use()
    model.to_gpu()

    optimizer = chainer.optimizers.SGD(Lr=0.01)
    optimizer.setup(model)

    train_imgs, train_labels = filelist_to_list(train_list_txt)
    test_imgs, test_labels = filelist_to_list(test_list_txt)

    train_iter = chainer.iterators.SerialIterator(train, 128)
    test_iter = chainer.iterators.SerialIterator(test, 128, repeat=False, shuffle=False)

    train = (train_imgs, train_labels)
    test = (test_imgs, test_labels)

    updater = training.StandardUpdater(train_iter, optimizer, device=0)
    trainer = training.Trainer(updater, (10, 'epoch'), out='result')

    trainer.extend(extensions, Evaluator(test_iter, model, device=0))

    trainer.extenr(extensions.dump_graph('main/loss')

    trainer.extend(extensions, snapshot(), trigger=(10, 'epoch'))

    trainer.extend(extensions.LogReport())

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    
    trainer.extend(extensions.ProgressBar())

    trainer.run()


def debug():

if __name__ == __main__:
    debug()
