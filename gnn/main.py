# -*-coding:utf8;-*-

from net import GNN
import utils.data_utils
from utils.utils import make_rgb
from trainer import Trainer, NAdam
import numpy as np


if __name__ == '__main__':

    data, vocab = utils.data_utils.make_data('data.txt', load_file=True)
    print('Loaded Data')

    net = GNN(data[0][0].shape,
              layers=(('fc', 128), ('RELU', 1.),
                      ('dropout', 0.6),
                      ('LSTM', 512),
                      ('dropout', 0.6),
                      ('LSTM', 512),
                      ('dropout', 0.6),
                      ('fc', -1), ('softmax',)),
              labels=vocab.size,
              dtype=np.float32)

    trainer = Trainer(kernel=NAdam)
    print('Built Network with ' +
          make_rgb('{} paramters'.format(net.num_params), 255, 100, 0) +
          ' and  Trainer')
    losses, perf = trainer.train(net=net,
                                 gek=Trainer.recurrent_gek,
                                 data=data,
                                 epochs=1,
                                 batch_size=8,
                                 plotparams=(True, True, True, True, 1000),
                                 seq_len=32,
                                 saveif='losses[-1] < 1.',
                                 responsive=True)
