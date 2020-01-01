# -*-coding:utf8;-*-

from gnn.net import GNN
import gnn.utils.data_utils
from gnn.utils.utils import make_rgb
from gnn.trainer import Trainer, NAdam
import numpy as np


if __name__ == '__main__':

    data, vocab = gnn.utils.data_utils.make_data('data.txt', load_file=True)
    print('Loaded Data.')

    net = GNN(data[0][0].shape,
              layers=(('fc', 128), ('RELU', 1.),
                      ('LSTM', 256),
                      ('LSTM', 512),
                      ('fc', -1), ('softmax',)),
              labels=vocab.size,
              dtype=np.float32)

    trainer = Trainer(kernel=NAdam)
    print('Built Network with ' +
          make_rgb('{} parameters'.format(net.num_params), 255, 100, 0) +
          (' and Trainer with ' + make_rgb('{}', 32, 180, 180) + ' kernel')
          .format(trainer.kernel.__name__) + '.')
#    losses, perf = trainer.train(net=net,
#                                 gek=Trainer.recurrent_gek,
#                                 data=data,
#                                 epochs=1,
#                                 batch_size=8,
#                                 plotparams=(True, True, True, True, 1000),
#                                 seq_len=32,
#                                 saveif='losses[-1] < 1.',
#                                 responsive=True)
