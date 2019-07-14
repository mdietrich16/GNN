# -*- coding: utf-8 -*-
"""Module containing the Generic Neural Network class.

Created on Wed May 22 14:43:32 2019

@author: maxid
"""

import numpy as np
from utils.cnn_utils import conv, pool, pool_back
from utils.cnn_utils import conv_back_im2col as conv_back
from utils.lstm_utils import LSTM, LSTM_back
from utils.utils import cross_entropy as loss
from utils.utils import cross_entropy_prime as lossp
from utils.utils import softmax, dropout_prime, RELU
from utils.utils import RELU_prime, ELU, ELU_prime, dropout


class GNN:
    r"""Neural network class for building any popular architecture.

    Initialize a neural network.

    Parameters
    ----------
    in_shape : tuple with length 4
        Batched data input array with shape ``(n, c, w, h)``, where
        ``n`` is the number of examples, ``c`` is the number of
        channels, ``w`` is the width and ``h`` is the height.
    layers : tuple of tuples, optional
        If ``test``, then this will switch to a test-case
        implementation, wich might include features only useful when
        deploying or architectural requirements.
        The default is (('conv', 64, 5, 5),
                        ('ELU', 1.),
                        ('pool', 2, 2),
                        ('conv', 64, 3, 3),
                        ('ELU', 1.),
                        ('pool', 2, 2),
                        ('fc', -1),
                        ('softmax',)).
    seed : unsigned integer, optional
        Used as a seed to prime the random generator if deterministic
        behaviour is wanted. The default is None.
    labels : TYPE, optional
        number labels or output nodes. Only optional for convenience,
        this is usually an importent parameter to set! The default is 10.

    Raises
    ------
    ValueError
        Raised when an unknown layer type or syntax is specified.

    Returns
    -------
    None.

    Methods
    -------
        save(name, date, folder, **kwds)
            Saves network in either './Nets/net***(date).npz' where
            *** is a numbering in ascending order or under
            corresponding name and folder, suppliable separately.

            If date, the date and time of the function call are appended
            to the name.
        feedforward(batch, test)
            Feeds the (possible multiple instances of)
            input data through the network and returns
            the output of the next layer and a cache
            for backpropagation.

            If ``test``, uses other internal mechanics
            for test-time use, such as scaled weights
            in dropout-layers.
        backprop(caches, labels, test)
            propagates the error through the
            network using caches from
            ``feedforward()``. ``test`` has to have
            the same value as in the
            ``feedforward()`` call used to generate
            the cache.
    """

    @staticmethod
    def __convolution(x, params, **kwargs):
        o, cols = conv(x, params[0], params[1], pad=(params[0].shape[2]-1)//2)
        return o, (cols, x.shape)

    @staticmethod
    def __convolution_back(dout, cache, params, **kwargs):
        x_cols, x_shape = cache
        dx, dw, db = conv_back(dout, x_cols, x_shape, params[0], params[1],
                               (params[0].shape[2]-1)//2, 1)
        return dx, [dw, db]

    @staticmethod
    def __pooling(x, params, **kwargs):
        o, ix = pool(x, params[0])
        return o, (ix, x.shape)

    @staticmethod
    def __pooling_back(dout, cache, params, **kwargs):
        ix, x_shape = cache
        dx = pool_back(dout, ix, x_shape, params[0])
        return dx, ()

    @staticmethod
    def __LSTM(x, params, **kwargs):
        w, b, h0, c0, _ = params
        if kwargs.get('reset', False) or h0.shape[0] != x.shape[0]:
            dtype = h0.dtype
            h0 = np.zeros((x.shape[0], h0.shape[1]), dtype=dtype)
            c0 = np.zeros((x.shape[0], h0.shape[1]), dtype=dtype)
            params[4] = [np.zeros((x.shape[0], h0.shape[1]), dtype=dtype),
                         np.zeros((x.shape[0], h0.shape[1]), dtype=dtype)]
        c = LSTM(x.reshape(x.shape[0], -1), w, b, h0, c0)
        params[2] = c[0]
        params[3] = c[1]
        return c[0].reshape(x.shape[0], -1, 1, 1), (c, x.shape)

    @staticmethod
    def __LSTM_back(dout, cache, params, **kwargs):
        w, b = params[:2]
        dlast = params[4]
        c, x_shape = cache
        dx, dw, db, dnext = LSTM_back(dout.reshape(dout.shape[0], -1), w, b,
                                      c, dlast)
        params[4] = dnext
        return dx.reshape(x_shape), [dw, db]

    @staticmethod
    def __RELU(x, params, **kwargs):
        return RELU(x), (x,)

    @staticmethod
    def __RELU_back(dout, cache, params, **kwargs):
        x = cache[0]
        dx = RELU_prime(x)
        return dx*dout, ()

    @staticmethod
    def __ELU(x, params, **kwargs):
        o = ELU(x.copy(), params[0])
        return o, (o,)

    @staticmethod
    def __ELU_back(dout, cache, params, **kwargs):
        o = cache[0]
        dx = ELU_prime(o, params[0])
        return dx*dout, ()

    @staticmethod
    def __fully_connected(x, params, **kwargs):
        if kwargs.get('test', False):
            o, cols = conv(x, params[0], params[1], pad=0)
            return o, (cols, x.shape)
        w = params[0].reshape(params[0].shape[0], -1)
        x_ = x.T.reshape(-1, x.shape[0])
        o = np.ascontiguousarray((w.dot(x_) + params[1]).T)
        return o[:, :, np.newaxis, np.newaxis], (x_, w, x.shape)

    @staticmethod
    def __fully_connected_back(dout, cache, params, **kwargs):
        if kwargs.get('test', False):
            x_cols, x_shape = cache
            dx, dw, db = conv_back(dout, x_cols, x_shape,
                                   params[0], params[1], 0)
            return dx, (dw, db)
        x, w, x_shape = cache
        db = dout.sum(axis=(0, 2, 3)).reshape(-1, 1)
        dout_r = dout.reshape(dout.shape[0], -1).T
        dw = dout_r.dot(x.T).reshape(params[0].shape)
        dx = dout_r.T.dot(w).reshape(x_shape)
        return dx, [dw, db]

    @staticmethod
    def __dropout(x, params, **kwargs):
        if kwargs.get('test', False):
            return x * params[0], ()
        o, c = dropout(x, params[0])
        return o, (c,)

    @staticmethod
    def __dropout_back(dout, cache, params, **kwargs):
        if kwargs.get('test', False):
            return dout*params[0], ()
        return dropout_prime(dout, cache[0]), ()

    @staticmethod
    def __softmax(x, params, **kwargs):
        o = softmax(x, alpha=kwargs.get('alpha', 1.))
        return o, (o,)

    @staticmethod
    def __softmax_back(dout, cache, params, **kwargs):
        # dout is the labels in this layer, not the gradient!!!!!
        dx = lossp(cache[0], dout, alpha=kwargs.get('alpha', 1.))
        return dx, ()

    def __init__(self,
                 in_shape,
                 layers=(('conv', 64, 5, 5),
                         ('ELU', 1.),
                         ('pool', 2, 2),
                         ('conv', 64, 3, 3),
                         ('ELU', 1.),
                         ('pool', 2, 2),
                         ('fc', -1),
                         ('softmax',)),
                 labels=10, seed=None, dtype=np.float64):
        """
        Initialize a neural network.

        Parameters
        ----------
        in_shape : tuple with length 4
            Batched data input array with shape ``(n, c, w, h)``, where
            ``n`` is the number of examples, ``c`` is the number of
            channels, ``w`` is the width and ``h`` is the height.
        layers : tuple of tuples, optional
            If ``test``, then this will switch to a test-case
            implementation, wich might include features only useful when
            deploying or architectural requirements.
            The default is (('conv', 64, 5, 5),
                            ('ELU', 1.),
                            ('pool', 2, 2),
                            ('conv', 64, 3, 3),
                            ('ELU', 1.),
                            ('pool', 2, 2),
                            ('fc', -1),
                            ('softmax',)).
        seed : unsigned integer, optional
            Used as a seed to prime the random generator if deterministic
            behaviour is wanted. The default is None.
        labels : int, optional
            Number of labels or output nodes. Only optional for convenience,
            this is usually an importent parameter to set! The default is 10.
        dtype : dtype, optional
            Sets the type for weights and biases. Smaller (e.g. `np.float16`)
            means less precision, but faster execution. The default is float64.

        Raises
        ------
        ValueError
            Raised when an unknown layer type or syntax is specified.

        Returns
        -------
        None.

        """
        self.layers = []
        self.shapes = [(1, *in_shape[1:])]
        self.num_params = 0
        if seed:
            np.random.seed(seed)

        for l in layers:
            if len(l) > 1 and isinstance(l[1], np.ndarray):
                self.layers = layers
                for i, lay in enumerate(layers):
                    if lay[0] in ('RELU', 'ELU', 'softmax', 'dropout'):
                        self.shapes.append(self.shapes[i])
                    elif lay[0] == 'pool':
                        self.shapes.append((1, self.shapes[i][1],
                                            self.shapes[i][2]//lay[1],
                                            self.shapes[i][3]//lay[2]))
                    elif lay[0] == 'fc':
                        self.shapes.append((1, lay[1].shape[0], 1, 1))
                        self.num_params += (np.prod(lay[1].shape) +
                                            lay[2].shape[0])
                    elif lay[0] == 'LSTM':
                        outs = lay[1].shape[1]//4
                        self.shapes.append((1, outs, 1, 1))
                        self.num_params += (np.prod(lay[1].shape) +
                                            lay[2].shape[0])
                    elif lay[0] == 'conv':
                        self.shapes.append((1, lay[1].shape[0],
                                            *self.shapes[i][2:]))
                        self.num_params += (np.prod(lay[1].shape) +
                                            lay[2].shape[0])
                break

        if not self.layers:
            for l in range(len(layers)):
                if layers[l][0] == 'conv':
                    w = (np.random.randn(layers[l][1], self.shapes[l][1],
                                         layers[l][2], layers[l][3])
                         .astype(dtype),
                         np.zeros((layers[l][1], 1), dtype=dtype))
                    self.num_params += (np.prod(layers[l][1:]) *
                                        self.shapes[l][1]
                                        + layers[l][1])
                    self.layers.append(('conv', *w))
                    self.shapes.append((1, layers[l][1],
                                        self.shapes[l][2], self.shapes[l][3]))
                elif layers[l][0] == 'pool':
                    self.layers.append(layers[l])
                    self.shapes.append((1, self.shapes[l][1],
                                        self.shapes[l][2]//layers[l][1],
                                        self.shapes[l][3]//layers[l][2]))
                elif layers[l][0] in ('RELU', 'softmax', 'dropout', 'ELU'):
                    self.layers.append(layers[l])
                    self.shapes.append(self.shapes[l])
                elif layers[l][0] == 'fc':
                    outs = layers[l][1] if layers[l][1] > 0 else labels
                    w = (np.random.randn(outs, *self.shapes[l][1:])
                         .astype(dtype) /
                         np.sqrt(self.shapes[l][2] * self.shapes[l][3] / 2.),
                         np.zeros((outs, 1), dtype=dtype))
                    self.num_params += (outs * np.prod(self.shapes[l][1:])
                                        + outs)
                    self.layers.append(('fc', *w))
                    self.shapes.append((1, outs, 1, 1))
                elif layers[l][0] == 'LSTM':
                    outs = layers[l][1] if layers[l][1] > 0 else labels
                    bias_init = 1.
                    if len(layers[l]) > 2:
                        bias_init = layers[l][2]
                    ins = np.prod(self.shapes[l][1:])
                    w = (np.random.randn(ins + outs, 4 * outs).astype(dtype)
                         / np.sqrt(ins + outs))
                    b = np.zeros((1, 4 * outs), dtype=dtype)
                    b[:, outs:2*outs] = bias_init
                    h = np.zeros((1, outs))
                    self.num_params += (outs * 4 * (ins + outs)
                                        + 4 * outs)
                    self.layers.append(np.array(['LSTM', w, b, h, h, [h, h]]))
                    self.shapes.append((1, outs, 1, 1))
                else:
                    raise ValueError('Unrecognized layer type <{}>'
                                     .format(layers[l][0]))

        self.funcs = {'conv':       GNN.__convolution,
                      'dropout':    GNN.__dropout,
                      'pool':       GNN.__pooling,
                      'LSTM':       GNN.__LSTM,
                      'RELU':       GNN.__RELU,
                      'ELU':        GNN.__ELU,
                      'fc':         GNN.__fully_connected,
                      'softmax':    GNN.__softmax}

        self.back_funcs = {'conv':      GNN.__convolution_back,
                           'dropout':   GNN.__dropout_back,
                           'pool':      GNN.__pooling_back,
                           'LSTM':      GNN.__LSTM_back,
                           'RELU':      GNN.__RELU_back,
                           'ELU':       GNN.__ELU_back,
                           'fc':        GNN.__fully_connected_back,
                           'softmax':   GNN.__softmax_back}

    def feedforward(self, batch, **kwargs):
        """
        Compute an  estimate given data.

        Parameters
        ----------
        batch : ndarray of shape (n, c, h, w)
            Input data. Has to match the networks input shape(except n, the
            `batch size`)
        **kwargs
            Additional arguments given to the layer functions.
            test : boolean
                Switches to different implementations in some layers for
                test-time and deployment, that may be faster or introduce
                architectural changes.
            reset : boolean
                Used only in recurrent layers to indicate the start of a new
                sequence. This is necessary to ensure proper functionality of
                these layers!

        Returns
        -------
        z : ndarray of shape (n, k, i, j)
            The networks output. Most likely with shape (n, o, 1, 1) if using
            feedforward or recurrent layers in the end.
        caches : tuple of tuples or lists
            The layer-generated caches only useful for gradient
            backpropagation.

        """
        z = np.ascontiguousarray(batch)
        caches = []
        for l in self.layers:
            z, cache = self.funcs[l[0]](z, l[1:], **kwargs)
            caches.append(cache)
        return z, caches

    def backprop(self, caches, labels, **kwargs):
        """Feedback method of neural network.

        Parameters
        ----------
        caches : tuple of tuples or lists
            Caches returned from the corresponding ``feedforward``
            function call.
        labels : ndarray of shape (n, y)
            Batched data output array with shape ``(n, y)``, where
                ``n`` is the number of examples, ``y`` is the number of
                output classes.
        **kwargs
            These have to absolutely have the same values as in the
            corresponding feedforward call.
            test
                If ``test``, then this will switch to a test-case
                implementation, wich might include features only useful
                when deploying or architectural requirements.
                This has to have the same value as the corresponding
                ``feedforward`` call``(default False)``

        Returns
        -------
        dparams : list of ndarrays
            A list of gradients with respect to the cost function with the
            shapes of the parameters.

        """
        # Note that dx holds the labels in the last layer before
        # becoming the downpropagated gradient
        dx = labels
        dparams = []
        for l in reversed(range(len(self.layers))):
            dx, dparam = self.back_funcs[self.layers[l][0]](dx, caches[l],
                                                            self.layers[l][1:],
                                                            **kwargs)
            dparams.append(dparam)
        return list(reversed(dparams))

    def save(self, name=None, date=True, folder=None, **kwds):
        r"""Save a GNN instance to a compressed file.

        The weights and important parameters of every network layer are saved
        with a keyword 'Layer_*', where ``*`` is the 1-based layer index.
        'Layer_0' is just the input size.
        The file is specified by ``name`` and ``date`` and the folder by
        ``folder``.

        Parameters
        ----------
        name : string, optional
            If given, specifies the name. This may not be the only part in
            the files name if ``date`` is ``True``, however. If ``None``,
            use ``net***``, where ``***`` is an ascending three-digit
            number. The default is None.
        date : boolean, optional
            If ``date``, then the current date and time are appended to the
            filename with a ``yy-mm-dd_HH-MM`` format. The default is True.
        folder : string, optional
            String specifying in which folder to save. May be a relative or
            absolute path. If it does not exist, a new folder is created.
            If ``None``, looks for the ``\\Nets`` folder in the projects
            working directory. The default is None.
        **kwds : anything
            Additional data that is appended to the saved files data.

        Returns
        -------
        folder : string
            The folder part of the path this data was saved to.
        name : string
            The actual name of the saved file.
        postfix : string
            The postfix part of the filename, possibly an ascending
            three-digit number as a unique ID and possibly the date at the
            time of the function-call.

        """
        import os
        post = ''
        if date:
            import datetime
            date = '-{:%y-%m-%d_%H-%M}'.format(datetime.datetime.now())
        else:
            date = ''

        if not folder:
            path = os.getcwd()
            found = False
            while not found:
                found = 'Nets' in os.listdir(path)
                if not found:
                    if 'CNN' in os.listdir(path):
                        path = path + '\\CNN\\Nets'
                        if not os.path.isdir(path):
                            os.mkdir(path)
                        break
                    else:
                        path = '\\'.join(path.split('\\')[:-1])
                else:
                    path += '\\Nets'
            folder = path
        else:
            if not os.path.isdir(folder):
                os.makedirs(folder)

        if not folder.endswith('\\'):
            folder = folder + '\\'

        if not name:
            import glob
            entries = glob.glob(folder + 'net[0-9][0-9][0-9]*.npz')
            if not entries:
                post = '000'
            else:
                import re
                entry = sorted(entries, reverse=True)[0]
                post = '{:03d}'.format(int(re.search(
                    r'.*(\d\d\d).*\.npz', entry).group(1)) + 1)

            name = 'net' + post + date
        params = {'Layer_' + str(i+1): np.array(l, dtype=object)
                  for i, l in zip(range(len(self.layers)), self.layers)}
        params['Layer_0'] = self.shapes[0]
        np.savez_compressed(folder + name, **params, **kwds)
        return folder, name, post + date

    @staticmethod
    def load(mode='newest', folder=None, name=None):
        r"""
        Load a GNN instance from a compressed file saved with ``GNN.save``.

        The weights and important parameters of consecutive network layers are
        loaded from a file specified by ``mode`` and possibly ``name`` and a
        folder specified by ``folder``. Additional saved information is
        returned with the resulting GNN instance.

        Parameters
        ----------
        mode : str, optional
            Either 'newest', 'name' or 'interactive'.

                'newest'
                    ``load`` chooses the highest numbered net with a name
                    matching with 'net***[postfix]', where '***' is a
                    three-digit number on which sorting is performed and
                    '[postfix]' is an optional postfix.
                'name'
                    Fetches the file with `name` in folder `folder` or the
                    default saving folder if possible. `name` has to match
                    exactly, with extension.
                'interactive'
                    Shows a list of loadable files in `folder` or the default
                    saving folder and lets you choose one in a CLI.
            The default is 'newest'.
        folder : str, optional
            String specifying in which folder to look for networks. May be
            a relative or absolute path. If None, looks for the
            ``\\Nets`` folder in the projects working directory. Applicable in
            all `mode` s. The default is None.
        name : str, optional
            The file to load. Only used in 'name' `mode`. The default is None.

        Raises
        ------
        FileNotFoundError
            Raised when the file `name` in `mode` 'name' does not exist or when
            'newest' does not find a file. Also raised when `folder` does not
            exist.
        ValueError
            Raised when `mode` has an invalid value.

        Returns
        -------
        net : instance of GNN
            The loaded network.
        out : dict
            Additional loaded information.

        """
        import os
        if not folder:
            path = os.getcwd()
            found = False
            while not found:
                found = 'Nets' in os.listdir(path)
                if not found:
                    if 'CNN' in os.listdir(path):
                        path = path + '\\CNN\\Nets'
                        if not os.path.isdir(path):
                            raise FileNotFoundError('You cannot load a \
                                                    network out of a \
                                                    non-existing directory: {}'
                                                    .format(path))
                        break
                    else:
                        path = '\\'.join(path.split('\\')[:-1])
                else:
                    path += '\\Nets'
            folder = path
        else:
            if not os.path.isdir(folder):
                raise FileNotFoundError('You cannot load a network out of a \
                                        non-existing directory: {}'
                                        .format(folder))
        if mode == 'name':
            if not name and not os.path.isfile(folder + name):
                raise FileNotFoundError('You cannot load a network out of a \
                                        non-existing file: {}'
                                        .format(folder + name))
            name = folder + name
        elif mode == 'interactive':
            import glob
            fs = glob.glob(folder + '\\*.npz')
            if not fs:
                raise FileNotFoundError('You cannot load a network \
                                        if you never saved one')
            if len(fs) == 1:
                name = fs[0]
                print('Only network {} found in folder {}, so taking that one'
                      .format(name, os.path.abspath(folder)))
            else:
                print('Networks found in dir ' + os.path.abspath(folder) + ':')
                for i in range(len(fs)):
                    print('#{} : {}'.format(i, os.path.split(fs[i])[1]))
                n = int(input('Which network would you like to load?\n>>> #'))
                name = fs[n]
        elif mode == 'newest':
            import glob
            entries = glob.glob(folder + '\\net[0-9][0-9][0-9]*.npz')
            if not entries:
                raise FileNotFoundError('You cannot load a network \
                                        if you never saved one')
            name = sorted(entries, reverse=True)[0]
        else:
            raise ValueError('<mode> argument has to have value \
                             of either \'name\', \'interactive\' \
                             or \'newest\', <{}> given'.format(mode))
        net = np.load(name, allow_pickle=True)
        out = {}
        layers = []
        shape = None
        for k, v in net.items():
            if k == 'Layer_0':
                shape = tuple(v)
            elif 'Layer_' in k:
                layers.append(list(v))
            else:
                out[k] = v
        if shape is None or not isinstance(shape, tuple):
            raise ValueError('Loading broken network file, \
                             input shape not given. Aborting!')
        return GNN(shape, layers), out

    def get_params(self, dparams):
        """
        Compiles the networks parameters and gradients in a more usable format.

        Parameters
        ----------
        dparams : list of ndarrays
            The gradients computed by `GNN.backprop`.

        Returns
        -------
        p : list of ndarrays
            The networks paramters prepared for use in the `trainer` modules
            kernels.
        dp : list of ndarrays
            The input gradients prepared for use in the `trainer` modules
            kernels.

        """
        """"""
        p = (self.layers[i][k] for i in range(len(dparams))
             for k in range(1, 1+len(dparams[i])))
        dp = (dparams[i][k] for i in range(len(dparams))
              for k in range(len(dparams[i])))
        return p, dp

    def __str__(self):
        """
        Convert the network to a string representation.

        Returns
        -------
        str : string
            A string symbolically representing the Network.

        """
        s = 'Network('
        for i, l in enumerate(self.layers):
            s += l[0] + '<' + str(self.shapes[i+1][1])
            if l[0] == 'conv':
                s += ', ' + str(l[1].shape[2]) + ', ' + str(l[1].shape[3])
            s += '>, '
        s += '{} params)'.format(self.num_params)
        return s

    def __grad_check(self, layer, **kwargs):
        """
        Private method to sanity-check gradients.

        Parameters
        ----------
        layer : int
            Layer index to sanity-check.
        **kwargs : dict
            Possible arguments to supply to `GNN.feedforward`.

        Returns
        -------
        None.

        """
        try:
            if self.layers[layer][0] == 'softmax':
                x = np.random.standard_normal((2, *self.shapes[layer][1:]))
                o, cache = self.funcs[self.layers[layer][0]](x,
                                                             self.
                                                             layers[layer][1:],
                                                             kwargs)
                ls = loss(o, self.test_data[1][:1])
                dx, dparams = (self.back_funcs[self.layers[layer][0]]
                               (np.ones(self.shapes[layer+1], dtype=np.int8),
                                cache, self.layers[layer][1:], kwargs))
                x[0, 0, 0, 0] += 1e-6
                ox, cache = (self.funcs[self.layers[layer][0]]
                             (x, self.layers[layer][1:], kwargs))
                lx = loss(ox, self.test_data[1][:1])
                x[0, 0, 0, 0] -= 1e-6
                print('All dx\' close: {}'.
                      format(np.allclose(np.sum(lx-ls)/1e-6, dx[0, 0, 0, 0])))
                return
            x = np.random.standard_normal((2, *self.shapes[layer][1:]))
            o, cache = (self.funcs[self.layers[layer][0]]
                        (x, self.layers[layer][1:], kwargs))
            dx, dparams = (self.back_funcs[self.layers[layer][0]]
                           (np.ones((2, *self.shapes[layer+1][1:]),
                                    dtype=np.int8),
                            cache,
                            self.layers[layer][1:], kwargs))
            x[0, 0, 0, 0] += 1e-6
            ox, cache = (self.funcs[self.layers[layer][0]]
                         (x, self.layers[layer][1:], kwargs))
            x[0, 0, 0, 0] -= 1e-6
            print('All dx\' close: {}'.
                  format(np.allclose(np.sum(ox-o)/1e-6, dx[0, 0, 0, 0])))
            for p in range(1, 1+len(self.layers[layer][1:])):
                param = self.layers[layer][p]
                if isinstance(param, np.ndarray):
                    param.itemset(0, param.item(0) + 1e-6)
                    op, _ = (self.funcs[self.layers[layer][0]]
                             (x, self.layers[layer][1:], kwargs))
                    param.itemset(0, param.item(0) - 1e-6)
                    print('Param #{} close: {}'.
                          format(p, np.allclose(np.sum(op-o)/1e-6,
                                                dparams[p-1].item(0))))
        except AttributeError:
            print('<Error: self.shapes not found.> Grad-checking is only \
                  possible on a randomly sampled \
                  network due to internal mechanics.')


if __name__ == '__main__':
    pass
