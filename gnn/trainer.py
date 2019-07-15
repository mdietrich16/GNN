# -*- coding: utf-8 -*-
"""
Trainer and optimizer kernels.

Created on Wed May 22 14:43:43 2019

@author: maxid
"""
import time

import numpy as np
import re

from gnn.utils.utils import progress, cross_entropy
from gnn.utils.data_utils import shuffle_data, gen_mini_batches
from gnn.utils.data_utils import one_hot_encoding, make_data


def SGD(params, dparams, hp=(0.01,), cache=()):
    """
    SGD optimizer kernel.

    Kernel implementing one step of the SGD optimizer
    on given parameters in-place.

    Positional arguments:
        * params:
            list of numpy arrays to optimize using
        * dparams:
            list of numpy arrays with same shapes as params
            representing gradients of parameters

    Keyword arguments:
        * hp:
            learning rate (kernel specific) ``(default 0.01)``
        * cache:
            cache used to have a time-consistent optimization,
            depends on specific kernel implementation.
            The first ocurrence of this function should be given an
            empty tuple, and subsequent ones should be given the cache
            returned by the last call of this function ``(default ())``
    Returns:
        * cache:
            supplied to next call to this function via cache keyword
    """
    lr = hp[0]
    for param, dparam in zip(params, dparams):
        param -= lr*dparam
    return cache


def Adam(params, dparams, hp=(0.01, 0.9, 0.999), cache=()):
    """
    Adam optimizer kernel.

    Kernel implementing one step of the
    Adam optimizer on given parameters in-place.

    Positional arguments:
        * params:
             list of numpy arrays to optimize using
        * dparams:
             list of numpy arrays with same shapes as params
             representing gradients of parameters

    Keyword arguments:
        * hp:
            hyperparameters, consisting of learning rate,
            alpha decay and beta decay(kernel specific)
            ``(default (0.01, 0.9, 0.999))``
        * cache:
            cache used to have a time-consistent optimization,
            depends on specific kernel implementation.
            The first ocurrence of this function should be given an
            empty tuple, and subsequent ones should be given the cache
            returned by the last call of this function ``(default ())``
    Returns:
        * cache:
            supplied to next call to this function via cache keyword
    """
    lr, alpha, beta = hp
    if cache == ():
        t = 1
        params = list(params)
        mparams = [np.zeros_like(p) for p in params]
        vparams = [np.zeros_like(p) for p in params]
    else:
        t, mparams, vparams = cache
    for param, dparam, m, v in zip(params, dparams, mparams, vparams):
        m[...] = alpha*m + (1. - alpha) * dparam
        v[...] = beta*v + (1. - beta) * np.square(dparam)
        ny = lr * np.sqrt(1. - np.power(beta, t)) / (1. - np.power(alpha, t))
        param -= ny * m / (np.sqrt(v) + 1e-8)
    t = t+1
    return (t, mparams, vparams)


def AdaMax(params, dparams, hp=(0.002, 0.9, 0.999), cache=()):
    """
    Apply AdaMax optimizer kernel.

    Kernel implementing one step of the
    AdaMax optimizer on given parameters in-place.

    Positional arguments:
        * params:
            list of numpy arrays to optimize using
        * dparams:
            list of numpy arrays with same shapes as params
            representing gradients of parameters

    Keyword arguments:
        * hp:
            hyperparameters, consisting of learning rate,
            alpha decay and beta decay(kernel specific)
            ``(default (0.002, 0.9, 0.999))``
        * cache:
            cache used to have a time-consistent optimization,
            depends on specific kernel implementation.
            The first ocurrence of this function should be given an
            empty tuple, and subsequent ones should be given the cache
            returned by the last call of this function ``(default ())``
    Returns:
       * cache:
           supplied to next call to this function via cache keyword
    """
    lr, alpha, beta = hp
    if cache == ():
        t = 1
        params = list(params)
        mparams = [np.zeros_like(p) for p in params]
        vparams = [np.zeros_like(p) for p in params]
    else:
        t, mparams, vparams = cache
    for param, dparam, m, v in zip(params, dparams, mparams, vparams):
        m[...] = alpha*m + (1. - alpha) * dparam
        v[...] = np.maximum(beta*v, np.abs(dparam))
        param -= lr * m / v
    t = t+1
    return (t, mparams, vparams)


def NAdam(params, dparams, hp=(0.01, 0.9, 0.999), cache=()):
    """
    Apply NAdam optimizer kernel.

    Kernel implementing one step of the
    NAdam optimizer on given parameters in-place.

    Positional arguments:
        * params:
            list of numpy arrays to optimize using
        * dparams:
            list of numpy arrays with same shapes as params
            representing gradients of parameters

    Keyword arguments:
        * hp:
            hyperparameters, consisting of learning rate,
            alpha decay and beta decay(kernel specific)
            ``(default (0.01, 0.9, 0.999))``
        * cache:
            cache used to have a time-consistent optimization,
            depends on specific kernel implementation.
            The first ocurrence of this function should be given an
            empty tuple, and subsequent ones should be given the cache
            returned by the last call of this function ``(default ())``
    Returns:
        * cache:
            supplied to next call to this function via cache keyword
    """
    lr, alpha, beta = hp
    if cache == ():
        t = 1
        params = list(params)
        mparams = [np.zeros_like(p) for p in params]
        vparams = [np.zeros_like(p) for p in params]
    else:
        t, mparams, vparams = cache
    for param, dparam, m, v in zip(params, dparams, mparams, vparams):
        m[...] = alpha*m + (1. - alpha) * dparam
        v[...] = beta*v + (1. - beta) * np.square(dparam)
        f = 1./(1. - np.power(alpha, t))
        mh = m*f
        vh = v/(1. - np.power(beta, t))
        param -= lr * (alpha * mh + (1. - alpha)
                       * f * dparam) / (np.sqrt(vh) + 1e-8)
    t = t+1
    return (t, mparams, vparams)


def AdaGrad(params, dparams, hp=(0.01,), cache=()):
    """
    Apply AdaGrad optimizer kernel.

    Kernel implementing one step of the
    AdaGrad optimizer on given parameters in-place.

    Positional arguments:
        * params:
              list of numpy arrays to optimize using
        * dparams:
            list of numpy arrays with same shapes as params
            representing gradients of parameters

    Keyword arguments:
        * hp:
            learning rate (kernel specific) ``(default 0.01)``
        * cache:
            cache used to have a time-consistent optimization,
            depends on specific kernel implementation.
            The first ocurrence of this function should be given an
            empty tuple, and subsequent ones should be given the cache
            returned by the last call of this function ``(default ())``
    Returns:
        * cache:
            supplied to next call to this function via cache keyword
    """
    lr = hp
    if cache == ():
        params = list(params)
        mparams = [np.zeros_like(p) for p in params]
        cache = (mparams,)
    mparams = cache[0]
    for param, dparam, mparam in zip(params, dparams, mparams):
        mparam += np.square(dparam)
        param -= lr * dparam / np.sqrt(mparam + 1e-8)
    return (mparams,)


def RMSProp(params, dparams, hp=(0.001, 0.9), cache=()):
    """
    Apply RMSProp optimizer kernel.

    Kernel implementing one step of the
    RMSProp optimizer on given parameters in-place.

    Arguments
    ---------
        params : list of ndarrays
            list of numpy arrays to optimize using
        dparams : list of ndarrays
            list of numpy arrays with same shapes as params
            representing gradients of parameters
        hp : tuple
            hyperparameters, consisting of learning rate
            and beta decay(kernel specific) (default (0.001, 0.9))
        cache : tuple or list
            cache used to have a time-consistent optimization,
            depends on specific kernel implementation.
            The first ocurrence of this function should be given an
            empty tuple, and subsequent ones should be given the cache
            returned by the last call of this function (default ())

    Returns
    -------
        cache : tuple
            supplied to next call to this function via cache keyword
    """
    lr, decay = hp
    if cache == ():
        params = list(params)
        mparams = [np.zeros_like(p) for p in params]
        cache = (mparams,)
    mparams = cache[0]
    for param, dparam, mparam in zip(params, dparams, mparams):
        mparam[...] = decay*mparam + (1. - decay) * np.square(dparam)
        param -= lr * dparam / np.sqrt(mparam + 1e-8)
    return (mparams,)


def Momentum(params, dparams, hp=(0.01, 0.9), cache=()):
    """
    Apply SGD + Momentum optimzer.

    Kernel implementing one step of the SGD
    + Momentum optimizer on given parameters in-place.

    Positional arguments:
        * params:
            list of numpy arrays to optimize using
        * dparams:
            list of numpy arrays with same shapes as params
            representing gradients of parameters

    Keyword arguments:
        * hp:
            hyperparameters, consisting of learning rate
            and decay rate (kernel specific) ``(default (0.01, 0.9))``
        * cache:
            cache used to have a time-consistent optimization,
            depends on specific kernel implementation.
            The first ocurrence of this function should be given an
            empty tuple, and subsequent ones should be given the cache
            returned by the last call of this function ``(default ())``
    Returns:
        * cache:
            supplied to next call to this function via cache keyword
    """
    lr, decay = hp
    if cache == ():
        params = list(params)
        mparams = [np.zeros_like(p) for p in params]
        cache = (mparams,)
    mparams = cache[0]
    for param, dparam, mparam in zip(params, dparams, mparams):
        mparam[...] = decay*mparam + lr * dparam
        param -= mparam
    return (mparams,)
    lr, decay = hp


class Trainer:
    """Init a Trainer for neural nets using a generic optimizer on any data.

    Keyword arguments
    -----------------
        kernel : callable
            kernel function object exposing the kernel interface.
            The default is `trainer.Adam`
        hyperparams : tuple or None, optional
            Directly supplied to the optimizer kernel.
            The default is None, using kernel default
    """

    def __init__(self, kernel=Adam, hyperparams=None):
        """Init a Trainer for neural nets using a generic optimizer on data.

        Keyword arguments
        -----------------
            kernel : callable
                kernel function object exposing the kernel interface.
                The default is `trainer.Adam`
            hyperparams : tuple or None, optional
                Directly supplied to the optimizer kernel.
                The default is None, using kernel default
        """
        self.kernel = kernel
        self.hp = hyperparams

    def train(self, net, gek, data, epochs, batch_size, **kwds):
        """
        Train a generic neural net using a generic optimizer on labelled data.

        Parameters
        ----------
        net : instance of `net.GNN`
            any Neural Net object exposing the network interface.
        gek : callable
            'Gradient Estimation Kernel', wraps the specfic mechanics of
            getting the gradient out of a (net, data, algorithm) combination.
        data : list of two lists of two ndarrays
            Labelled data in the right shape for the network,
            a list consisting of two lists of ndarrays of training data
            and labels and similarly testing data and labels.
        epochs : int or float
            Number of total epochs to train on,
            if less than zero train on only that fraction of the data.
        batch_size : int
            grouping size for single training steps.
        save : boolean
            If True, save the network and possibly the performance plots
            at the end. This defaults to True.
        saveif : str
            Only save if given expression evaluates to True
            at the end of training. This defaults to None.
        plot : boolean
            Plot a graph of network performance
            against training time. This defaults to True
        plotparams : tuple of four booleans and one int
            First 4 elements determine which
            performance measures are used,
                1.Testing accuracy
                2. Testing cost
                3. Training accuracy
                4. Training cost
            and last element determines how often
            per epoch performance is evaluated.
        seq_len : int, optional
            Determines the sequence length for recurrent neural networks. Only
            used with the recurrent_gek. This defaults to 32.
        responsive : boolean, optional
            If True, look for input to pause or stop training.

        Returns
        -------
        losses : list
            A list of losses at each training step.
        perf : ndarray
            Array of performance data the plots are based on.

        """
        save = kwds.get('save', True)
        saveif = kwds.get('saveif')
        plotting = kwds.get('plot', True)
        responsive = kwds.get('responsive', False)
        recurrent = gek == Trainer.recurrent_gek
        seq_len = kwds.get('seq_len', 32)
        if recurrent:
            batch_size = seq_len*batch_size

        train_data, test_data = data
        if isinstance(epochs, float) and epochs < 1.:
            train_data[0] = train_data[0][:int(np.round(epochs *
                                                        len(train_data[0])))]
            train_data[1] = train_data[1][:int(np.round(epochs *
                                                        len(train_data[1])))]
            """
            test_data[0] = test_data[0][:int(np.round(epochs
                                        * len(test_data[0])))]
            test_data[1] = test_data[1][:int(np.round(epochs
                                        * len(test_data[1])))]
            """

        epochs = int(np.ceil(epochs))
        data_len = train_data[0].shape[0]
        num_batches = data_len//batch_size or 1

        print(('Performing SGD with {} optimizer for {} epochs with '
               'mini-batch size of {}, learning rate of {} and '
               '{} parameter updates')
              .format(self.kernel.__name__, epochs, batch_size,
                      self.hp[0] if isinstance(self.hp, tuple)
                      else self.hp, num_batches*epochs))

        optim_cache = ()
        losses = []
        perf = None
        if plotting:
            plotparam = kwds.setdefault('plotparams', (True, True,
                                                       True, True, 10))
            plot_every = (int(num_batches//plotparam[4])
                          if type(plotparam[4]) is int
                          else int(num_batches/plotparam[4])) or 1
            perf = np.zeros((int(epochs * num_batches // plot_every + 1), 4))
            tacc, tcost, acc, cost = \
                Trainer.cost_accuracy(net, data=data, samples=100,
                                      cost_on_training=plotparam[2],
                                      accuracy_on_training=plotparam[3],
                                      recurrent=recurrent)
            perf[0] = np.array([tacc, tcost, acc, cost])
        running_loss = np.log(data[0][0].shape[0])
        if responsive:
            print('Press <p> + <Enter> for pause and <c> + <Enter> for abort.')
            import threading
            import queue
            q = queue.SimpleQueue()

            def handle_input(q):
                s = ''
                while 'c' not in s:
                    s = input()
                    q.put(s)

            thread = threading.Thread(target=handle_input, args=[q])
            thread.start()

        n = 0
        ts = time.time()
        cumtime = 0
        remaining = None
        elapsed = np.inf

        for epoch in range(epochs):
            if not recurrent:
                shuffle_data(train_data)

            mini_batches = gen_mini_batches(train_data, batch_size,
                                            strict=True)

            for x, y in mini_batches:

                loss, dparams = gek(net, x, y, seq_len=seq_len)

                losses.append(loss)
                running_loss = .99 * running_loss + .01 * loss

                p, dp = net.get_params(dparams)

                if self.hp:
                    optim_cache = self.kernel(p, dp, self.hp, optim_cache)
                else:
                    optim_cache = self.kernel(p, dp, cache=optim_cache)

                n = n + 1

                now = time.time()
                if now - ts < 6.*elapsed:
                    elapsed = now - ts
                    cumtime += elapsed
                ts = now
                done = n / (epochs*num_batches)
                if cumtime > 10 and (n > 1 or epoch > 0):
                    remaining = cumtime/done - cumtime

                progress(done, running_loss, remaining)

                if plotting and n % plot_every == 0:
                    tacc, tcost, acc, cost = \
                        (Trainer.
                         cost_accuracy(net, data=data, samples=100,
                                       cost_on_training=plotparam[2],
                                       accuracy_on_training=plotparam[3],
                                       recurrent=recurrent))
                    perf[n // plot_every] = \
                        np.array([tacc, tcost, acc, cost])
                if responsive and not q.empty():
                    s = q.get_nowait()
                    if 'p' in s:
                        print('\bPaused...' + ' '*50, end='\r')
                        s = s.replace('p', '')
                        while True:
                            if not q.empty():
                                s += q.get_nowait()
                                if 'p' in s or 'c' in s:
                                    s = s.replace('p', '')
                                    break
                            time.sleep(0.5)

                    if 'c' in s:
                        print('\rAborted training loop...' + ' '*50, end='\r')
                        s = s.replace('c', '')
                        break
            else:
                continue
            break

        h = int(cumtime // 3600)
        m = int(cumtime % 3600//60)
        sec = int(cumtime % 60)
        print('\nIt took friggin\' {:2d}h{:2d}m{:2d}s'.format(h, m, sec))

        if plotting:
            print('Now plotting graphs...')
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(8, 8))
            cst, acc = fig.subplots(2, 1, sharex=True)
            xs = np.linspace(0., n/num_batches, n//plot_every+1)
            if plotparam[3]:
                cst.plot(xs, perf[:n//plot_every+1, 3], 'r-',
                         label='Training cost')
            if plotparam[1]:
                cst.plot(xs, perf[:n//plot_every+1, 1], 'b-',
                         label='Testing cost')
            cst.legend()
            if plotparam[2]:
                acc.plot(xs, perf[:n//plot_every+1, 2], 'r-',
                         label='Training accuracy')
            if plotparam[0]:
                acc.plot(xs, perf[:n//plot_every+1, 0], 'b-',
                         label='Testing accuracy')
            acc.legend()
            fig.show()

        if save and ((isinstance(saveif, str) and eval(saveif))
                     or saveif is None):
            print('Now saving graphs and the network...')
            path, _, post = net.save(date=False, perf=perf, losses=losses)
            if plotting:
                fig.savefig(path + 'perfplot_' + post + '.png')

        return losses, perf

    @staticmethod
    def vanilla_gek(net, x, y, **kwargs):
        """
        Gradient Estimation Kernel wrapping the gradient acquisition.

        Parameters
        ----------
        net : TYPE
            DESCRIPTION.
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        loss : TYPE
            DESCRIPTION.
        dparams : TYPE
            DESCRIPTION.

        """
        o, c = net.feedforward(x, test=False)
        dparams = net.backprop(c, y, test=False)

        loss = np.mean(cross_entropy(o, y))
        return loss, dparams

    @staticmethod
    def recurrent_gek(net, x, y, seq_len=64):
        caches = []
        zs = []
        loss = 0.
        reset = True
        for k in range(seq_len):
            o, c = net.feedforward(x[k::seq_len],
                                   test=False,
                                   reset=reset)
            reset = False
            zs.append(o)
            caches.append(c)
            loss += np.mean(cross_entropy(o, y[k::seq_len]))

        dparams = []
        for k in reversed(range(seq_len)):
            dp = net.backprop(caches[k],
                              y[k::seq_len],
                              test=False)
            if len(dparams) == 0:
                dparams = dp
            else:
                for ls, l in zip(dparams, dp):
                    if len(l) > 0:
                        for gs, g in zip(ls, l):
                            gs += g
        return loss/seq_len, dparams

    def sample(self, net, seed, vocab, samples=100, seedlen=np.inf, **kwargs):
        reset = True

        if not isinstance(seed, np.ndarray):
            if not isinstance(seed, str):
                raise TypeError('seed has to be either \
                                a data array or a string.')
            data = make_data(seed + vocab[0, 0], vocab=vocab)[0]
        else:
            data = [[seed]]

        for i in range(min(len(seed), seedlen)):
            x = data[0][0][i:i+1]
            print(vocab[0, np.argmax(x, axis=1)].squeeze(), end='')
            p, c = net.feedforward(x, reset=reset, **kwargs)
            reset = False
        o = np.random.choice(p.size, p=p.ravel())
        x = one_hot_encoding([o], p.size)
        print(vocab[0, o].squeeze(), end='')
        for i in range(samples):
            p, c = net.feedforward(x, **kwargs)
            o = np.random.choice(p.size, p=p.ravel())
            x = one_hot_encoding([o], p.size)
            print(vocab[0, o].squeeze(), end='')

    @staticmethod
    def performance(net, test_data, samples=-1):
        """
        Compute several performance metrics.

        Parameters
        ----------
        net : TYPE
            DESCRIPTION.
        test_data : TYPE
            DESCRIPTION.
        samples : TYPE, optional
            DESCRIPTION. The default is -1.

        Returns
        -------
        perf : TYPE
            DESCRIPTION.

        """
        """"""
        if samples < 0.:
            if samples > -1.:
                samples = -1
            samples = test_data[1].size / -samples
        samples = int(round(samples)) % test_data[0].shape[0]
        offset = np.random.randint(test_data[1].shape[0]-samples)
        x = test_data[0][offset:samples+offset]
        y = test_data[1][offset:samples+offset]

        o = net.feedforward(x, test=True)[0].reshape(samples, -1)
        num_cls = o.shape[1]

        p = np.argmax(o, axis=1)

        conf_mat = np.zeros((num_cls, num_cls), dtype=np.uint16)
        np.add.at(conf_mat, (y, p), 1)

        tp = np.diag(conf_mat)
        cond_pos = np.sum(conf_mat, axis=1)
        pred_pos = np.sum(conf_mat, axis=0)

        p_precision = tp / pred_pos
        p_recall = tp / cond_pos
        p_fscore = 2 * (p_precision * p_recall) / (p_precision + p_recall)
        p_informedness = p_precision + p_recall - 1.

        accuracy = np.sum(tp)/samples
        precision = np.sum(p_precision)/num_cls
        recall = np.sum(p_recall)/num_cls
        fscore = 2 * (precision * recall) / (precision + recall)
        informedness = precision + recall - 1.

        perf = {'accuracy': accuracy,
                'precision': (precision, p_precision),
                'recall': (recall, p_recall),
                'F-score': (fscore, p_fscore),
                'informedness': (informedness, p_informedness),
                'confusion': conf_mat}
        return perf

    @staticmethod
    def cost_accuracy(net, data, samples=1000,
                      cost_on_test=True,
                      accuracy_on_test=True,
                      cost_on_training=True,
                      accuracy_on_training=True,
                      recurrent=False):
        """
        Compute cost and accuracy metrics on both test and train datasets.

        Parameters
        ----------
        net : TYPE
            DESCRIPTION.
        data : TYPE
            DESCRIPTION.
        samples : TYPE, optional
            DESCRIPTION. The default is 1000.
        cost_on_test : TYPE, optional
            DESCRIPTION. The default is True.
        accuracy_on_test : TYPE, optional
            DESCRIPTION. The default is True.
        cost_on_training : TYPE, optional
            DESCRIPTION. The default is True.
        accuracy_on_training : TYPE, optional
            DESCRIPTION. The default is True.
        recurrent : boolean, optional
            DESCRIPTION, The default is False.

        Returns
        -------
        tacc : TYPE
            DESCRIPTION.
        tcost : TYPE
            DESCRIPTION.
        acc : TYPE
            DESCRIPTION.
        cost : TYPE
            DESCRIPTION.

        """
        tacc = 0.
        tcost = 0.
        acc = 0.
        cost = 0.
        train_data, test_data = data
        if accuracy_on_test or cost_on_test:
            if not recurrent:
                shuffle_data(test_data)
            x = test_data[0][:samples]
            y = test_data[1][:samples]
            _samples = y.shape[0]
            if recurrent:
                seq_len = np.maximum(_samples//16, _samples//(_samples//8))
                out = None
                reset = True
                for i in range(seq_len):
                    tout = net.feedforward(x[i::seq_len], reset=reset)[0]
                    reset = False
                    if out is None:
                        out = np.zeros((_samples, *tout.shape[1:]))
                    out[i::seq_len] = tout
            else:
                out = net.feedforward(x)[0]
            if accuracy_on_test:
                tacc = (np.sum(np.argmax(out, axis=1) == y[:, np.newaxis,
                                                           np.newaxis])
                        / float(_samples) * 100.)
            if cost_on_test:
                tcost = (np.sum(cross_entropy(out, y))
                         / _samples)

        if accuracy_on_training or cost_on_training:
            _samples = np.minimum(train_data[1].shape[0], samples)
            i = (0 if _samples == train_data[1].shape[0]
                 else np.random.randint(train_data[1].size-_samples))
            x = train_data[0][i:i+_samples]
            y = train_data[1][i:i+_samples]
            if recurrent:
                seq_len = np.maximum(_samples//16, _samples//(_samples//8))
                out = None
                reset = True
                for i in range(seq_len):
                    tout = net.feedforward(x[i::seq_len], reset=reset)[0]
                    reset = False
                    if out is None:
                        out = np.empty((_samples, *tout.shape[1:]))
                    out[i::seq_len] = tout
            else:
                out = net.feedforward(x)[0]

            if accuracy_on_training:
                acc = (np.sum(np.argmax(out, axis=1) == y[:, np.newaxis,
                                                          np.newaxis])
                       / float(_samples) * 100.)
            if cost_on_training:
                cost = (np.sum(cross_entropy(out, y))
                        / _samples)
        return tacc, tcost, acc, cost


__geks = list(filter(re.compile(".*_gek").match, dir(Trainer)))
