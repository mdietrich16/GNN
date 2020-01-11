# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:45:32 2019

@author: maxid
"""

import numpy as np


def progress(p, cost=None, tr=None):
    s = ('\r{:5.2f}% |'.format(p*100.) + int(np.ceil(p*25))*'#' +
         int(np.floor((1.-p)*25 + 1e-5))*' ' + '|')
    if cost is not None:
        s += ' Loss: {:.3f}'.format(cost)
    if tr is not None:
        h = int(tr // 3600)
        m = int(tr % 3600//60)
        sec = int(tr % 60)
        s += '\tRemaining time: {:2d}h{:2d}m{:2d}s'.format(h, m, sec)
    print(s, end='\r')


def make_rgb(string, r, g, b, background=False):
    if isinstance(r, float) or isinstance(g, float) or isinstance(b, float):
        r, g, b = int(r), int(g), int(b)
    return ('\033[38;{};{};{};{}m'.format(5 if background else 2, r, g, b) +
            string + '\033[0m')


def one_hot_encoding(x, size, dtype=np.float64):
    y = np.zeros((len(x), size, 1, 1), dtype=dtype)
    y[np.arange(len(x)), x] = 1.
    return y


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def sigmoid_prime(o):
    return o * (1-o)


def tanh_prime(o):
    return 1. - np.square(o)


def RELU(x):
    z = np.copy(x)
    z[x < 0] = 0
    return z


def RELU_prime(x):
    o = np.ones_like(x)
    o[x < 0] = 0.
    return o


def ELU(x, alpha=1.):
    z = np.copy(x)
    b = x < 0
    z[b] = alpha*(np.exp(x[b]) - 1.)
    return z


def ELU_prime(o, alpha=1.):
    z = np.ones_like(o)
    b = o < 0
    z[b] = o[b] + alpha
    return z


def dropout(x, p=0.5):
    d = np.random.random(x.shape) > p
    z = np.copy(x)
    z[d] = 0.
    return z, d


def dropout_prime(o, d):
    z = np.copy(o)
    z[d] = 0.
    return z


def softmax(x, alpha=1.):
    e = np.exp(alpha*(x - np.max(x, axis=1, keepdims=True)))
    return e/np.sum(e, axis=1, keepdims=True)


def softmax_prime(o):
    return 1.


def cross_entropy(o, y):
    return -np.log(o[np.arange(o.shape[0]), y] + 1e-45)


def cross_entropy_prime(o, y, alpha=1.):
    dCdo = o.copy()
    m, _, n, o = o.shape
    i, j, k = np.arange(m), np.arange(n), np.arange(o)
    dCdo[i, y, j, k] -= 1
    return dCdo


def quadratic_cost(o, y):
    return 0.5 * np.sum(np.square(o - one_hot_encoding(y, 10)),
                        axis=0)


def quadratic_cost_prime(o, y):
    return o - one_hot_encoding(y, 10)


def find_factors(n):
    n = int(n)
    f = []
    for i in range(1, int(np.sqrt(n))+1):
        if n % i == 0:
            f.append((i, n//i))
    return f


def visualize_kernels(img=None, kernels=None, activations=None):
    import matplotlib.pyplot as plt
    if img is not None:
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(img.transpose(0, 2, 3, 1).squeeze(), cmap='gray')
    if kernels is not None:
        fig = plt.figure(figsize=(8, 8))
        i = 1
        fs = find_factors(kernels.shape[0])
        h, w = fs[-1]
        for k in kernels:
            fig.add_subplot(h, w, i)
            plt.imshow(k.transpose(1, 2, 0).sum(axis=2), cmap='gray')
            i += 1
    if activations is not None:
        fig = plt.figure(figsize=(8, 8))
        i = 1
        fs = find_factors(activations.shape[1])
        h, w = fs[-1]
        for k in activations[0]:
            fig.add_subplot(h, w, i)
            plt.imshow(k, cmap='gray')
            i += 1


def cov(data):
    mean = np.mean(data, axis=0, keepdims=True)
    return data.T.dot(data)/(data.shape[0]-1) - mean.T.dot(mean)


def norm(x):
    return np.sqrt(np.sum(np.square(x)))


def normalize(x):
    ln = norm(x)
    if ln < 1e-10:
        return x
    else:
        return x/ln


def eigen(x, m, number=1000, eta=1e-2):
    data = x.copy()
    n = x.shape[0]
    eta = np.power(eta, n)
    if m > n:
        raise ValueError('Cannot find more eigenvalues than dimensions')
    eigen_pairs = []
    lmbda = 0.
    import time
    start = time.time()
    for e in range(m):
        vec = normalize(np.random.randn(n, 1))
        diff = np.inf
        for i in range(number):
            vec_ = normalize(data.dot(vec))
            diff = 1.-np.abs(np.dot(vec.T, vec_))
            vec = vec_
            if diff < eta:
                break
            if i > 0 or e > 0:
                elapsed = time.time() - start
                done = (e + i/number) / m
                progress(done, float(diff), elapsed/done - elapsed)
        lmbda = vec.T.dot(x.dot(vec))/vec.T.dot(vec)
        data = data - lmbda/norm(vec)*vec.dot(vec.T)
        eigen_pairs.append((lmbda, vec))
    return eigen_pairs


def PCA(x, y=None, ndim=2, number=1000, eta=1e-2, plot=True):
    if plot:
        ndim = np.clip(ndim, 2, 3)
    x_ = x.reshape(x.shape[0], -1)
    x_ = np.nan_to_num((x_ - x_.mean(axis=0, keepdims=True))/x_.std(axis=0, keepdims=True))
    ndim = x_.shape[1] if ndim < 0 else ndim
    covariance = cov(x.reshape(x.shape[0], -1))
    eigens = eigen(covariance, ndim, number, eta)
    transform = np.concatenate(tuple([eigen[1] for eigen in eigens]), axis=1)
    x_ = x_.dot(transform)
    if plot:
        import matplotlib.pyplot as plt
        if ndim == 2:
            fig, ax = plt.subplots()
        else:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        cmap = plt.get_cmap('gist_rainbow')
        size = 201.6546088 / np.sqrt(x.shape[0]) - 0.8518268264
        if ndim == 2:
            for i in range(10):
                ax.scatter(x_[y == i, 0], x_[y == i, 1],
                           c=[cmap(i/10)],
                           s=size,
                           label='Class {}'.format(i))
        else:
            for i in range(10):
                ax.scatter(x_[y == i, 0], x_[y == i, 1], x_[y == i, 2],
                           c=[cmap(i/10)],
                           s=size,
                           label='Class {}'.format(i))
        lgnd = ax.legend()
        for h in lgnd.legendHandles:
            h._sizes = [30]
    return transform, eigens
