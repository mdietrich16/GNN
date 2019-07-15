# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:45:32 2019

@author: maxid
"""

import numpy as np
from numba import njit, prange

def progress(p, cost=None, tr=None):
    s = ('\r{:5.2f}% |'.format(p*100.) + int(np.ceil(p*25))*'#' +
         int(np.floor((1.-p)*25 + 1e-5))*' ' + '|')
    if cost:
        s += ' Loss: {:.3f}'.format(cost)
    if tr:
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


def one_hot_encoding(x, size, dtype=np.float32):
    y = np.zeros((len(x), size, 1, 1), dtype=dtype)
    y[np.arange(len(x)), x] = 1.
    return y


# @njit(parallel=True, fastmath=True)
# def one_hot_encoding(x, size):
#     le = len(x)
#     s = np.zeros((le, size, 1, 1), dtype=x.dtype)
#     for i in prange(le):
#         s[i, x[i]] = 1.
#     return s


@njit(parallel=True, fastmath=True)
def sigmoid(x):
    # return np.exp(-np.logaddexp(0, -x))
    return 1./(1. + np.exp(-x))


@njit(parallel=True, fastmath=True)
def sigmoid_prime(o):
    return o * (1-o)


@njit(parallel=True)
def tanh_prime(o):
    return 1. - np.square(o)


@njit(parallel=True)
def RELU(x):
    z = np.copy(x)
    z.reshape(-1)[(x < 0).ravel()] = 0
    return z


@njit(parallel=True)
def RELU_prime(x):
    o = np.ones_like(x)
    o.reshape(-1)[(x < 0).ravel()] = 0.
    return o


@njit(parallel=True, fastmath=True)
def ELU(x, alpha=1.):
    z = np.copy(x)
    b = (x < 0).ravel()
    z.reshape(-1)[b] = alpha*(np.exp(x.reshape(-1)[b]) - 1.)
    return z


@njit(parallel=True, fastmath=True)
def ELU_prime(o, alpha=1.):
    z = np.ones_like(o)
    b = (o < 0).ravel()
    z.reshape(-1)[b] = o.reshape(-1)[b] + alpha
    return z


@njit(parallel=True)
def dropout(x, p=0.5):
    d = np.random.random(x.size) > p
    z = np.copy(x)
    z.reshape(-1)[d] = 0.
    return z, d


@njit(parallel=True, fastmath=True)
def dropout_prime(o, d):
    z = np.copy(o)
    z.reshape(-1)[d] = 0.
    return z


@njit(parallel=True, fastmath=True)
def softmax(x, alpha=1.):
    e = np.empty_like(x)
    for i in prange(x.shape[0]):
        m = x[i]
        r = np.exp(alpha * (m - np.max(m)))
        r = r/np.sum(r)
        e[i] = r
    return e


def softmax_prime(o):
    return 1.


@njit(parallel=True, fastmath=True)
def cross_entropy(o, y):
    s = np.empty((y.size, *o.shape[2:]), dtype=o.dtype)
    for i in prange(y.size):
        s[i] = -np.log(o[i, y[i]])
    return s


@njit(parallel=True, fastmath=True)
def cross_entropy_prime(o, y, alpha=1.):
    dCdo = o.copy()
    m, _, n, o = o.shape
    for i in prange(m):
        for j in prange(n):
            for k in prange(o):
                p = y[i]
                dCdo[i, p, j, k] -= 1.
    dCdo *= alpha
    return dCdo


def quadratic_cost(o, y):
    return 0.5 * np.sum(np.square(o - one_hot_encoding(y, 10)), axis=0)


def quadratic_cost_prime(o, y):
    return o - one_hot_encoding(y, 10)


def find_factors(n):
    n = int(n)
    f = []
    for i in range(1, int(np.sqrt(n))):
        if n % i == 0:
            f.append((i, n//i))
    return f


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
    for e in range(m):
        vec = normalize(np.random.randn(n, 1))
        diff = np.inf
        for i in range(number):
            vec_ = normalize(data.dot(vec))
            diff = 1.-np.abs(np.dot(vec.T, vec_))
            vec = vec_
            if diff < eta:
                break
        lmbda = vec.T.dot(x.dot(vec))/vec.T.dot(vec)
        data = data - lmbda/norm(vec)*vec.dot(vec.T)
        eigen_pairs.append((lmbda, vec))
    return eigen_pairs
