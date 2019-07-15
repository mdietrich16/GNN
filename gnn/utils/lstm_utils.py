# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 20:02:11 2019

@author: TheBeast
"""

import numpy as np
from gnn.utils.utils import sigmoid
from gnn.utils.utils import sigmoid_prime, tanh_prime, softmax
from gnn.utils.utils import cross_entropy, cross_entropy_prime as lossp


def LSTM(x, w, b, h0, c0):
    X = np.concatenate((x, h0), axis=1)
    d = h0.shape[1]
    d2 = d*2
    d3 = d*3

    z = X.dot(w) + b
    zf = np.empty_like(z)
    zf[:, :d3] = sigmoid(z[:, :d3])
    zf[:, d3:] = np.tanh(z[:, d3:])
    c = zf[:, d:d2] * c0 + zf[:, :d] * zf[:, d3:]
    cf = np.tanh(c)
    h = zf[:, d2:d3] * cf
    cache = (h, c, X, c0, cf, z, zf)
    return cache


def LSTM_back(dout, w, b, cache, dlast):
    h, c, X, c0, cf, z, zf = cache

    d = w.shape[1]//4

    d2 = 2*d
    d3 = 3*d

    dz = np.empty_like(z)
    dnext = np.empty_like(dlast)

    dh = dout + dlast[0]
    dz[:, d2:d3] = cf * dh
    dc = tanh_prime(cf) * zf[:, d2:d3] * dh + dlast[1]
    dz[:, d:d2] = dc * c0
    dnext[1] = zf[:, d:d2] * dc
    dz[:, :d] = zf[:, d3:] * dc
    dz[:, d3:] = zf[:, :d] * dc

    dz[:, d3:] = tanh_prime(zf[:, d3:]) * dz[:, d3:]
    dz[:, :d3] = sigmoid_prime(zf[:, :d3]) * dz[:, :d3]

    dw = X.T.dot(dz)
    dx, dnext[0] = np.split(dz.dot(w.T), [-d], axis=1)
    return dx, dw, dz.sum(axis=0, keepdims=True), dnext


def test():
    #xs = [np.random.randn(5, 3), np.random.randn(5, 3), np.random.randn(5, 3)]
    xs = [np.zeros((5, 3)), np.zeros((5, 3)), np.zeros((5, 3))]
    xs[0][:, 1] = 1
    xs[1][:, 1] = 1
    xs[2][:, 2] = 1

    a = 5
    I, H = 3, 2
    for k in range(1):
        w = np.random.randn(H+I, 4*H)
        b = np.zeros((1, 4*H))
        b[:, 2:4] = 1.
        Wy = np.random.randn(H, I)
        By = np.zeros((1, I))
        h0 = np.zeros((a, H))
        c0 = np.copy(h0)

        c, h = c0.copy(), h0.copy()

        caches = []
        ls = []
        ps = []

        for n in range(len(xs)):
            cache = LSTM(xs[n], w, b, h, c)
            h = cache[0].copy()
            c = cache[1].copy()
            e = h.dot(Wy) + By
            p = softmax(e)
            caches.append(cache)
            ps.append(p[:, :, np.newaxis, np.newaxis])
            ls.append(cross_entropy(p, 0))

        dnext = (h0, c0)
        d = 0
        for n in reversed(range(len(xs))):
            do = lossp(ps[n], 0).reshape(5,3)
            dWy = caches[n][0].T.dot(do)
            dBy = do.sum(axis=0, keepdims=True)
            dh = do.dot(Wy.T)

            dx, dW, dB, dnext = LSTM_back(dh, w, b, caches[n], dnext)
            d += dW[0, 0]

        c = np.zeros((a, H))
        h = np.copy(c)

        w[0, 0] += 1e-8
        g = 0
        def addat(a, ix, val):
            np.add.at(a, ix, val)
            return a
        for n in range(len(xs)):
            x = xs[n]
            #x[0, 0] += 1e-8
            cache = LSTM(x, w, b, h, c)
            h = cache[0].copy()
            c = cache[1].copy()
            e = h.dot(Wy) + By
            p = softmax(e)
            l = cross_entropy(p, 0)
            g += np.sum(l - ls[n])/1e-8
        print(g, d)
        print(np.allclose(d, g, atol=1e-5))