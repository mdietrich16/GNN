# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:43:12 2019

@author: maxid
"""

import numpy as np


def im2col(x, f_shape, p=1, s=1):

    N, C, H, W = x.shape
    K, C1, Fh, Fw = f_shape

    assert C == C1, 'Idi nahui! Channels are wong'
    assert (H + 2*p - Fh) % s == 0, 'Amena! Height is wong'
    assert (W + 2*p - Fw) % s == 0, 'Cyka! Width is wong'

    padx = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    out_H = int((H + 2*p - Fh)/s + 1)
    out_W = int((W + 2*p - Fw)/s + 1)

    i0 = np.repeat(np.arange(Fh), Fw)
    i0 = np.tile(i0, C)
    i1 = s * np.repeat(np.arange(out_H), out_W)
    j0 = np.tile(np.arange(Fw), Fh*C)
    j1 = s * np.tile(np.arange(out_W), out_H)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), Fh*Fw).reshape(-1, 1)
    out = padx[:, k, i, j].transpose(1, 2, 0).reshape(Fh*Fw*C, -1)
    return out, K, out_H, out_W, N


def conv(x, w, b, stride=1, pad=1):
    """Inner mechanics of a convolutional Layer.

    Parameters
    ----------
    x : ndarray
        Input matrix of shape (n, c, w, h) with n datapoints, c channels,
        and width w, height h.
    w : ndarray
        Weight matrix of shape (k, c, w, h) with k kernels (ouptuts), c input
        channels and width w and height h.
    b : ndarray
        Biases with shape (k,1).
    stride : int
        Description of parameter `stride`. The default is 1.
    pad : int
        Description of parameter `pad`. The default is 1.

    Returns
    -------
    type
        Description of returned object.

    """
    z, K, H, W, N = im2col(x, w.shape, pad, stride)
    o = w.reshape(K, -1).dot(z) + b.reshape(-1, 1)
    o = o.reshape(K, H, W, N).transpose(3, 0, 1, 2)
    return o, z


def col2im_indices(cols, x_shape, HH=3, WW=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    out_H = int((H + 2*padding - HH)/stride + 1)
    out_W = int((W + 2*padding - WW)/stride + 1)

    i0 = np.repeat(np.arange(HH), WW)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_H), out_W)
    j0 = np.tile(np.arange(WW), HH*C)
    j1 = stride * np.tile(np.arange(out_W), out_H)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), HH*WW).reshape(-1, 1)
    cols_reshaped = cols.reshape(C * HH * WW, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


def conv_back_im2col(dout, x_cols, x_shape, w, b, pad=1, stride=1):
    db = np.sum(dout, axis=(0, 2, 3)).reshape(-1, 1)
    K, C, HH, WW = w.shape
    dout_r = dout.transpose(1, 2, 3, 0).reshape(K, -1)
    dw = dout_r.dot(x_cols.T).reshape(K, C, HH, WW)
    dx_cols = w.reshape(K, -1).T.dot(dout_r)
    dx = col2im_indices(dx_cols, x_shape, HH, WW, pad, stride)
    return dx, dw, db


def conv_fast(x, w, b, stride=1, pad=1):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    # Check dimensions
    assert (W + 2 * pad - WW) % stride == 0, 'width does not work'
    assert (H + 2 * pad - HH) % stride == 0, 'height does not work'

    # Pad the input
    p = pad
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    # Figure out output dimensions
    H += 2 * pad
    W += 2 * pad
    out_h = int((H - HH) / stride + 1)
    out_w = int((W - WW) / stride + 1)

    # Perform an im2col operation by picking clever strides
    shape = (C, HH, WW, N, out_h, out_w)
    strides = (H * W, W, 1, C * H * W, stride * W, stride)
    strides = x.itemsize * np.array(strides)
    x_stride = np.lib.stride_tricks.as_strided(x_padded,
                                               shape=shape, strides=strides)
    x_cols = np.ascontiguousarray(x_stride)
    x_cols.shape = (C * HH * WW, N * out_h * out_w)
    # Now all our convolutions are a big matrix multiply
    res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)
    # Reshape the output
    res.shape = (F, N, out_h, out_w)
    out = res.transpose(1, 0, 2, 3)

    out = np.ascontiguousarray(out)
    return out, x_cols


def conv_back_strides(dout, x_cols, x_shape, w, b, stride=1, pad=1):

    N, C, H, W = x_shape
    F, _, HH, WW = w.shape
    _, _, out_h, out_w = dout.shape

    db = np.sum(dout, axis=(0, 2, 3))

    dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    dx_cols = w.reshape(F, -1).T.dot(dout_reshaped)
    dx_cols.shape = (C, HH, WW, N, out_h, out_w)
    dx = naive_strides_back(dx_cols, N, C, H, W, HH, WW, pad, stride)
    return dx, dw, db


def naive_strides_back(cols, N, C, H, W, HH, WW, pad, stride):
    out_h = (H + 2 * pad - HH) // stride + 1
    out_w = (W + 2 * pad - WW) // stride + 1
    x_padded = np.zeros((N, C, H+2*pad, W+2*pad))
    for n in range(N):
        for c in range(C):
            for hh in range(HH):
                for ww in range(WW):
                    for h in range(out_h):
                        for w in range(out_w):
                            x_padded[n, c, stride*h+hh, stride*w+ww] += \
                                cols[c, hh, ww, n, h, w]
    return x_padded[:, :, pad:-pad, pad:-pad]


def pool_slow(x, size):
    N, C, H, W = x.shape
    z, K, out_H, out_W, _ = im2col(x.reshape(N*C, 1, H, W),
                                   (N*C, 1, size, size), p=0, s=size)
    return np.max(z, axis=0).reshape(out_H, out_W, N, K).transpose(2, 3, 0, 1)


def pool(x, size):
    N, C, H, W = x.shape
    x_reshaped = x.reshape(N, C, H // size, size, W // size, size)
    ix = (x_reshaped.transpose(0, 1, 2, 4, 3, 5)
                    .reshape(N, C, H // size, W // size, size*size)
                    .argmax(axis=4))
    out = x_reshaped.max(axis=3).max(axis=4)
    return out, ix


def pool_back(dout, ix, shape, size):
    N, C, H, W = shape
    dx = np.zeros(shape)
    m, n, o, p = np.indices(dout.shape).reshape(4, -1)
    o, p = o*size, p*size
    i, j, k, h = np.unravel_index(ix.flatten(),
                                  (shape[0], shape[1], size, size))
    dx[i+m, j+n, k+o, h+p] = dout.flatten()
    return dx
