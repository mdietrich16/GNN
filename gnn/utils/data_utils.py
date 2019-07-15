# -*- coding: utf-8 -*-
"""
Utilities for data preparation and fetching.

Created on Wed May 22 15:50:38 2019

@author: maxid
"""
import pickle
import numpy as np

from utils.utils import one_hot_encoding


def shuffle_data(data, volumetric=True):
    """
    Shuffles the input data on its first data axis.

    Parameters
    ----------
    data : ndarray or array_like
        Input data which is to be permuted.
    volumetric : boolean, optional
        Wether the data is shaped `(n, c, w, h)` as in batched images.
        The default is True.

    Returns
    -------
    data : ndarray or array_like of same shape as the input
        The shuffled data.

    """
    ix = np.random.permutation(data[0].shape[0])
    data[0] = data[0][ix]
    data[1] = data[1][ix]
    return data


def load_mnist(file='mnist.pkl', folder=None, volumetric=True):
    r"""
    Load a pickled dataset with the same format as the MNIST dataset.

    Parameters
    ----------
    file : str, optional
        The filename with extension from which to load the data.
        The default is 'mnist.pkl'.
    folder : str or None, optional
        Either a path string in which the `file` is to be found or None to
        enable automatic search. All subdirectories of ``...\GNN\`` are
        searched. The default is None for automatic search.
    volumetric : boolean, optional
        Wether the data is shaped `(n, c, w, h)` as in batched images.
        If True, the data will have shape (n, 1, 28, 28) else (n, 784)
        The default is True.

    Raises
    ------
    FileNotFoundError
        If the specified folder does not exist or when the file can not be
        found.

    Returns
    -------
    training_data : list of two ndarrays
        The training data as given in the dataset.
    test_data : list of two ndarrays
        The test data as given in the dataset.

    """
    import os
    if not folder:
        import glob
        path = '.\\'
        if file not in os.listdir(path):
            while 'GNN' not in os.listdir(path):
                path = path + '..\\'
            files = glob.glob(path + '*\\' + file)
            if len(files) > 1:
                print('---!Warning!--- >>> File {} can be found in multiple \
                      locations! Using first occurrence.'.format(file))
            elif len(files) == 0:
                raise FileNotFoundError('You cannot load a non-existing file: \
                                        {}'.format(file))
            file = files[0]
    else:
        if not os.path.isdir(folder):
            raise FileNotFoundError('You cannot load data out of a \
                                    non-existing directory: {}'
                                    .format(folder))
        elif not os.path.isfile(folder+file):
            raise FileNotFoundError('You cannot load a non-existing file: {}'
                                    .format(folder+file))
        else:
            file = folder+file

    with open(file, 'rb') as f:
        mnist = pickle.load(f, encoding='latin1')
    if volumetric:
        inputs = (mnist['training_images'].reshape(-1, 1, 28, 28)
                                          .astype(np.float32))
        labels = mnist['training_labels'].reshape(-1)

        tinputs = (mnist['test_images'].reshape(-1, 1, 28, 28)
                                       .astype(np.float32))
        tlabels = mnist['test_labels'].reshape(-1)

    else:
        inputs = np.moveaxis(mnist['training_images'].reshape(-1, 784)
                             .astype(np.float32), 0, -1)
        labels = mnist['training_labels'].reshape(-1)

        tinputs = np.moveaxis(mnist['test_images'].reshape(-1, 784)
                              .astype(np.float32), 0, -1)
        tlabels = mnist['test_labels'].reshape(-1)

    training_data = [inputs/inputs.max(), labels]
    test_data = [tinputs/tinputs.max(), tlabels]

    return training_data, test_data


def make_data(string, test_string=None, dtype=np.float64,
              load_file=False, folder=None, vocab=None):
    """
    Generate formatted data from a string to use in NLP networks.

    Parameters
    ----------
    string : str
        Input string or filename.
    test_string : str, optional
        String to generate testing data from. The default is None.
    load_file : boolean, optional
        If True, use `string` as a filename and possibly `folder` as the folder
        path. The default is False.
    folder : string or None, optional
        The folder in which the file is to be found. This defaults to None.
    vocab : array, optional
        Use this as the vocabulary if given. This defaults to None.
    Returns
    -------
    data : list of two lists of two ndarrays of shape (n, c, 1, 1)
        Resulting data with the input string converted to input and label
        arrays in the first element and the optional `test_string` with the
        same format in the second entry.

    """

    if load_file:
        print('Loading file {}...'.format(string))
        import os
        if not folder:
            import glob
            path = '.\\'
            if string not in os.listdir(path):
                while 'GNN' not in os.listdir(path):
                    path = path + '..\\'
                files = glob.glob(path + '**\\' + string, recursive=True)
                if len(files) > 1:
                    print('---!Warning!--- >>> File {} can be found in \
                          multiple locations! Using first occurrence.'.
                          format(string))
                elif len(files) == 0:
                    raise FileNotFoundError('You cannot load a non-existing \
                                             file: {}'.format(string))
                file = files[0]
            else:
                file = string
        else:
            if not os.path.isdir(folder):
                raise FileNotFoundError('You cannot load data out of a \
                                        non-existing directory: {}'
                                        .format(folder))
            elif not os.path.isfile(folder+string):
                raise FileNotFoundError('You cannot load a non-existing \
                                        file: {}'.format(folder+string))
            else:
                file = folder+string
        with open(file, 'r') as f:
            string = f.read()

    if vocab is None:
        vocab = np.array(sorted(list(set(string))), ndmin=2)
        if test_string:
            vocab.update(set(test_string))
    s = np.array(list(string), ndmin=2)
    labels = (s.T == vocab).nonzero()[1]
    inputs = one_hot_encoding(labels, vocab.size, dtype=dtype)[:-1]
    train = [inputs, labels[1:]]
    test = []
    if test_string:
        s = np.array(list(test_string), ndmin=2)
        labels = (s.T == vocab).nonzero()[1]
        inputs = one_hot_encoding(labels, vocab.size)[:-1]
        test = [inputs, labels[1:]]
    else:
        test = [inputs.copy(), labels[1:].copy()]
    return [train, test], vocab


def gen_mini_batches(data, batch_size, strict=False):
    """
    Generate batches of the given data.

    Parameters
    ----------
    data : list of two ndarrays
        Data which is to be partitioned. First entry a ndarray of shape
        (n, c, w, h), second entry a ndarray of shape (n, 1) which will be
        split up in `n//batch_size` (batch_size, c, w, h) and (batch_size, 1)
        partitions or 'mini_batches'.
    batch_size : TYPE
        Langth of each resualting partition.

    Returns
    -------
    iterable
        An iterable over views of the data arrays.

    """
    data_len = data[0].shape[0]
    if strict:
        data_len -= data_len % batch_size
    return ((data[0][k:k+batch_size], data[1][k:k+batch_size])
            for k in range(0, data_len, batch_size))
