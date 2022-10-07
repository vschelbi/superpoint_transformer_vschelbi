import os
import h5py
import torch
from datetime import datetime
from superpoint_transformer.utils.tensor import tensor_idx


__all__ = ['dated_dir', 'read_hdf5']


def dated_dir(root, create=False):
    """Returns a directory path in root, named based on the current date
    and time.
    """
    date = '-'.join([
        f'{getattr(datetime.now(), x)}'
        for x in ['year', 'month', 'day']])
    time = '-'.join([
        f'{getattr(datetime.now(), x)}'
        for x in ['hour', 'minute', 'second']])
    dir_name = f'{date}_{time}'
    path = os.path.join(root, dir_name)
    if create and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def read_hdf5(path, idx=None, idx_keys=None, keys=None):
    """Read an HDF5 file and return its content as a dictionary.

    :param path: path to the HDF5 file
    :param idx: optional. int, list, numpy.ndarray, torch.Tensor used to
        index the elements in `keys_idx`. Supports fancy indexing
    :param idx_keys: optional. List of keys on which the indexing should
        be applied
    :param keys: optional. Indicates which keys should be loaded from
        the file, ignoring the rest.
    :return:
    """
    idx = tensor_idx(idx)
    idx_keys = [] if idx_keys is None or idx.shape[0] == 0 else idx_keys
    keys = [] if keys is None else keys
    data = {}

    with h5py.File(path, 'r') as f:
        for k in f.keys():

            # Read everything if there is no 'idx_keys' of 'keys'
            if idx_keys is None and keys is None:
                data[k] = torch.from_numpy(f[k][:])
                continue

            # Index the key elements if key is in 'idx_keys'
            if k in idx_keys:
                data[k] = torch.from_numpy(f[k][:])[idx]
                continue

            # Read everything if key is in 'keys'
            if k in keys:
                data[k] = torch.from_numpy(f[k][:])
                continue

    return data
