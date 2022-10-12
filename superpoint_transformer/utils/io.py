import os
import h5py
import torch
import socket
from datetime import datetime
from superpoint_transformer.utils.tensor import tensor_idx


__all__ = [
    'dated_dir', 'select_hdf5_data', 'host_data_root', 'select_hdf5_data']


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


def host_data_root():
    """Read the host machine's name and return the known $DATA_ROOT
    directory
    """
    #TODO: remove this for deployment
    HOST = socket.gethostname()
    if HOST == 'DEL2001W017':
        DATA_ROOT = '/media/drobert-admin/DATA2/datasets'
    elif HOST == 'HP-2010S002':
        DATA_ROOT = '/var/data/drobert/datasets'
    elif HOST == '9c81b1a54ad8':
        DATA_ROOT = '/raid/dataset/pointcloud/data'
    else:
        raise NotImplementedError(
            f"Unknown host '{HOST}', cannot set DATA_ROOT")
    return DATA_ROOT


def select_hdf5_data(f, idx=None, idx_keys=None, keys=None):
    """Read an HDF5 file and return its content as a dictionary.

    :param path: HDF5 file opened using
    :param idx: optional. int, list, numpy.ndarray, torch.Tensor used to
        index the elements in `keys_idx`. Supports fancy indexing
    :param idx_keys: optional. List of keys on which the indexing should
        be applied
    :param keys: optional. Indicates which keys should be loaded from
        the file, ignoring the rest.
    :return:
    """
    assert isinstance(f, (h5py.File, h5py.Group))

    idx = tensor_idx(idx)
    idx_keys = None if idx_keys is None or idx.shape[0] == 0 else idx_keys
    keys = None if keys is None else keys
    data = {}

    for k in f.keys():

        # Read everything if there is no 'idx_keys' or 'keys'
        if idx_keys is None and keys is None:
            data[k] = torch.from_numpy(f[k][:])

        # Index the key elements if key is in 'idx_keys'
        elif idx_keys is not None and k in idx_keys:
            data[k] = torch.from_numpy(f[k][:])[idx]

        # Read everything if key is in 'keys'
        elif keys is not None and k in keys:
            data[k] = torch.from_numpy(f[k][:])

        # By default, convert int32 to int64, might cause issues for
        # tensor indexing otherwise
        if k in data.keys() and data[k].dtype == torch.int32:
            data[k] = data[k].long()

    return data
