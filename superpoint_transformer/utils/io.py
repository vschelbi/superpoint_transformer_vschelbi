import os
import h5py
import torch
import socket
import numpy as np
from datetime import datetime
from superpoint_transformer.utils.tensor import tensor_idx, numpyfy
from superpoint_transformer.utils.sparse import dense_to_csr, csr_to_dense


__all__ = [
    'dated_dir', 'select_hdf5_data', 'host_data_root', 'select_hdf5_data',
    'save_dense_to_csr_hdf5']


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


def save_dense_to_csr_hdf5(x, f, x32=True):
    """Compress a 2D tensor with CSR format and save it in an
    already-open HDF5.

    :param x: 2D torch.Tensor
    :param f: h5 file path of h5py.File or h5py.Group
    :param x32: bool
        Convert 64-bit data to 32-bit before saving.
    :return:
    """
    if not isinstance(f, (h5py.File, h5py.Group)):
        with h5py.File(f, 'w') as file:
            save_dense_to_csr_hdf5(x, file, x32=x32)
        return

    assert isinstance(x, torch.Tensor) and x.dim() == 2

    pointers, columns, values = dense_to_csr(x)
    d = numpyfy(pointers, x32=x32)
    f.create_dataset('pointers', data=d, dtype=d.dtype)
    d = numpyfy(columns, x32=x32)
    f.create_dataset('columns', data=d, dtype=d.dtype)
    d = numpyfy(values, x32=x32)
    f.create_dataset('values', data=d, dtype=d.dtype)
    d = np.array(x.shape)
    f.create_dataset('shape', data=d, dtype=d.dtype)


def load_csr_hdf5_to_dense(f, idx=None):
    """Read an HDF5 file of group produced using `dense_to_csr_hdf5` and
    return the dense tensor. An optional idx can be passed to only read
    corresponding rows from the dense tensor.

    :param f: h5 file path of h5py.File or h5py.Group
    :param idx: int, list, numpy.ndarray, torch.Tensor
        Used to select and read only some rows of the dense tensor.
        Supports fancy indexing
    :return:
    """
    KEYS = ['pointers', 'columns', 'values', 'shape']

    if not isinstance(f, (h5py.File, h5py.Group)):
        with h5py.File(f, 'r') as file:
            out = load_csr_hdf5_to_dense(file, idx=idx)
        return out

    assert all(k in f.keys() for k in KEYS)

    idx = tensor_idx(idx)

    if idx is None or idx.shape[0] == 0:
        pointers = torch.from_numpy(f['pointers'][:])
        columns = torch.from_numpy(f['columns'][:])
        values = torch.from_numpy(f['values'][:])
        shape = torch.from_numpy(f['shape'][:])
        return csr_to_dense(pointers, columns, values, shape=shape)

    # Read only pointers start and end indices based on idx
    ptr_start = torch.from_numpy(f['pointers'][:])[idx]
    ptr_end = torch.from_numpy(f['pointers'][:])[idx + 1]

    # Create the new pointers
    pointers = torch.cat([
        torch.zeros(1, dtype=ptr_start.dtype),
        torch.cumsum(ptr_end - ptr_start, 0)])

    # Create the indexing tensor to select and order values.
    # Simply, we could have used a list of slices but we want to
    # avoid for loops and list concatenations to benefit from torch
    # capabilities.
    sizes = pointers[1:] - pointers[:-1]
    val_idx = torch.arange(pointers[-1])
    val_idx -= torch.arange(pointers[-1] + 1)[
        pointers[:-1]].repeat_interleave(sizes)
    val_idx += ptr_start.repeat_interleave(sizes)

    # Read the columns and values, now we have computed the val_idx.
    # Make sure to update the output shape too, since the rows have been
    # indexed
    columns = torch.from_numpy(f['columns'][:])[val_idx]
    values = torch.from_numpy(f['values'][:])[val_idx]
    shape = torch.from_numpy(f['shape'][:])
    shape[0] = idx.shape[0]

    return csr_to_dense(pointers, columns, values, shape=shape)


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

        from time import time
        start = time()

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

        print(f'  reading {k:<20}: {time() - start:0.5f}s')

    return data
