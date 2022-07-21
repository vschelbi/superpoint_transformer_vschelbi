import os
import re
import subprocess


def available_cpu_count():
    """ Number of available virtual or physical CPUs on this system, i.e.
    user/real as output by time(1) when called with an optimally scaling
    userspace-only program"""

    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                      open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0:
                return res
    except IOError:
        pass

    # Python 2.6+
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass

    # https://github.com/giampaolo/psutil
    try:
        import psutil
        return psutil.cpu_count()   # psutil.NUM_CPUS on old versions
    except (ImportError, AttributeError):
        pass

    # POSIX
    try:
        res = int(os.sysconf('SC_NPROCESSORS_ONLN'))

        if res > 0:
            return res
    except (AttributeError, ValueError):
        pass

    # Windows
    try:
        res = int(os.environ['NUMBER_OF_PROCESSORS'])

        if res > 0:
            return res
    except (KeyError, ValueError):
        pass

    # jython
    try:
        from java.lang import Runtime
        runtime = Runtime.getRuntime()
        res = runtime.availableProcessors()
        if res > 0:
            return res
    except ImportError:
        pass

    # BSD
    try:
        sysctl = subprocess.Popen(['sysctl', '-n', 'hw.ncpu'],
                                  stdout=subprocess.PIPE)
        scStdout = sysctl.communicate()[0]
        res = int(scStdout)

        if res > 0:
            return res
    except (OSError, ValueError):
        pass

    # Linux
    try:
        res = open('/proc/cpuinfo').read().count('processor\t:')

        if res > 0:
            return res
    except IOError:
        pass

    # Solaris
    try:
        pseudoDevices = os.listdir('/devices/pseudo/')
        res = 0
        for pd in pseudoDevices:
            if re.match(r'^cpuid@[0-9]+$', pd):
                res += 1

        if res > 0:
            return res
    except OSError:
        pass

    # Other UNIXes (heuristic)
    try:
        try:
            dmesg = open('/var/run/dmesg.boot').read()
        except IOError:
            dmesgProcess = subprocess.Popen(['dmesg'], stdout=subprocess.PIPE)
            dmesg = dmesgProcess.communicate()[0]

        res = 0
        while '\ncpu' + str(res) + ':' in dmesg:
            res += 1

        if res > 0:
            return res
    except OSError:
        pass

    raise Exception('Can not determine number of CPUs on this system')

print(available_cpu_count())

import numpy as np
import sys, os
import matplotlib.image as mpimg
import argparse
import ast

file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # this is for the .py script but does not work in a notebook
# file_path = os.path.dirname(os.path.abspath('')) # this is for a notebook
sys.path.append(file_path)
# sys.path.append(os.path.join(file_path, "grid-graph/python/bin"))
# sys.path.append(os.path.join(file_path, "parallel-cut-pursuit/python/wrappers"))


#-----------------------------------------------------------------------

# Data loading
from time import time
import glob
from torch_geometric.data import Data
from superpoint_transformer.datasets.kitti360 import read_kitti360_window
from superpoint_transformer.datasets.kitti360_config import KITTI360_NUM_CLASSES

i_window = 0
all_filepaths = sorted(glob.glob('/media/drobert-admin/DATA2/datasets/kitti360/shared/data_3d_semantics/*/static/*.ply'))
filepath = all_filepaths[i_window]

start = time()
data = read_kitti360_window(filepath, semantic=True, instance=False, remap=True)
print(f'Loading data {i_window+1}/{len(all_filepaths)}: {time() - start:0.3f}s')
print(f'Number of loaded points: {data.num_nodes} ({data.num_nodes // 10**6:0.2f}M)')

#-----------------------------------------------------------------------

# Voxelization
voxel = 0.05

### Pytorch Geometric
from torch_geometric.nn.pool import voxel_grid

start = time()
data_sub = voxel_grid(data.pos, size=0.1)
print(f'Data  voxelization at {voxel}m: {time() - start:0.3f}s')
print(f'Number of sampled points: {data_sub.shape[0]} ({data_sub.shape[0] // 10**6:0.2f}M, {100 * data_sub.shape[0] / data.num_nodes:0.1f}%)')

### TorchPoints3D
import torch
import re
from torch_geometric.nn.pool import voxel_grid
from torch_cluster import grid_cluster
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn.pool.consecutive import consecutive_cluster

MAPPING_KEY = 'mapping_index'

# Label will be the majority label in each voxel
_INTEGER_LABEL_KEYS = ["y", "instance_labels"]


def group_data(data, cluster=None, unique_pos_indices=None, mode="last", skip_keys=[]):
    """ Group data based on indices in cluster.
    The option ``mode`` controls how data gets agregated within each cluster.

    Parameters
    ----------
    data : Data
        [description]
    cluster : torch.Tensor
        Tensor of the same size as the number of points in data. Each element is the cluster index of that point.
    unique_pos_indices : torch.tensor
        Tensor containing one index per cluster, this index will be used to select features and labels
    mode : str
        Option to select how the features and labels for each voxel is computed. Can be ``last`` or ``mean``.
        ``last`` selects the last point falling in a voxel as the representent, ``mean`` takes the average.
    skip_keys: list
        Keys of attributes to skip in the grouping
    """

    assert mode in ["mean", "last"]
    if mode == "mean" and cluster is None:
        raise ValueError("In mean mode the cluster argument needs to be specified")
    if mode == "last" and unique_pos_indices is None:
        raise ValueError("In last mode the unique_pos_indices argument needs to be specified")

    num_nodes = data.num_nodes
    for key, item in data:
        if bool(re.search("edge", key)):
            raise ValueError("Edges not supported. Wrong data type.")
        if key in skip_keys:
            continue

        if torch.is_tensor(item) and item.size(0) == num_nodes:
            if mode == "last" or key == "batch" \
                    or key == SaveOriginalPosId.KEY \
                    or key == MAPPING_KEY:
                data[key] = item[unique_pos_indices]
            elif mode == "mean":
                is_item_bool = item.dtype == torch.bool
                if is_item_bool:
                    item = item.int()
                if key in _INTEGER_LABEL_KEYS:
                    item_min = item.min()
                    item = torch.nn.functional.one_hot(item - item_min)
                    item = scatter_add(item, cluster, dim=0)
                    data[key] = item.argmax(dim=-1) + item_min
                else:
                    data[key] = scatter_mean(item, cluster, dim=0)
                if is_item_bool:
                    data[key] = data[key].bool()
    return data


class SaveOriginalPosId:
    """ Transform that adds the index of the point to the data object
    This allows us to track this point from the output back to the input data object
    """

    KEY = "origin_id"

    def __init__(self, key=None):
        self.KEY = key if key is not None else self.KEY

    def _process(self, data):
        if hasattr(data, self.KEY):
            return data

        setattr(data, self.KEY, torch.arange(0, data.pos.shape[0]))
        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return self.__class__.__name__


class GridSampling3D:
    """ Clusters points into voxels with size :attr:`size`.
    Parameters
    ----------
    size: float
        Size of a voxel (in each dimension).
    quantize_coords: bool
        If True, it will convert the points into their associated sparse
        coordinates within the grid and store the value into a new
        `coords` attribute.
    mode: string:
        The mode can be either `last` or `mean`.
        If mode is `mean`, all the points and their features within a
        cell will be averaged. If mode is `last`, one random points per
        cell will be selected with its associated features.
    setattr_full_pos: bool
        If True, the input point positions will be saved into a new
        'full_pos' attribute. This memory-costly step may reveal
        necessary for subsequent local feature computation.
    """

    def __init__(self, size, quantize_coords=False, mode="mean", verbose=False,
                 setattr_full_pos=False):
        self._grid_size = size
        self._quantize_coords = quantize_coords
        self._mode = mode
        self._setattr_full_pos = setattr_full_pos
        if verbose:
            log.warning(
                "If you need to keep track of the position of your points, use "
                "SaveOriginalPosId transform before using GridSampling3D.")

            if self._mode == "last":
                log.warning(
                    "The tensors within data will be shuffled each time this "
                    "transform is applied. Be careful that if an attribute "
                    "doesn't have the size of num_points, it won't be shuffled")

    def _process(self, data):
        if self._mode == "last":
            data = shuffle_data(data)

        full_pos = data.pos
        coords = torch.round((data.pos) / self._grid_size)
        if "batch" not in data:
            cluster = grid_cluster(coords, torch.tensor([1, 1, 1]))
        else:
            cluster = voxel_grid(coords, data.batch, 1)
        cluster, unique_pos_indices = consecutive_cluster(cluster)

        data = group_data(data, cluster, unique_pos_indices, mode=self._mode)
        if self._quantize_coords:
            data.coords = coords[unique_pos_indices].int()

        data.grid_size = torch.tensor([self._grid_size])

        # Keep track of the initial full-resolution point cloud for
        # later use. Typically needed for local features computation.
        # However, for obvious memory-wary considerations, it is
        # recommended to delete the 'full_pos' attribute as soon as it
        # is no longer needed.
        if self._setattr_full_pos:
            data.full_pos = full_pos

        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(grid_size={}, quantize_coords={}, mode={})".format(
            self.__class__.__name__, self._grid_size, self._quantize_coords, self._mode
        )

start = time()
data_sub = GridSampling3D(size=voxel)(data.clone())
print(f'Data  voxelization at {voxel}m: {time() - start:0.3f}s')
print(f'Number of sampled points: {data_sub.num_nodes} ({data_sub.num_nodes // 10**6:0.2f}M, {100 * data_sub.num_nodes / data.num_nodes:0.1f}%)')

### Loïc's C implem
import superpoint_transformer.partition.utils.libpoint_utils as point_utils

# WARNING: the pruning must know the number of classes. All labels are
# offset to account for the -1 unlabeled points !
start = time()
xyz, rgb, labels, dump = point_utils.prune(data.pos.float().numpy(), voxel, (data.rgb * 255).byte().numpy(), data.y.byte().numpy() + 1, np.zeros(1, dtype='uint8'), KITTI360_NUM_CLASSES + 1, 0)
print(f'Data  voxelization at {voxel}m: {time() - start:0.3f}s')
print(f'Number of sampled points: {xyz.shape[0]} ({xyz.shape[0] // 10**6:0.2f}M, {100 * xyz.shape[0] / data.num_nodes:0.1f}%)')
