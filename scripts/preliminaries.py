import re
import subprocess
from torch_geometric.nn.pool.consecutive import consecutive_cluster


#-----------------------------------------------------------------------

# voxel = 5
voxel = 0.05
radius = 1
# radius = 10
k_min = 5
# k_min = 2
k_feat = 30
# k_feat = 5
k_adjacency = 10
# k_adjacency = 3
lambda_edge_weight = 1

#-----------------------------------------------------------------------

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

print(f'CPUs available: {available_cpu_count()}')

import numpy as np
import torch
import sys, os

file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # this is for the .py script but does not work in a notebook
# file_path = os.path.dirname(os.path.abspath(''))
sys.path.append(file_path)
# sys.path.append(os.path.join(file_path, "grid-graph/python/bin"))
# sys.path.append(os.path.join(file_path, "parallel-cut-pursuit/python/wrappers"))
sys.path.append(os.path.join(file_path, "superpoint_transformer/partition/grid_graph/python/bin"))
sys.path.append(os.path.join(file_path, "superpoint_transformer/partition/parallel_cut_pursuit/python/wrappers"))

#-----------------------------------------------------------------------

from time import time
import glob
from superpoint_transformer.data import Data
from superpoint_transformer.datasets.kitti360 import read_kitti360_window
from superpoint_transformer.datasets.kitti360_config import KITTI360_NUM_CLASSES

# DATA_ROOT
DATA_ROOT = '/media/drobert-admin/DATA2'
# DATA_ROOT = '/var/data/drobert'

i_window = 0
all_filepaths = sorted(glob.glob(DATA_ROOT + '/datasets/kitti360/shared/data_3d_semantics/*/static/*.ply'))
filepath = all_filepaths[i_window]

start = time()
data = read_kitti360_window(filepath, semantic=True, instance=False, remap=True)
print(f'Loading data {i_window+1}/{len(all_filepaths)}: {time() - start:0.3f}s')
print(f'Number of loaded points: {data.num_nodes} ({data.num_nodes // 10**6:0.2f}M)')

# Offset labels by 1 to account for unlabelled points -> !!!!!!!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!!!!
data.y += 1
KITTI360_NUM_CLASSES += 1

#-----------------------------------------------------------------------

from superpoint_transformer.transforms import GridSampling3D

voxel = 0.05
# voxel = 1

# GPU
torch.cuda.synchronize()
start = time()
n_in = data.num_nodes
data = GridSampling3D(size=voxel, bins={'y': KITTI360_NUM_CLASSES})(data.cuda()).cpu()
torch.cuda.synchronize()
print(f'Data voxelization at {voxel}m: {time() - start:0.3f}s')
print(f'Number of sampled points: {data.num_nodes} ({data.num_nodes / 10**6:0.2f}M, {100 * data.num_nodes / n_in:0.1f}%)')

#-----------------------------------------------------------------------

from superpoint_transformer.transforms import search_outliers, search_neighbors

radius = 1
# radius = 10
k_min = 5
k_feat = 30
k_adjacency = 10

data = data.cuda()

torch.cuda.synchronize()
start = time()
data, data_outliers = search_outliers(data, k_min, r_max=radius, recursive=True)
data_outliers = data_outliers.cpu()
torch.cuda.synchronize()
print(f'Outliers search: {time() - start:0.3f}s')

torch.cuda.synchronize()
start = time()
data = search_neighbors(data, k_feat, r_max=radius)
# Make sure all points have k neighbors (no "-1" missing neighbors)
assert (data.neighbors != -1).all(), "Some points have incomplete neighborhoods, make sure to remove the outliers to avoid this issue."
torch.cuda.synchronize()
print(f'Neighbor search: {time() - start:0.3f}s')

#-----------------------------------------------------------------------

# from superpoint_transformer.transforms import compute_pointfeatures

# data = data.cpu()
# torch.cuda.synchronize()

# start = time()
# data = compute_pointfeatures(data, pos=True, radius=5, rgb=True, linearity=True, planarity=True, scattering=True, verticality=True, normal=False, length=False, surface=False, volume=False)
# print(f'Geometric features: {time() - start:0.3f}s')

data.x = torch.cat((data.pos / 5, data.rgb), dim=1)
data = data.cpu()

#-----------------------------------------------------------------------

from superpoint_transformer.transforms import compute_ajacency_graph

k_adjacency = 10
lambda_edge_weight = 1


start = time()

data = compute_ajacency_graph(data, k_adjacency, lambda_edge_weight)

print(f'Adjacency graph: {time() - start:0.3f}s')

#-----------------------------------------------------------------------

from superpoint_transformer.transforms import compute_partition

torch.cuda.synchronize()
start = time()

# Parallel cut-pursuit
# data, data_c = compute_partition(data, 0.5, cutoff=10, verbose=True, iterations=10)
nag = compute_partition(data, 0.5, cutoff=10, verbose=True, iterations=5)

torch.cuda.synchronize()
print(f'Partition num_nodes={data.num_nodes}, num_edges={data.num_edges}: {time() - start:0.3f}s')

#-----------------------------------------------------------------------

from superpoint_transformer.transforms import compute_cluster_graph

nag = compute_cluster_graph(nag, n_max_node=32, n_max_edge=64, n_min=5)


#-----------------------------------------------------------------------

from superpoint_transformer.transforms import sample_clusters

# Sample points among the clusters. These will be used to compute
# cluster geometric features as well as cluster adjacency graph and
# edge features
idx_samples, ptr_samples = sample_clusters(
    data_c, n_max=32, low=5, return_pointers=True)

# Compute cluster geometric features
xyz = data.pos[idx_samples].cpu().numpy()
nn = np.arange(idx_samples.shape[0]).astype(
    'uint32')  # !!!! IMPORTANT CAREFUL WITH UINT32 = 4 BILLION points MAXIMUM !!!!
nn_ptr = ptr_samples.cpu().numpy().astype(
    'uint32')  # !!!! IMPORTANT CAREFUL WITH UINT32 = 4 BILLION points MAXIMUM !!!!

# -***************************************
xyz = xyz + torch.rand(xyz.shape).numpy() * 1e-5

torch.cuda.synchronize()

print()
print(f'xyz={xyz.shape}')
print(f'nn={nn.shape}')
print(f'nn_ptr={nn_ptr.shape}, max={np.max(nn_ptr)}, nn_ptr[-1]={nn_ptr[-1]}, nn_ptr[0]={nn_ptr[0]}')
print(f'torch ptr_samples={ptr_samples.shape}, max={ptr_samples.max()}, ptr_samples[-1]={ptr_samples[-1]}, ptr_samples[0]={ptr_samples[0]}')
print()

# C++ geometric features computation on CPU
f = point_utils.compute_geometric_features(xyz, nn, nn_ptr, False)
f = torch.from_numpy(f.astype('float32'))

print('done')

#-----------------------------------------------------------------------