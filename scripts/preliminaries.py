import sys, os
import socket

HOST = socket.gethostname()
if HOST == 'DEL2001W017':
    DATA_ROOT = '/media/drobert-admin/DATA2/datasets'
elif HOST == 'HP-2010S002':
    DATA_ROOT = '/var/data/drobert/datasets'
elif HOST == '9c81b1a54ad8':
    DATA_ROOT = '/raid/dataset/pointcloud/data'
else:
    raise NotImplementedError(f"Unknown host '{HOST}', cannot set DATA_ROOT")

file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # for .py script
# file_path = os.path.dirname(os.path.abspath(''))  # for .ipynb notebook
sys.path.append(file_path)

from time import time
import glob
import torch
from superpoint_transformer.transforms import *
from superpoint_transformer.data import Data
from superpoint_transformer.metrics import ConfusionMatrix
from superpoint_transformer.datasets.kitti360 import read_kitti360_window
from superpoint_transformer.datasets.kitti360_config import WINDOWS, KITTI360_NUM_CLASSES, CLASS_NAMES
from superpoint_transformer.utils.cpu import available_cpu_count

print(f'CPUs available: {available_cpu_count()}')

# Hyperparameters and experiment info
infos = Data(
    i_window=0,
    voxel=0.1,
    radius=10,
    k_min=1,
    k_feat=50,
    k_adjacency=10,
    lambda_edge_weight=-1,
    rgb=True,
    hsv=False,
    lab=False,
    density=False,
    linearity=True,
    planarity=True,
    scattering=True,
    verticality=True,
    normal=False,
    length=False,
    surface=False,
    volume=False,
    curvature=True,
    regularization=0.04,
    # regularization=[0.04, 0.2, 0.8, 1.4]
    spatial_weight=1e-2, ******* #for first level partition, prevent too large sp road instead of sidewalk, for higher levels, don't care
    cutoff=10,
    # cutoff=[10, 100, 1000, 10000],
    iterations=15,
    times={})

# Data loading
key = 'loading'
print(f'{key:<12} : ', end='')
start = time()
# all_filepaths = sorted(glob.glob(os.path.join(
#     DATA_ROOT, 'kitti360/shared/data_3d_semantics/*/static/*.ply')))
root = os.path.join(DATA_ROOT, 'kitti360/shared/data_3d_semantics')
train_filepaths = [
    os.path.join(root, f"{x.split('/')[0]}/static/{x.split('/')[1]}.ply")
    for x in WINDOWS['train']]
filepath = train_filepaths[infos.i_window]
data = read_kitti360_window(filepath, semantic=True, instance=False, remap=True)
#TODO Offset labels by 1 to account for unlabelled points -> !!!!!!!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!!!!
data.y[data.y == -1] = KITTI360_NUM_CLASSES
infos.filepath = filepath
infos.num_nodes_raw = data.num_nodes
infos.times[key] = round(time() - start, 3)
print(f'{infos.times[key]} s')

# Voxelization
key = 'voxelization'
print(f'{key:<12} : ', end='')
torch.cuda.synchronize()
start = time()
n_in = data.num_nodes
data = GridSampling3D(
    size=infos.voxel, bins={'y': KITTI360_NUM_CLASSES + 1})(data.cuda()).cpu()
torch.cuda.synchronize()
infos.num_nodes_voxel = data.num_nodes
infos.times[key] = round(time() - start, 3)
print(f'{infos.times[key]} s')

# Neighbors
key = 'neighbors'
print(f'{key:<12} : ', end='')
torch.cuda.synchronize()
start = time()
data = data.cuda()
data = search_neighbors(data, infos.k_feat, r_max=infos.radius)
torch.cuda.synchronize()
infos.times[key] = round(time() - start, 3)
print(f'{infos.times[key]} s')

# Point features
key = 'features'
print(f'{key:<12} : ', end='')
data = data.cpu()
torch.cuda.synchronize()
start = time()
data = compute_point_features(
    data, rgb=infos.rgb, linearity=infos.linearity, planarity=infos.planarity,
    scattering=infos.scattering, verticality=infos.verticality,
    normal=infos.normal, length=infos.length, surface=infos.surface,
    volume=infos.volume, curvature=infos.curvature, k_min=infos.k_min)
infos.times[key] = round(time() - start, 3)
print(f'{infos.times[key]} s')

# Adjacency
key = 'adjacency'
print(f'{key:<12} : ', end='')
start = time()
data = data.cuda()
data = compute_adjacency_graph(data, infos.k_adjacency, infos.lambda_edge_weight)
infos.num_edges = data.num_edges
infos.times[key] = round(time() - start, 3)
print(f'{infos.times[key]} s')

# Connect isolated points
key = 'isolated'
print(f'{key:<12} : ', end='')
start = time()
data = data.cuda()
data = data.connect_isolated(k=infos.k_min)
infos.num_edges = data.num_edges
infos.times[key] = round(time() - start, 3)
print(f'{infos.times[key]} s')

# Partition
key = 'partition'
print(f'{key:<12} : ', end='')
torch.cuda.synchronize()
start = time()
data = data.cpu()
nag = compute_partition(
    data, infos.regularization, spatial_weight=infos.spatial_weight,
    k_adjacency=infos.k_adjacency, cutoff=infos.cutoff, verbose=True,
    iterations=infos.iterations)
torch.cuda.synchronize()
infos.num_super = nag[1].num_nodes
infos.times[key] = round(time() - start, 3)
print(f'{infos.times[key]} s')

# Compute total time
infos.total_time = round(sum(v for v in infos.times.values()), 3)
print(f'*** Total = {infos.total_time} ***\n')

# Compute total time
infos.total_time = round(sum(v for v in infos.times.values()), 3)
print(f'*** Total = {infos.total_time} ***\n')

# Compute the corresponding pointwise prediction and ground truth
# NB: make sure not to count 'ignored' points
# !!! we do not properly count the outliers here (CAREFUL FOR LATER) !!!
infos.cm = []
for i, d in enumerate(nag):
    print(f'\nLevel-{i} purity')
    cm = ConfusionMatrix.from_histogram(
        d.y[:, :KITTI360_NUM_CLASSES], class_names=CLASS_NAMES, verbose=True)
    infos.cm.append(cm)

# Print the infos
print('\nInfos')
print(infos)
