import numpy as np
import torch
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
import pickle
from tqdm import tqdm
from superpoint_transformer.transforms import *
from superpoint_transformer.data import Data
from superpoint_transformer.metrics import ConfusionMatrix
from superpoint_transformer.datasets.kitti360 import read_kitti360_window
from superpoint_transformer.datasets.kitti360_config import WINDOWS, KITTI360_NUM_CLASSES, CLASS_NAMES

from time import time
import glob
import pickle
from tqdm import tqdm
from superpoint_transformer.transforms import *
from superpoint_transformer.data import Data
from superpoint_transformer.metrics import ConfusionMatrix
from superpoint_transformer.datasets.kitti360 import read_kitti360_window
from superpoint_transformer.datasets.kitti360_config import WINDOWS, KITTI360_NUM_CLASSES, CLASS_NAMES
from superpoint_transformer.utils.io import dated_dir


def process(i_cloud, args):
    info = Data(
        i_cloud=i_cloud,
        times={})

    # Data loading
    # !!! KITTI-360 only !!!
    key = 'loading'
    start = time()
    filepath = args.filepaths[i_cloud]
    data = read_kitti360_window(filepath, semantic=True, instance=False, remap=True)
    data.y[data.y == -1] = KITTI360_NUM_CLASSES
    info.filepath = filepath
    info.num_nodes_raw = data.num_nodes
    info.times[key] = round(time() - start, 3)

    # Voxelization
    key = 'voxelization'
    torch.cuda.synchronize()
    start = time()
    n_in = data.num_nodes
    data = GridSampling3D(
        size=args.voxel, bins={'y': KITTI360_NUM_CLASSES + 1})(data.cuda()).cpu()
    torch.cuda.synchronize()
    info.num_nodes_voxel = data.num_nodes
    info.times[key] = round(time() - start, 3)

    # Neighbors
    key = 'neighbors'
    torch.cuda.synchronize()
    start = time()
    data = data.cuda()
    data = search_neighbors(data, args.k_feat, r_max=args.radius)
    torch.cuda.synchronize()
    info.times[key] = round(time() - start, 3)

    # Point features
    key = 'features'
    data = data.cpu()
    torch.cuda.synchronize()
    start = time()
    data = compute_point_features(
        data, rgb=args.rgb, linearity=args.linearity, planarity=args.planarity,
        scattering=args.scattering, verticality=args.verticality,
        normal=args.normal, length=args.length, surface=args.surface,
        volume=args.volume, k_min=args.k_min)
    info.times[key] = round(time() - start, 3)

    # Adjacency
    key = 'adjacency'
    start = time()
    data = data.cuda()
    data = compute_adjacency_graph(data, args.k_adjacency, args.lambda_edge_weight)
    info.num_edges = data.num_edges
    info.times[key] = round(time() - start, 3)

    # Connect isolated points
    key = 'isolated'
    start = time()
    data = data.cuda()
    data = data.connect_isolated(k=args.k_min)
    info.num_edges = data.num_edges
    info.times[key] = round(time() - start, 3)

    # Partition
    key = 'partition'
    torch.cuda.synchronize()
    start = time()
    data = data.cpu()
    nag = compute_partition(
        data, args.regularization, spatial_weight=args.spatial_weight,
        k_adjacency=args.k_adjacency, cutoff=args.cutoff, verbose=args.verbose,
        iterations=args.iterations, parallel=args.parallel)
    torch.cuda.synchronize()
    info.num_sp = nag[1].num_nodes
    info.times[key] = round(time() - start, 3)

    # Compute total time
    info.total_time = round(sum(v for v in info.times.values()), 3)

    # Iteratively voxelize the data with increasingly coarse voxels,
    # while keeping track of the corresponding number of voxels and the
    # oracle confusion matrix
    d_voxel = read_kitti360_window(
        filepath, semantic=True, instance=False, remap=True)
    d_voxel.y[d_voxel.y == -1] = KITTI360_NUM_CLASSES
    maxi = np.log10(args.voxel_max_size / args.voxel)
    voxel_range = np.logspace(0, maxi, num=args.voxel_steps) * args.voxel
    info.voxel_oracle = {
        'voxel': voxel_range, 'num': [], 'cm': [], 'oa': [], 'miou': []}
    for v in voxel_range:
        d_voxel = GridSampling3D(
            size=v, bins={'y': KITTI360_NUM_CLASSES + 1})(d_voxel.cuda()).cpu()
        cm = ConfusionMatrix.from_histogram(d_voxel.y[:, :KITTI360_NUM_CLASSES])
        info.voxel_oracle['num'].append(d_voxel.num_nodes)
        info.voxel_oracle['cm'].append(cm)
        info.voxel_oracle['oa'].append(cm.get_overall_accuracy())
        info.voxel_oracle['miou'].append(cm.get_average_intersection_union())
    del d_voxel

    # Compute the oracle pointwise prediction for each segmentation level
    # NB: make sure not to count 'ignored' points
    # !!! we do not properly count the outliers here (CAREFUL FOR LATER) !!!
    info.cm = []
    info.segment_oracle = {'num': [], 'oa': [], 'miou': []}
    for i, d in enumerate(nag):
        if args.verbose:
            print(f'\nLevel-{i} purity')
        cm = ConfusionMatrix.from_histogram(
            d.y[:, :KITTI360_NUM_CLASSES], class_names=CLASS_NAMES,
            verbose=args.verbose)
        info.cm.append(cm)
        info.segment_oracle['num'].append(d.num_nodes)
        info.segment_oracle['oa'].append(cm.get_overall_accuracy())
        info.segment_oracle['miou'].append(cm.get_average_intersection_union())

    return info


REG_LIST = [0.04, 0.03, 0.05, 0.06]
root = os.path.join(DATA_ROOT, 'kitti360/shared/data_3d_semantics')
out_dir = dated_dir(
    os.path.join(DATA_ROOT, 'kitti360/spt/preprocessing_study'), create=True)
filepaths = [
    os.path.join(root, f"{x.split('/')[0]}/static/{x.split('/')[1]}.ply")
    for x in WINDOWS['train']]

for REG in REG_LIST:
    print(f'\nRegularization 1: {REG}')

    # Hyperparameters and experiment info
    out_path = os.path.join(out_dir, f'{REG * 100:0.0f}e-2.p')
    args = Data(
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
        # regularization=0.05,
        regularization=[REG, 0.2, 0.8, 1.4],
        # spatial_weight=1e-2,
        spatial_weight=[1e-2, 0, 0, 0],
        # cutoff=10,
        cutoff=[10, 100, 1000, 10000],
        iterations=15,
        filepaths=filepaths,
        num_clouds=len(filepaths),
        voxel_max_size=250,
        voxel_steps=200,
        parallel=True,
        verbose=False,
        failed=[])
    info = []

    # Save voxelization info to disk
    pickle.dump((args, info), open(out_path, "wb"))

    # Compute the info for each cloud in the dataset
    generator = range(args.num_clouds) if args.verbose \
        else tqdm(range(args.num_clouds))
    for i_cloud in generator:
        if args.verbose:
            print(f'\n\n{"_" * 16} Cloud {i_cloud} (num_clouds={args.num_clouds}) {"_" * 16}')
            print(f'path: {filepaths[i_cloud]}')
        args, info = pickle.load(open(out_path, "rb"))
        try:
            info.append(process(i_cloud, args))
        except:
            args.failed.append(i_cloud)
            continue
        pickle.dump((args, info), open(out_path, "wb"))
