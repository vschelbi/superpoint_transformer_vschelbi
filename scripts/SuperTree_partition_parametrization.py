import os
import sys
import itertools
import laspy
import torch
import numpy as np
import time
from datetime import datetime
import json

# Add the project's files to the python path
file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_path)

from src.data import Data, InstanceData
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from src.transforms import *

FOR_Instance_num_classes = 3
ID2TRAINID = np.asarray([
    FOR_Instance_num_classes,   # 0 Unclassified        ->  3 Ignored
    1,                          # 1 Low vegetation      ->  1 Low vegetation
    0,                          # 2 Terrain             ->  0 Ground
    FOR_Instance_num_classes,   # 3 Out-points          ->  3 Ignored
    2,                          # 4 Stem                ->  2 Tree
    2,                          # 5 Live branches       ->  2 Tree
    2,                          # 6 Woody branches      ->  2 Tree
])

FOR_Instance_CLASS_NAMES = [
    'Ground',
    'Low vegetation',
    'Tree',
    'Ignored']

FOR_Instance_CLASS_COLORS = np.asarray([
    [243, 214, 171],    # Ground
    [204, 213, 174],    # Low vegetation
    [ 70, 115,  66],    # Tree
    [  0,   0,   0]     # Ignored
])

filepaths = [
    "/home/valerio/git/superpoint_transformer_vschelbi/data/FORinstance/raw/CULS/plot_3_annotated.las",
    "/home/valerio/git/superpoint_transformer_vschelbi/data/FORinstance/raw/NIBIO/plot_10_annotated.las",
    "/home/valerio/git/superpoint_transformer_vschelbi/data/FORinstance/raw/RMIT/train.las",
    "/home/valerio/git/superpoint_transformer_vschelbi/data/FORinstance/raw/SCION/plot_35_annotated.las",
    "/home/valerio/git/superpoint_transformer_vschelbi/data/FORinstance/raw/TUWIEN/train.las"
]

#############################
# PARTITION PARAMETRIZATION #
#############################
param_values = {
    # Voxelization (GridSampling3D)
    'voxel_size': [0.1, 0.2],    # size of the voxels in the partitions in meters. The voxel size is the same for all the plots

    # KNN
    'k': [20, 25],               # number of nearest neighbors to consider in the KNN search => only 25 or 25 and 40
    'r_max': [3, 5],          # search nearest neighbors within this radius in meters

    # GroundElevation
    'threshold': [5],            # ground as a planar surface located within `threshold` of the lowest point in the cloud.
    'scale': [20],               # Pointwise distance to the plane is normalized by `scale`

    # PointFeatures
    'features': [('intensity', 'linearity', 'planarity', 'scattering', 'verticality', 'elevation')],        ## TODO add all features
                                 # handcrafted geometric features characterizing each point's neighborhood. The following features are currently supported:
                                 #    - density
                                 #    - linearity
                                 #    - planarity
                                 #    - scattering
                                 #    - verticality
                                 #    - normal
                                 #    - length
                                 #    - surface
                                 #    - volume
                                 #    - curvature
                                 #    - elevation
                                 #    - (RGB color)  
                                 #    - (HSV color)  
                                 #    - (LAB color)
                                 #
                                 # can specify a k_min below which a point will receive 0 geometric features to mitigate low-quality features for sparse neighborhoods
                                 # `PointFeatures(k_step=..., k_min_search=...)` will search for the optimal neighborhood size among available neighbors for each point, based on eigenfeatures entropy

    # AdjacencyGraph
    'k_adj_graph': [5, 10],      # use edges of the `k`_adj_graph nearest neighbors
    'w': [1],                    # weight of the edges in the adjacency graph with w

    # AddKeysTo
    # features that we want to use for the partition generation, same features available as in PointFeatures
    'features_to_x': [
        ('linearity', 'planarity', 'scattering', 'verticality', 'elevation'),
        ('intensity', 'linearity', 'planarity', 'scattering', 'verticality', 'elevation'),
        ('density', 'linearity', 'planarity', 'scattering', 'verticality', 'elevation'),
        ('linearity', 'planarity', 'scattering', 'verticality'),
    ],

    # CutPursuitPartition
    'regularization': [[0.05, 0.1], [0.1, 0.2], [0.2, 0.5]],   # List of increasing float values determining the granularity of hierarchical superpoint partitions.
    'spatial_weight': [[0.1, 0.01], [1, 0.1], [0.1, 0.1]],                 # Float value indicating the importance of point coordinates relative to point features in grouping points.
    'cutoff': [[10, 30], [20, 50]],                            # Integer specifying the minimum number of points in each superpoint, ensuring small superpoints are merged with others.
    'iterations': [15],                                        # Integer specifying the number of iterations for the Cut Pursuit algorithm.
    'k_adjacency': [5, 10]                                     # Integer preventing superpoints from being isolated.
}


# Generate all parameter combinations
param_combinations = list(itertools.product(*param_values.values()))

#############
# FUNCTIONS #
#############
def read_FORinstance_plot(filepath, xyz=True, intensity=True, semantic=True, instance=True, remap=True, max_intensity=None):
    """
    Read a FORinstance plot from a LAS file and return the data object.
    """
    data = Data()
    las = laspy.read(filepath)

    if xyz:
        pos = torch.stack([
            torch.as_tensor(np.array(las[axis]))
            for axis in ["X", "Y", "Z"]], dim=-1)
        pos *= las.header.scale
        pos_offset = pos[0]
        data.pos = (pos - pos_offset).float()
        data.pos_offset = pos_offset

    intensity_remaped = True
    if intensity:
        data.intensity = torch.FloatTensor(
            las['intensity'].astype('float32')
        )
        if intensity_remaped:
            if max_intensity is None:
                max_intensity = data.intensity.max()
            data.intensity = data.intensity.clip(min=0, max=max_intensity) / max_intensity

    if semantic:
        y = torch.LongTensor(np.array(las['classification']))
        data.y = torch.from_numpy(ID2TRAINID)[y] if remap else y

    if instance:
        idx = torch.arange(data.num_points)
        obj = torch.LongTensor(np.array(las['treeID']))
        
        y = torch.LongTensor(np.array(las['classification']))
        y = torch.from_numpy(ID2TRAINID)[y] if remap else y

        if remap:
            ground_mask = (obj == 0) & (y == 0)
            low_veg_mask = (obj == 0) & (y == 1)
            if low_veg_mask.any() or ground_mask.any():
                ground_instance_label = obj.max().item() + 1
                low_veg_instance_label = ground_instance_label + 1
                obj[ground_mask] = ground_instance_label
                obj[low_veg_mask] = low_veg_instance_label

        obj = consecutive_cluster(obj)[0]
        count = torch.ones_like(obj)

        data.obj = InstanceData(idx, obj, count, y, dense=True)
    
    return data

def calc_point_density(voxel_size=0.1, data=None):
    """
    Helper function to calculate the point density for the specified voxel size.
    """
    data_voxelized = GridSampling3D(size=voxel_size)(data)
    voxel_ratio = data.num_nodes / data_voxelized.num_nodes
    data_voxelized_1m = GridSampling3D(size=1)(data)
    vol_density = data_voxelized.num_nodes / data_voxelized_1m.num_nodes

    return vol_density, voxel_ratio

def point_density_experiments(data_list, plot_titles, voxel_sizes=[0.1, 0.2, 0.5, 1.0]):
    """
    Calculate the point density for the specified voxel sizes for all plots and print the results.
    """
    for voxel_size in voxel_sizes:
        print("-----------------------------")
        print(f"Voxel size = {voxel_size}")
        for data, plot_title in zip(data_list, plot_titles):
            density, ratio = calc_point_density(voxel_size, data)
            print(f"{plot_title}: density = {density:.2f} points/m^3, voxel ratio = {ratio:.2f}")

def apply_transform(data, transform, *args, **kwargs):
    """
    Helper function to apply a transform to the data and return the transformed data.
    """
    return transform(*args, **kwargs)(data)

def pre_transform_performance(data_list, plot_titles, param_set):
    """
    Pre-transform all the data in the list with the specified parameters and return the oracle performance metrics.

    The function applies a series of transformations to each dataset in the input list and extracts various performance 
    metrics related to semantic segmentation, instance segmentation, and panoptic segmentation for two partition levels (P1 and P2).
    The results are aggregated in a nested dictionary format, where each plot title corresponds to a dictionary containing 
    the metrics for that specific plot. Additionally, a summary of mean metrics across all plots is included.

    Parameters:
    data_list (list): A list of data objects to be transformed and evaluated.
    plot_titles (list): A list of strings representing the titles of each plot, used as keys in the resulting dictionary.
    param_set (tuple): A tuple containing the parameters for the transformations in the following order:
                          (voxel_size, k, r_max, threshold, scale, features, k_adj_graph, w, features_to_x, regularization,
                            spatial_weight, cutoff, iterations, k_adjacency)

    Returns:
    dict: A nested dictionary containing the performance metrics for each plot title and the mean metrics across all plots.
          The structure of the returned dictionary is as follows:
          {
              'plot_title_1': {
                  'level_ratios': {
                      '|P_0| / |P_1|': float,
                      '|P_1| / |P_2|': float
                  },
                  'semantic_segmentation_miou_p1': float,
                  'instance_segmentation_map_p1': float,
                  'panoptic_segmentation_pq_p1': float,
                  'semantic_segmentation_miou_p2': float,
                  'instance_segmentation_map_p2': float,
                  'panoptic_segmentation_pq_p2': float
              },
              'plot_title_2': { ... },
              ...
              'mean_metrics': {
                  'level_ratios': {
                      '|P_0| / |P_1|': float,
                      '|P_1| / |P_2|': float
                  },
                  'semantic_segmentation_miou_p1': float,
                  'instance_segmentation_map_p1': float,
                  'panoptic_segmentation_pq_p1': float,
                  'semantic_segmentation_miou_p2': float,
                  'instance_segmentation_map_p2': float,
                  'panoptic_segmentation_pq_p2': float
              }
          }
    """

    # Extract parameters
    (voxel_size, k, r_max, threshold, scale, features, k_adj_graph, w, features_to_x, 
         regularization, spatial_weight, cutoff, iterations, k_adjacency) = param_set

    performance_metrics = {}

    for i in range(len(data_list)):
        data = data_list[i].clone()
        data = apply_transform(data, GridSampling3D, size=voxel_size, hist_key='y', hist_size=FOR_Instance_num_classes + 1)
        data = apply_transform(data, KNN, k, r_max)
        data = apply_transform(data, GroundElevation, threshold, scale)
        data = apply_transform(data, PointFeatures, keys=features_to_x)
        data = apply_transform(data, AdjacencyGraph, k_adj_graph, w)
        data = apply_transform(data, AddKeysTo, keys=features_to_x, to='x', delete_after=False)
        data = apply_transform(data, CutPursuitPartition, regularization=regularization, \
                                       spatial_weight=spatial_weight, cutoff=cutoff, iterations=iterations, k_adjacency=k_adjacency)

        plot_title = plot_titles[i]
        performance_metrics[plot_title] = {}

        # Extract level ratios
        performance_metrics[plot_title]['level_ratios'] = {
            '|P_0| / |P_1|': data.level_ratios['|P_0| / |P_1|'],
            '|P_1| / |P_2|': data.level_ratios['|P_1| / |P_2|']
        }

        # Extract miou for semantic segmentation (P1)
        semantic_segmentation_p1 = data[1].semantic_segmentation_oracle(FOR_Instance_num_classes)
        performance_metrics[plot_title]['semantic_segmentation_miou_p1'] = semantic_segmentation_p1['miou'].item()

        # Extract map for instance segmentation (P1)
        instance_segmentation_p1 = data[1].instance_segmentation_oracle(FOR_Instance_num_classes)
        performance_metrics[plot_title]['instance_segmentation_map_p1'] = instance_segmentation_p1['map'].item()

        # Extract pq for panoptic segmentation (P1)
        panoptic_segmentation_p1 = data[1].panoptic_segmentation_oracle(FOR_Instance_num_classes)
        performance_metrics[plot_title]['panoptic_segmentation_pq_p1'] = panoptic_segmentation_p1['pq'].item()

        # Extract miou for semantic segmentation (P2)
        semantic_segmentation_p2 = data[2].semantic_segmentation_oracle(FOR_Instance_num_classes)
        performance_metrics[plot_title]['semantic_segmentation_miou_p2'] = semantic_segmentation_p2['miou'].item()

        # Extract map for instance segmentation (P2)
        instance_segmentation_p2 = data[2].instance_segmentation_oracle(FOR_Instance_num_classes)
        performance_metrics[plot_title]['instance_segmentation_map_p2'] = instance_segmentation_p2['map'].item()

        # Extract pq for panoptic segmentation (P2)
        panoptic_segmentation_p2 = data[2].panoptic_segmentation_oracle(FOR_Instance_num_classes)
        performance_metrics[plot_title]['panoptic_segmentation_pq_p2'] = panoptic_segmentation_p2['pq'].item()

        del data
        torch.cuda.empty_cache()

    # Calculate the mean of each metric across all plots
    mean_metrics = {
        'level_ratios': {
            '|P_0| / |P_1|': np.mean([performance_metrics[title]['level_ratios']['|P_0| / |P_1|'] for title in plot_titles]),
            '|P_1| / |P_2|': np.mean([performance_metrics[title]['level_ratios']['|P_1| / |P_2|'] for title in plot_titles])
        },
        'semantic_segmentation_miou_p1': np.mean([performance_metrics[title]['semantic_segmentation_miou_p1'] for title in plot_titles]),
        'instance_segmentation_map_p1': np.mean([performance_metrics[title]['instance_segmentation_map_p1'] for title in plot_titles]),
        'panoptic_segmentation_pq_p1': np.mean([performance_metrics[title]['panoptic_segmentation_pq_p1'] for title in plot_titles]),
        'semantic_segmentation_miou_p2': np.mean([performance_metrics[title]['semantic_segmentation_miou_p2'] for title in plot_titles]),
        'instance_segmentation_map_p2': np.mean([performance_metrics[title]['instance_segmentation_map_p2'] for title in plot_titles]),
        'panoptic_segmentation_pq_p2': np.mean([performance_metrics[title]['panoptic_segmentation_pq_p2'] for title in plot_titles])
    }

    performance_metrics['mean_metrics'] = mean_metrics

    return performance_metrics


##########################
# FILE OPERATION METHODS #
##########################
def initialize_file_with_metadata(filename, metadata):
    # Check if file exists, if not, create it with metadata
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            json.dump(metadata, f)
            f.write('\n')

def append_result(param_set, performance_metrics, filename='logs/partition_parametrization.jsonl'):
    with open(filename, 'a') as f:
        json.dump({'param_set': param_set, 'performance_metrics': performance_metrics}, f)
        f.write('\n')

def save_current_iteration(iteration, filename='logs/current_iteration.txt'):
    with open(filename, 'w') as f:
        f.write(str(iteration))

def load_current_iteration(filename='logs/current_iteration.txt'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return int(f.read().strip())
    return 0


########################
# SUPERPOINT TRANSFORM #
########################
if __name__ == '__main__':
    version = '2.0'
    json_file_name = f'/home/valerio/git/superpoint_transformer_vschelbi/logs/partition_parametrization_{version}.jsonl'
    curr_iteration_file_name = f'/home/valerio/git/superpoint_transformer_vschelbi/logs/current_iteration_{version}.txt'
    metadata = {
        "description": "This file contains results of the superpoint transform experiment. Changed the features to try out, and added one more set for spatial weight.",
        "version": version,
        "author": "Valerio Schelbert",
        "date": datetime.now().strftime("%Y-%m-%d")
    }

    # Initialize file with metadata if it does not exist
    initialize_file_with_metadata(json_file_name, metadata)

    start_iteration = load_current_iteration(curr_iteration_file_name)
    plot_titles = []
    data_list = []

    for filepath in filepaths:
        plot_title = os.path.join(os.path.basename(os.path.dirname(filepath)), os.path.splitext(os.path.basename(filepath))[0])
        plot_titles.append(plot_title)

        data = read_FORinstance_plot(filepath, instance=True)
        data_list.append(data)

    # iterate through all parameter combinations starting from the last saved iteration
    for i, param_set in enumerate(param_combinations[start_iteration:], start=start_iteration):
        print(f"Iteration {i + 1}/{len(param_combinations)}")
        performance_metrics = pre_transform_performance(data_list, plot_titles, param_set)
        # append param_set and performance_metrics to the file
        append_result(param_set, performance_metrics, json_file_name)
        # save current iteration
        save_current_iteration(i + 1, curr_iteration_file_name)

    print("Process completed.")