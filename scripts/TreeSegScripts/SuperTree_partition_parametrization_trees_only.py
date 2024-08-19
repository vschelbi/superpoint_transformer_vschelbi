import os
import sys
import itertools
import laspy
import torch
import numpy as np
import pandas as pd
import time
from datetime import datetime
import json

# Add the project's files to the python path
file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_path)

from src.data import Data, InstanceData
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from src.transforms import *

FOR_Instance_num_classes = 2

ID2TRAINID = np.asarray([2, 0, 0, 2, 1, 1, 1])

CLASS_NAMES = [
    'Ground and low vegetation',  # 2 Ground, 1 Low vegetation
    'Tree',                       # 4 Stem, 5 Live branches, 6 Woody branches
    'Unknown'                     # 0 Unclassified, 3 Out-points
]

CLASS_COLORS = np.asarray([
    [243, 214, 171],    # Ground and Low vegetation
    [ 70, 115,  66],    # Tree
    [  0,   8, 116]     # Unknown
])

filepaths = [
    "/home/valerio/git/superpoint_transformer_vschelbi/data/forinstance/raw/CULS/plot_3_annotated.las",
    "/home/valerio/git/superpoint_transformer_vschelbi/data/forinstance/raw/NIBIO/plot_10_annotated.las",
    "/home/valerio/git/superpoint_transformer_vschelbi/data/forinstance/raw/RMIT/train.las",
    "/home/valerio/git/superpoint_transformer_vschelbi/data/forinstance/raw/SCION/plot_35_annotated.las",
    "/home/valerio/git/superpoint_transformer_vschelbi/data/forinstance/raw/TUWIEN/train.las"
]

#############################
# PARTITION PARAMETRIZATION #
#############################
param_values = {
    # Voxelization (GridSampling3D)
    'voxel_size': [0.2],    # size of the voxels in the partitions in meters. The voxel size is the same for all the plots

    # KNN
    'k': [20],               # number of nearest neighbors to consider in the KNN search => only 25 or 25 and 40
    'r_max': [3],          # search nearest neighbors within this radius in meters

    # GroundElevation
    'threshold': [5],            # ground as a planar surface located within `threshold` of the lowest point in the cloud.
    'scale': [20],               # Pointwise distance to the plane is normalized by `scale`
                      
    # AdjacencyGraph
    'k_adj_graph': [10],      # use edges of the `k`_adj_graph nearest neighbors
    'w': [1],                    # weight of the edges in the adjacency graph with w

    # AddKeysTo
    # features that we want to use for the partition generation, same features available as in PointFeatures
    'features_to_x': [
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
        ('intensity', 'linearity', 'planarity', 'scattering', 'verticality', 'elevation')
    ],

    # CutPursuitPartition
    'regularization': [[0.2, 0.5]],   # List of increasing float values determining the granularity of hierarchical superpoint partitions.
    'spatial_weight': [[0.1, 0.01]],                 # Float value indicating the importance of point coordinates relative to point features in grouping points.
    'cutoff': [[10, 30]],                            # Integer specifying the minimum number of points in each superpoint, ensuring small superpoints are merged with others.
    'iterations': [15],                                        # Integer specifying the number of iterations for the Cut Pursuit algorithm.
    'k_adjacency': [10]                                     # Integer preventing superpoints from being isolated.
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

def apply_transform(data, transform, *args, **kwargs):
    """
    Helper function to apply a transform to the data and return the transformed data.
    """
    return transform(*args, **kwargs)(data)

def get_result_dict(data, plot_title, param_set_id, param_set):
    '''
    Create a dictionary with the results of the oracle performance metrics for the given data object.
    '''
    result = {
        'param_set_id': param_set_id,
        'param_set': param_set,
        'dataset': plot_title,
        '|P_0| / |P_1|': data.level_ratios['|P_0| / |P_1|'],
        '|P_1| / |P_2|': data.level_ratios['|P_1| / |P_2|']
    }

    # Extract metrics for tree 1
    tree_metrics_1 = data[1].panoptic_segmentation_oracle(FOR_Instance_num_classes)
    result.update({
        'tree iou 1': data[1].semantic_segmentation_oracle(FOR_Instance_num_classes)['iou_per_class'][-1].item(),
        'tree map 1': data[1].instance_segmentation_oracle(FOR_Instance_num_classes)['map_per_class'][-1].item(),
        'tree pq 1': tree_metrics_1['pq_per_class'][-1].item(),
        'tree precision 1': tree_metrics_1['precision_per_class'][-1].item(),
        'tree recall 1': tree_metrics_1['recall_per_class'][-1].item(),
        'tree tp 1': tree_metrics_1['tp_per_class'][-1].item(),
        'tree fp 1': tree_metrics_1['fp_per_class'][-1].item(),
        'tree fn 1': tree_metrics_1['fn_per_class'][-1].item(),
    })

    # Calculating detection, omission, commission, f1 for tree 1
    result.update({
        'detection 1': result['tree tp 1'] / (result['tree tp 1'] + result['tree fn 1']),
        'omission 1': result['tree fn 1'] / (result['tree tp 1'] + result['tree fn 1']),
        'commission 1': result['tree fp 1'] / (result['tree tp 1'] + result['tree fp 1']),
        'f1 1': 2 * result['tree tp 1'] / (2 * result['tree tp 1'] + result['tree fp 1'] + result['tree fn 1'])
    })
    
    # Extract metrics for tree 2
    tree_metrics_2 = data[2].panoptic_segmentation_oracle(FOR_Instance_num_classes)
    result.update({
        'tree iou 2': data[2].semantic_segmentation_oracle(FOR_Instance_num_classes)['iou_per_class'][-1].item(),
        'tree map 2': data[2].instance_segmentation_oracle(FOR_Instance_num_classes)['map_per_class'][-1].item(),
        'tree pq 2': tree_metrics_2['pq_per_class'][-1].item(),
        'tree precision 2': tree_metrics_2['precision_per_class'][-1].item(),
        'tree recall 2': tree_metrics_2['recall_per_class'][-1].item(),
        'tree tp 2': tree_metrics_2['tp_per_class'][-1].item(),
        'tree fp 2': tree_metrics_2['fp_per_class'][-1].item(),
        'tree fn 2': tree_metrics_2['fn_per_class'][-1].item(),
    })

    # Calculating detection, omission, commission, f1 for tree 2
    result.update({
        'detection 2': result['tree tp 2'] / (result['tree tp 2'] + result['tree fn 2']),
        'omission 2': result['tree fn 2'] / (result['tree tp 2'] + result['tree fn 2']),
        'commission 2': result['tree fp 2'] / (result['tree tp 2'] + result['tree fp 2']),
        'f1 2': 2 * result['tree tp 2'] / (2 * result['tree tp 2'] + result['tree fp 2'] + result['tree fn 2'])
    })
    
    return result


def pre_transform_performance(data_list, plot_titles, param_set, param_set_id):
    """
    Pre-transform all the data in the list with the specified parameters and return the oracle performance metrics.

    The function applies a series of transformations to each dataset in the input list and extracts various performance 
    metrics related to semantic segmentation, instance segmentation, and panoptic segmentation for two partition levels (P1 and P2).
    The results are aggregated in a DataFrame.

    Parameters:
    data_list (list): A list of data objects to be transformed and evaluated.
    plot_titles (list): A list of strings representing the titles of each plot, used as keys in the resulting dictionary.
    param_set (tuple): A tuple containing the parameters for the transformations in the following order:
                          (voxel_size, k, r_max, threshold, scale, features, k_adj_graph, w, features_to_x, regularization,
                            spatial_weight, cutoff, iterations, k_adjacency)
    param_set_id (int): Unique identifier for the parameter set.

    Returns:
    pd.DataFrame: A DataFrame containing the performance metrics for each plot title.
    """

    # Extract parameters
    (voxel_size, k, r_max, threshold, scale, k_adj_graph, w, features_to_x, 
         regularization, spatial_weight, cutoff, iterations, k_adjacency) = param_set

    performance_metrics = []

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

        result = get_result_dict(data, plot_titles[i], param_set_id, param_set)
        performance_metrics.append(result)

        # clear memory
        del data
        torch.cuda.empty_cache()

    output = pd.DataFrame(performance_metrics)

    return output


##########################
# FILE OPERATION METHODS #
##########################
def append_result(df, filename='logs/partition_parametrization.csv'):
    if not os.path.exists(filename):
        df.to_csv(filename)
    else:
        df.to_csv(filename, mode='a', header=False)

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
    version = '6.0'
    csv_file_name = f'/home/valerio/git/superpoint_transformer_vschelbi/logs/partition_parametrization_{version}.csv'
    curr_iteration_file_name = f'/home/valerio/git/superpoint_transformer_vschelbi/logs/current_iteration_{version}.txt'
    metadata = {
        "description": "This file contains results of the superpoint transform experiment. Changed the voxel size to 0.5 and removed some other combinations.",
        "version": version,
        "author": "Valerio Schelbert",
        "date": datetime.now().strftime("%Y-%m-%d")
    }

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
        performance_metrics_df = pre_transform_performance(data_list, plot_titles, param_set, i)
        # append param_set and performance_metrics to the file
        append_result(performance_metrics_df, csv_file_name)
        # save current iteration
        save_current_iteration(i + 1, curr_iteration_file_name)

    print("Process completed.")
