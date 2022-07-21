"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
    Script for partioning into simples shapes
"""
import os.path
import sys
import numpy as np
import argparse
from timeit import default_timer as timer
sys.path.append("./partition/cut-pursuit/build/src")
sys.path.append("./partition/point_utils")
sys.path.append("./partition")
import libcp
import superpoint_transformer.partition.utils.libpoint_utils as point_utils
from graphs import *
from provider import *

            if args.voxel_width > 0:
                xyz, rgb, labels, dump = point_utils.prune(xyz.astype('f4'), args.voxel_width, rgb.astype('uint8'), labels.astype('uint8'), np.zeros(1, dtype='uint8'), n_labels, 0)
                    #an example of pruning without labels
                xyz, rgb, labels = point_utils.prune(xyz, args.voxel_width, rgb, np.array(1,dtype='u1'), 0)
                #another one without rgb information nor labels
                xyz = point_utils.prune(xyz, args.voxel_width, np.zeros(xyz.shape,dtype='u1'), np.array(1,dtype='u1'), 0)[0]
            #if no labels available simply set here labels = []
            #if no rgb available simply set here rgb = [] and make sure to not use it later on

            graph_nn, target_fea = compute_graph_nn_2(xyz, args.k_nn_adj, args.k_nn_geof)
            #---compute geometric features-------
            geof = point_utils.compute_geof(xyz, target_fea, args.k_nn_geof).astype('float32')
            end = timer()
            times[0] = times[0] + end - start
            del target_fea
            write_features(fea_file, geof, xyz, rgb, graph_nn, labels)
        #--compute the partition------
        sys.stdout.flush()
        if os.path.isfile(spg_file) and not args.overwrite:
            print("    reading the existing superpoint graph file...")
            graph_sp, components, in_component = read_spg(spg_file)
        else:
            print("    computing the superpoint graph...")
            #--- build the spg h5 file --
            start = timer()
            if args.dataset=='s3dis':
                features = np.hstack((geof, rgb/255.)).astype('float32')#add rgb as a feature for partitioning
                features[:,3] = 2. * features[:,3] #increase importance of verticality (heuristic)
            elif args.dataset=='sema3d':
                 features = geof
                 geof[:,3] = 2. * geof[:, 3]
            elif args.dataset=='custom_dataset':
                #choose here which features to use for the partition
                 features = geof
                 geof[:,3] = 2. * geof[:, 3]
                
            graph_nn["edge_weight"] = np.array(1. / ( args.lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])), dtype = 'float32')
            print("        minimal partition...")
            components, in_component = libcp.cutpursuit(features, graph_nn["source"], graph_nn["target"]
                                         , graph_nn["edge_weight"], args.reg_strength)
            components = np.array(components, dtype = 'object')
            end = timer()
            times[1] = times[1] + end - start
            print("        computation of the SPG...")
            start = timer()
            graph_sp = compute_sp_graph(xyz, args.d_se_max, in_component, components, labels, n_labels)
            end = timer()
            times[2] = times[2] + end - start
            write_spg(spg_file, graph_sp, components, in_component)
        
        print("Timer : %5.1f / %5.1f / %5.1f " % (times[0], times[1], times[2]))
