import sys
import os.path as osp
import torch
import numpy as np
from superpoint_transformer.data import Data, Cluster, NAG
from superpoint_transformer.transforms.graph import compute_ajacency_graph, \
    edge_to_superedge
from torch_scatter import scatter_sum, scatter_mean

partition_folder = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(partition_folder)
sys.path.append(osp.join(partition_folder, "partition/grid_graph/python/bin"))
sys.path.append(osp.join(partition_folder, "partition/parallel_cut_pursuit/python/wrappers"))

from grid_graph import edge_list_to_forward_star
from cp_kmpp_d0_dist import cp_kmpp_d0_dist


def compute_partition(
        data, reg_strength, k_adjacency=10, lambda_edge_weight=1, cutoff=1,
        parallel=True, balance=True, iterations=10, verbose=False):
    """Partition the graph with parallel cut-pursuit."""
    # Sanity checks
    assert isinstance(data, Data)
    assert 'x' in data.keys, "Expected node features in `data.x`"
    assert data.has_neighbors, "Expected node neighbors in `data.neighbors`"
    assert 'distances' in data.keys, "Expected node distances in `data.distances`"
    assert data.num_nodes < np.iinfo(np.uint32).max, "Too many nodes for `uint32` indices"
    assert data.num_edges < np.iinfo(np.uint32).max, "Too many edges for `uint32` indices"
    assert isinstance(reg_strength, (int, list)), "Expected an int or a List"

    # Number of threads depending on the parallelization
    max_thread = 0 if parallel else 1

    # The number of levels of hierarchy
    reg_list = [reg_strength] if isinstance(reg_strength, int) else reg_strength

    # Prepare the output as list of Data objects that will be stored in
    # a NAG structure
    data_list = [data]

    # Recover level-0pointwise features from Data object
    x = np.asfortranarray(data.x.numpy().T)

    # Compute a level-0 adjacency graph based on 'Data.neighbors' and
    # 'Data.distances'. NB: here we choose to delete the adjacency graph
    # from the Data attributes right after, since it is redundant with
    # 'neighbors' and 'distances' attributes and we might not need it
    # again outside of this scope
    data = compute_ajacency_graph(data, k_adjacency, lambda_edge_weight)
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    num_nodes = data.num_nodes
    del data.edge_index, data.edge_attr

    # Iteratively run the partition on the previous level of partition
    for i_level, reg in enumerate(reg_list):

        # Convert to forward-star graph representation
        # TODO: IMPORTANT CAREFUL WITH UINT32 = 4 BILLION points MAXIMUM
        first_edge, adj_vertices, reindex = edge_list_to_forward_star(
            num_nodes, edge_index.T.contiguous().numpy())
        first_edge = first_edge.astype('uint32')
        adj_vertices = adj_vertices.astype('uint32')

        # Update edge_weights based on the regularization strength and
        # forward-star reindexing
        edge_weights = reg * edge_attr.numpy()[reindex]

        # We choose to only enforce a cutoff for the level-0 partition
        cutoff = cutoff if i_level == 0 else 1

        # Partition computation
        super_index, x_c, cluster, times = cp_kmpp_d0_dist(
            1, x, first_edge, adj_vertices, edge_weights=edge_weights,
            min_comp_weight=cutoff, cp_dif_tol=1e-2, cp_it_max=iterations,
            split_damp_ratio=0.7, verbose=verbose, max_num_threads=max_thread,
            compute_Time=True, compute_List=True,
            balance_parallel_split=balance)

        if verbose:
            delta_t = times[1:] - times[:-1]
            print(f'Level {i_level} iteration times: {delta_t:0.2f}')

        # Save the super_index for the i-level
        super_index = torch.from_numpy(super_index.astype('int64'))
        data_list[-1].super_index = super_index

        # Save cluster information in another Data object. Convert
        # cluster-to-point indices in a CSR format
        sizes = torch.LongTensor([c.shape[0] for c in cluster])
        pointers = torch.cat([torch.LongTensor([0]), sizes.cumsum(dim=0)])
        values = torch.cat([torch.from_numpy(x.astype('int64')) for x in cluster])
        data_sup = Data(x=torch.from_numpy(x_c.T), sub=Cluster(pointers, values))

        # Prepare the features and adjacency graph for the i+1-level
        # partition
        x = x_c
        num_nodes = data_sup.num_nodes
        edge_index, idx_se, _, edge_attr = edge_to_superedge(
            edge_index, super_index, edge_attr=edge_attr)
        edge_attr = scatter_sum(edge_attr.cuda(), idx_se.cuda(), dim=0).cpu()
        # TODO: scatter operations on CUDA ?

        # Aggregate some point attributes into the clusters. This is not
        # performed dynamically since not all attributes can be
        # aggregated (eg 'neighbors', 'distances', 'edge_index',
        # 'edge_attr'...)
        data_sub = data_list[-1]

        # TODO: scatter operations on CUDA ?
        if 'pos' in data_sub.keys:
            data_sup.pos = scatter_mean(
                data_sub.pos.cuda(), data_sub.super_index.cuda(), dim=0).cpu()
            torch.cuda.empty_cache()

        if 'rgb' in data_sub.keys:
            data_sup.rgb = scatter_mean(
                data_sub.rgb.cuda(), data_sub.super_index.cuda(), dim=0).cpu()
            torch.cuda.empty_cache()

        if 'y' in data_sub.keys:
            assert data_sub.y.dim() == 2, \
                "Expected Data.y to hold `(num_nodes, num_classes)` " \
                "histograms, not single labels"
            data_sup.y = scatter_sum(
                data_sub.y.cuda(), data_sub.super_index.cuda(), dim=0).cpu()
            torch.cuda.empty_cache()

        if 'pred' in data_sub.keys:
            assert data_sub.pred.dim() == 2, \
                "Expected Data.pred to hold `(num_nodes, num_classes)` " \
                "histograms, not single labels"
            data_sup.pred = scatter_sum(
                data_sub.pred.cuda(), data_sub.super_index.cuda(), dim=0).cpu()
            torch.cuda.empty_cache()

        # TODO: aggregate other attributes ?

        # Add the level-i+1 Data object to data_list
        data_list.append(data_sup)

    # Create the NAG object
    nag = NAG(data_list)

    return nag
