import sys
import os.path as osp
import torch
import numpy as np
from superpoint_transformer.data import Data, Cluster, NAG

partition_folder = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(partition_folder)
sys.path.append(osp.join(partition_folder, "partition/grid_graph/python/bin"))
sys.path.append(osp.join(partition_folder, "partition/parallel_cut_pursuit/python/wrappers"))

from grid_graph import edge_list_to_forward_star
from cp_kmpp_d0_dist import cp_kmpp_d0_dist


def compute_partition(
        data, reg_strength, cutoff=1, parallel=True, balance=True,
        iterations=10, verbose=False):
    """Partition the graph with parallel cut-pursuit."""
    # Sanity checks
    assert 'x' in data.keys, "Expected node features in `data.x`"
    assert 'edge_attr' in data.keys, "Expected edge features in `data.edge_attr`"
    assert data.num_nodes < np.iinfo(np.uint32).max, "Too many nodes for `uint32` indices"
    assert data.num_edges < np.iinfo(np.uint32).max, "Too many edges for `uint32` indices"

    # Recover needed tensors from Data object
    x = np.asfortranarray(data.x.numpy().T)
    edge_weights = data.edge_attr.numpy()

    # Convert to forward-star graph representation
    first_edge, adj_vertices, reindex = edge_list_to_forward_star(
        data.num_nodes, data.edge_index.T.contiguous().numpy())
    first_edge = first_edge.astype('uint32')  # IMPORTANT CAREFUL WITH UINT32 = 4 BILLION points MAXIMUM
    adj_vertices = adj_vertices.astype('uint32')  # IMPORTANT CAREFUL WITH UINT32 = 4 BILLION points MAXIMUM

    # Update edge_weights based on the regularization strength and
    # forward-star reindexing
    edge_weights = reg_strength * edge_weights[reindex]

    # Number of threads depending on the parallelization
    max_thread = 0 if parallel else 1

    # Partition computation
    super_index, x_c, cluster, times = cp_kmpp_d0_dist(
        1, x, first_edge, adj_vertices, edge_weights=edge_weights,
        min_comp_weight=cutoff, cp_dif_tol=1e-2, cp_it_max=iterations,
        split_damp_ratio=0.7, verbose=verbose, max_num_threads=max_thread,
        compute_Time=True, compute_List=True, balance_parallel_split=balance)

    if verbose:
        print(f'Iteration times: {(times[1:] - times[:-1]).round(2)}')

    # Save the partition attributes in Data objects
    # TODO: adapt 'super_index' to 'super_index' or 'superpoint_index' to make
    #  use of pgy's built-in mechanisms ? eg in GridSampling or batching ?
    data.super_index = torch.from_numpy(super_index.astype('int64'))

    # Save cluster information in another Data object. Convert
    # cluster-to-point indices in a CSR format
    sizes = torch.LongTensor([c.shape[0] for c in cluster])
    pointers = torch.cat([torch.LongTensor([0]), sizes.cumsum(dim=0)])
    values = torch.cat([torch.from_numpy(x.astype('int64')) for x in cluster])
    data_c = Data(x=torch.from_numpy(x_c.T), sub=Cluster(pointers, values))

    # Create the NAG object
    nag = NAG([data, data_c])

    return nag
