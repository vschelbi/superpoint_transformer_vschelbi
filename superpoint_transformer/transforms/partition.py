import sys
import os.path as osp
import torch
import numpy as np
from superpoint_transformer.data import Data, Cluster, NAG
from superpoint_transformer.transforms.graph import compute_ajacency_graph
from torch_scatter import scatter_sum

partition_folder = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(partition_folder)
sys.path.append(osp.join(partition_folder, "partition/grid_graph/python/bin"))
sys.path.append(osp.join(partition_folder, "partition/parallel_cut_pursuit/python/wrappers"))

from grid_graph import edge_list_to_forward_star
from cp_kmpp_d0_dist import cp_kmpp_d0_dist


def compute_partition(
        data, regularization, k_adjacency=10, lambda_edge_weight=1,
        spatial_weight=1, cutoff=1, parallel=True, iterations=10,
        verbose=False):
    """Partition the graph with parallel cut-pursuit.

    Parameters
    ----------
    data:
    regularization:
    k_adjacency:
    lambda_edge_weight:
    spatial_weight: float
        Weight used to scale the point position features, to mitigate
        the maximum superpoint size.
    cutoff:
    parallel:
    iterations:
    verbose:
    """
    # Sanity checks
    assert isinstance(data, Data)
    assert 'x' in data.keys, "Expected node features in `data.x`"
    assert data.has_neighbors, "Expected node neighbors in `data.neighbors`"
    assert 'distances' in data.keys, "Expected node distances in `data.distances`"
    assert data.num_nodes < np.iinfo(np.uint32).max, "Too many nodes for `uint32` indices"
    assert data.num_edges < np.iinfo(np.uint32).max, "Too many edges for `uint32` indices"
    assert isinstance(regularization, (int, float, list)), "Expected a scalar or a List"
    assert isinstance(cutoff, (int, list)), "Expected an int or a List"
    assert isinstance(spatial_weight, (int, float, list)), "Expected a scalar or a List"

    # Compute a level-0 adjacency graph based on 'Data.neighbors' and
    # 'Data.distances'. NB: here we choose to delete the adjacency graph
    # from the Data attributes right after, since it is redundant with
    # 'neighbors' and 'distances' attributes and we might not need it
    # again outside of this scope
    data = compute_ajacency_graph(data, k_adjacency, lambda_edge_weight)

    # Remove self loops, redundant edges and undirected edges
    #TODO: calling this on the level-0 adjacency graph is a bit sluggish
    # but still saves partition time overall. May be worth finding a
    # quick way of removing self loops and redundant edges...
    data = data.clean_graph()

    # Initialize the hierarchical partition parameters. In particular,
    # prepare the output as list of Data objects that will be stored in
    # a NAG structure
    max_thread = 0 if parallel else 1
    data.node_size = torch.ones(data.num_nodes)  # level-0 points all have the same importance
    data_list = [data]
    if not isinstance(regularization, list):
        regularization = [regularization]
    if isinstance(cutoff, int):
        cutoff = [cutoff] * len(regularization)
    if not isinstance(spatial_weight, list):
        spatial_weight = [spatial_weight] * len(regularization)
    n_dim = data.pos.shape[1]
    n_feat = data.x.shape[1]

    # Iteratively run the partition on the previous level of partition
    for level, (reg, cut, sw) in enumerate(zip(
            regularization, cutoff, spatial_weight)):

        if verbose:
            print(f'Launching partition level={level} reg={reg}, cutoff={cut}')

        # Recover the Data object on which we will run the partition
        d1 = data_list[level]

        # Convert to edges forward-star (or CSR) graph representation
        # TODO: IMPORTANT CAREFUL WITH UINT32 = 4 BILLION points MAXIMUM
        source_csr, target, reindex = edge_list_to_forward_star(
            d1.num_nodes, d1.edge_index.T.contiguous().cpu().numpy())
        source_csr = source_csr.astype('uint32')
        target = target.astype('uint32')
        edge_weights = d1.edge_attr.cpu().numpy()[reindex] * reg

        # Recover attributes features from Data object
        x = np.asfortranarray(
            torch.cat((d1.pos.cpu(), d1.x.cpu()), dim=1).numpy().T)
        node_size = d1.node_size.cpu().numpy()
        coor_weights = np.ones(n_dim + n_feat, dtype=np.float32)
        coor_weights[:n_dim] *= sw

        # Partition computation
        super_index, x_c, cluster, edges, times = cp_kmpp_d0_dist(
            1, x, source_csr, target, edge_weights=edge_weights,
            vert_weights=node_size, coor_weights=coor_weights,
            min_comp_weight=cut, cp_dif_tol=1e-2,
            cp_it_max=iterations, split_damp_ratio=0.7, verbose=verbose,
            max_num_threads=max_thread, compute_Time=True, compute_List=True,
            compute_Graph=True)

        if verbose:
            delta_t = (times[1:] - times[:-1]).round(2)
            print(f'Level {level} iteration times: {delta_t}')
            print(f'partition {level} done')

        # Save the super_index for the i-level
        super_index = torch.from_numpy(super_index.astype('int64'))
        d1.super_index = super_index

        # Save cluster information in another Data object. Convert
        # cluster-to-point indices in a CSR format
        size = torch.LongTensor([c.shape[0] for c in cluster])
        pointer = torch.cat([torch.LongTensor([0]), size.cumsum(dim=0)])
        value = torch.cat([torch.from_numpy(x.astype('int64')) for x in cluster])
        pos = torch.from_numpy(x_c[:n_dim].T)
        x = torch.from_numpy(x_c[n_dim:].T)
        s = torch.arange(edges[0].shape[0] - 1).repeat_interleave(
            torch.from_numpy((edges[0][1:] - edges[0][:-1]).astype("int64")))
        t = torch.from_numpy(edges[1].astype("int64"))
        edge_index = torch.vstack((s, t))
        edge_attr = torch.from_numpy(edges[2] / reg)
        node_size = torch.from_numpy(node_size)
        node_size_new = scatter_sum(node_size.cuda(), super_index.cuda(), dim=0).cpu()
        d2 = Data(
            pos=pos, x=x, edge_index=edge_index, edge_attr=edge_attr,
            sub=Cluster(pointer, value), node_size=node_size_new)

        # Remove self loops, redundant edges and undirected edges
        d2 = d2.clean_graph()

        # If some nodes are isolated in the graph, connect them to their
        # nearest neighbors, so their absence of connectivity does not
        # "pollute" higher levels of partition
        d2 = d2.connect_isolated(k=k_adjacency)

        # Aggregate some point attributes into the clusters. This is not
        # performed dynamically since not all attributes can be
        # aggregated (eg 'neighbors', 'distances', 'edge_index',
        # 'edge_attr'...)
        if 'y' in d1.keys:
            assert d1.y.dim() == 2, \
                "Expected Data.y to hold `(num_nodes, num_classes)` " \
                "histograms, not single labels"
            d2.y = scatter_sum(d1.y.cuda(), d1.super_index.cuda(), dim=0).cpu()
            torch.cuda.empty_cache()

        if 'pred' in d1.keys:
            assert d1.pred.dim() == 2, \
                "Expected Data.pred to hold `(num_nodes, num_classes)` " \
                "histograms, not single labels"
            d2.pred = scatter_sum(
                d1.pred.cuda(), d1.super_index.cuda(), dim=0).cpu()
            torch.cuda.empty_cache()

        # TODO: aggregate other attributes ?

        # Add the l+1-level Data object to data_list and update the
        # l-level after super_index has been changed
        data_list[level] = d1
        data_list.append(d2)

        if verbose:
            print('\n' + '-' * 64 + '\n')

    # Create the NAG object
    nag = NAG(data_list)

    return nag
