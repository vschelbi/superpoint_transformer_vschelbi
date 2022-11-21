import torch
from torch_geometric.nn.pool.consecutive import consecutive_cluster


__all__ = ['is_pyg_edge_format', 'isolated_nodes', 'edge_to_superedge']


def is_pyg_edge_format(edge_index):
    """Check whether edge_index follows pytorch geometric graph edge
    format: a [2, N] torch.LongTensor.
    """
    return \
        isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2 \
        and edge_index.dtype == torch.long and edge_index.shape[0] == 2


def isolated_nodes(edge_index, num_nodes=None):
    """Return a boolean mask of size num_nodes indicating which node has
    no edge in edge_index.
    """
    assert is_pyg_edge_format(edge_index)
    num_nodes = edge_index.max() + 1 if num_nodes is None else num_nodes
    device = edge_index.device
    mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
    mask[edge_index.unique()] = False
    return mask


def edge_to_superedge(edges, super_index, edge_attr=None):
    """Convert point-level edges into superedges between clusters, based
    on point-to-cluster indexing 'super_index'. Optionally 'edge_attr'
    can be passed to describe edge attributes that will be returned
    filtered and ordered to describe the superedges.

    NB: this function treats (i, j) and (j, i) superedges as identical.
    By default, the final edges are expressed with i <= j
    """
    # We are only interested in the edges connecting two different
    # clusters and not in the intra-cluster connections. So we first
    # identify the edges of interest. This step requires having access
    # to the 'super_index' to convert point indices into their
    # corresponding cluster indices
    se = super_index[edges]
    inter_cluster = torch.where(se[0] != se[1])[0]

    # Now only consider the edges of interest (ie inter-cluster edges)
    edges_inter = edges[:, inter_cluster]
    edge_attr = edge_attr[inter_cluster] if edge_attr is not None else None
    se = se[:, inter_cluster]

    # Search for undirected edges, ie edges with (i,j) and (j,i)
    # both present in edge_index. Flip (j,i) into (i,j) to make them
    # redundant. By default, the final edges are expressed with i <= j
    s_larger_t = se[0] > se[1]
    se[:, s_larger_t] = se[:, s_larger_t].flip(0)

    # So far we are manipulating inter-cluster edges, but there may be
    # multiple of those for a given source-target pair. If, we want to
    # aggregate those into 'superedges' and compute corresponding
    # features (designated with 'se_'), we will need unique and
    # compact inter-cluster edge identifiers for torch_scatter
    # operations. We use 'se' to designate 'superedge' (ie an edge
    # between two clusters)
    se_id = \
        se[0] * (max(se[0].max(), se[1].max()) + 1) + se[1]
    se_id, perm = consecutive_cluster(se_id)
    se = se[:, perm]

    return se, se_id, edges_inter, edge_attr
