import torch
import numpy as np
import itertools
from scipy.spatial import Delaunay
from torch_scatter import scatter_mean, scatter_std, scatter_min
from superpoint_transformer.data import NAG
import superpoint_transformer.partition.utils.libpoint_utils as point_utils
from superpoint_transformer.transforms.sampling import sample_clusters
from torch_geometric.nn.pool.consecutive import consecutive_cluster


def compute_ajacency_graph(data, k_adjacency, lambda_edge_weight):
    """Create the adjacency graph edges based on the 'Data.neighbors'
    and 'Data.distances'.

    NB: this graph is directed wrt Pytorch Geometric, but cut-pursuit
    happily takes this as an input.
    """
    source = torch.arange(
        data.num_nodes, device=data.device).repeat_interleave(k_adjacency)
    target = data.neighbors[:, :k_adjacency].flatten()
    data.edge_index = torch.stack((source, target))

    # Create the edge features for the adjacency graph
    distances = data.distances[:, :k_adjacency].flatten()
    data.edge_attr = 1 / (lambda_edge_weight + distances / distances.mean())

    return data


def edge_to_superedge(edges, super_index, edge_attr=None):
    """Convert point-level edges into superedges between clusters, based
    on point-to-cluster indexing 'super_index'. Optionally 'edge_attr'
    can be passed to describe edge attributes that will be returned
    filtered and ordered to describe the superedges.
    """
    print('\n............  edge_to_superedge  ............')
    print(f'    edges: shape={edges.shape}, min={edges.min()}, max={edges.max()}, unique.shape={edges.unique().shape}')
    print(f'    super_index: shape={super_index.shape}, min={super_index.min()}, max={super_index.max()}, unique.shape={super_index.unique().shape}')

    # We are only interested in the edges connecting two different
    # clusters and not in the intra-cluster connections. So we first
    # identify the edges of interest. This step requires having access
    # to the 'super_index' to convert point indices into their
    # corresponding cluster indices
    idx_source = super_index[edges[0]]
    idx_target = super_index[edges[1]]
    inter_cluster = torch.where(idx_source != idx_target)[0]

    print(f'    idx_source: shape={idx_source.shape}, min={idx_source.min()}, max={idx_source.max()}, unique.shape={idx_source.unique().shape}')
    print(f'    idx_target: shape={idx_target.shape}, min={idx_target.min()}, max={idx_target.max()}, unique.shape={idx_target.unique().shape}')
    if inter_cluster.shape[0] > 0:
        print(f'    inter_cluster: shape={inter_cluster.shape}, min={inter_cluster.min()}, max={inter_cluster.max()}, unique.shape={inter_cluster.unique().shape}')
    else:
        print('    inter_cluster shape is 0')

    # Now only consider the edges of interest (ie inter-cluster edges)
    edges_inter = edges[:, inter_cluster]
    edge_attr = edge_attr[inter_cluster] if edge_attr is not None else None
    idx_source = idx_source[inter_cluster]
    idx_target = idx_target[inter_cluster]

    print(f'    idx_source: shape={idx_source.shape}, min={idx_source.min()}, max={idx_source.max()}, unique.shape={idx_source.unique().shape}')
    print(f'    idx_target: shape={idx_target.shape}, min={idx_target.min()}, max={idx_target.max()}, unique.shape={idx_target.unique().shape}')

    # So far we are manipulating inter-cluster edges, but there may be
    # multiple of those for a given source-target pair. Next, we want to
    # aggregate those into 'superedge" and compute corresponding
    # features (designated with 'se_'). Here, we create unique and
    # consecutive inter-cluster edge identifiers for torch_scatter
    # operations. We use 'se' to designate 'superedge' (ie an edge
    # between two clusters)
    idx_se = idx_source + idx_source.max() * idx_target
    print(f'    idx_se: shape={idx_se.shape}, min={idx_se.min()}, max={idx_se.max()}, unique.shape={idx_se.unique().shape}')
    idx_se, perm = consecutive_cluster(idx_se)
    print(f'    idx_se: shape={idx_se.shape}, min={idx_se.min()}, max={idx_se.max()}, unique.shape={idx_se.unique().shape}')
    print(f'    perm: shape={perm.shape}, min={perm.min()}, max={perm.max()}, unique.shape={perm.unique().shape}')
    idx_se_source = idx_source[perm]
    idx_se_target = idx_target[perm]
    print(f'    idx_se_source: shape={idx_se_source.shape}, min={idx_se_source.min()}, max={idx_se_source.max()}, unique.shape={idx_se_source.unique().shape}')
    print(f'    idx_se_target: shape={idx_se_target.shape}, min={idx_se_target.min()}, max={idx_se_target.max()}, unique.shape={idx_se_target.unique().shape}')
    se = torch.vstack((idx_se_source, idx_se_target))
    print(f'    se: shape={se.shape}, min={se.min()}, max={se.max()}, unique.shape={se.unique().shape}')
    print()

    return se, idx_se, edges_inter, edge_attr


def _compute_cluster_graph(
        i_level, nag, n_max_node=32, n_max_edge=64, n_min=5, max_dist=-1):
    # TODO: WARNING the cluster geometric features will only work if we
    #  enforced a cutoff on the minimum superpoint size ! Make sure you
    #  enforce this

    # TODO: QUESTION: we currently build superedges and sample superedge
    #  points based on the assumption that they connect level-i clusters
    #  iff there is at least on elevel-0 adjacency edge connecting points
    #  belonging to each cluster. But this is not true, some clusters are
    #  just isolated and will never be connected to the rest. For the same
    #  reason, those will never be aggregated in the hierarchical partition,
    #  producing weird annoying artifacts, potentially hurting training too...
    #  How should we deal with those at partition time and at superedge
    #  construction time ?

    # TODO: define recursive sampling for super(n)point features
    # TODO: define recursive edge for super(n)edge features
    # TODO: return all eigenvectors from the C++ geometric features, for
    #  superborder features computation
    # TODO: other superedge ideas to better describe how 2 clusters
    #  relate and the geometry of their border (S=source, T=target):
    #  - avg distance S/T points in border to centroid S/T (how far
    #    is the border from the cluster center)
    #  - angle of mean S->T direction wrt S/T principal components (is
    #    the border along the long of short side of objects ?)
    #  - PCA of points in S/T cloud (is it linear border or surfacic
    #    border ?)
    #  - mean dist of S->T along S/T normal (offset along the objects
    #    normals, eg offsets between steps)

    assert isinstance(nag, NAG)
    assert i_level > 0
    assert nag[0].has_edges, \
        "Level-0 must have an adjacency structure in 'edge_index' to allow " \
        "guided sampling for superedges construction."

    # Recover the i_level Data object we will be working on
    data = nag[i_level]

    print()
    print()
    print('************************************************')
    print(f'        cluster graph for i_level={i_level}')
    print('************************************************')
    print()
    print(f'nag: {nag}')
    print(f'data: {data}')

    # Sample points among the clusters. These will be used to compute
    # cluster geometric features as well as cluster adjacency graph and
    # edge features
    idx_samples, ptr_samples = sample_clusters(
        i_level, nag, low=0, n_max=n_max_node, n_min=n_min, pointers=True)

    print()
    print(f'idx_samples: {idx_samples.shape}')
    print(f'idx_samples.min(): {idx_samples.min()}')
    print(f'idx_samples.max(): {idx_samples.max()}')
    from superpoint_transformer.utils import has_duplicates, is_sorted
    print(f'idx_samples duplicates: {has_duplicates(idx_samples)}')
    print()
    print(f'ptr_samples: {ptr_samples.shape}')
    print(f'ptr_samples.min(): {ptr_samples.min()}')
    print(f'ptr_samples.max(): {ptr_samples.max()}')
    print(f'ptr_samples delta size min: {(ptr_samples[1:] - ptr_samples[:-1]).min()}')
    print(f'ptr_samples delta size max: {(ptr_samples[1:] - ptr_samples[:-1]).max()}')
    print(f'ptr_samples is_sorted: {is_sorted(ptr_samples)}')
    print(f'all clusters have a ptr: {ptr_samples.shape[0] - 1 == nag[i_level].num_nodes}')
    print(f'all clusters received n_min+ samples: {(ptr_samples[1:] - ptr_samples[:-1]).ge(n_min).all()}')
    #TODO: this is temp--------
    super_index = nag[0].super_index
    for i in range(1, i_level):
        super_index = nag[i].super_index[super_index]
    #TODO: this is temp--------
    print(f'some clusters received receive no sample: {torch.where(ptr_samples[1:] == ptr_samples[:-1])[0].shape[0]}/{nag[i_level].num_nodes}')
    # TODO: ********temp
    a = torch.repeat_interleave(torch.arange(nag[i_level].num_nodes), ptr_samples[1:] - ptr_samples[:-1])
    print(f'all points belong to the correct clusters: {torch.equal(super_index[idx_samples], a)}')
    # TODO: ********temp

    #TODO: !!!!! something FISHY here: SAMPLES ONLY BELONG TO ONE / A FEW CLUSTERS !!!!!




    # Compute cluster geometric features
    #TODO: # !!!! IMPORTANT CAREFUL WITH UINT32 = 4 BILLION points MAXIMUM !!!
    xyz = nag[0].pos[idx_samples].cpu().numpy()
    nn = np.arange(idx_samples.shape[0]).astype('uint32')
    nn_ptr = ptr_samples.cpu().numpy().astype('uint32')

    # Heuristic to avoid issues when a cluster sampling is such that
    # it produces singular covariance matrix (eg the sampling only
    # contains the same point repeated multiple times)
    xyz = xyz + torch.rand(xyz.shape).numpy() * 1e-5

    # C++ geometric features computation on CPU
    f = point_utils.compute_geometric_features(xyz, nn, nn_ptr, False)
    f = torch.from_numpy(f.astype('float32'))

    # Recover length, surface and volume
    data.length = f[:, 7].to(data.pos.device)
    data.surface = f[:, 8].to(data.pos.device)
    data.volume = f[:, 9].to(data.pos.device)
    data.normal = f[:, 4:7].view(-1, 3).to(data.pos.device)

    # To guide the sampling for superedges, we want to sample among
    # points whose neighbors in the level-0 adjacency graph belong to
    # a different cluster in the i_level graph. To this end, we first
    # need to tell whose i_level cluster each level-0 point belongs to.
    # This step requires having access to the whole NAG, since we need
    # to convert level-0 point indices into their corresponding level-i
    # superpoint indices
    super_index = nag[0].super_index
    for i in range(1, i_level):
        super_index = nag[i].super_index[super_index]

    # Once we know the i_level cluster each level-0 point belongs to,
    # we can search for level-0 edges between i_level clusters. These
    # in turn tell us which level-0 points to sample from
    edges_0 = super_index[nag[0].edge_index]
    inter_cluster = torch.where(edges_0[0] != edges_0[1])[0]
    idx_edge_point = nag[0].edge_index[:, inter_cluster].unique()

    # Some nodes may be isolated and not be connected to the other nodes
    # in the level-0 adjacency graph. For that reason, we need to look
    # for such isolated nodes and sample point inside them, since the
    # above approach will otherwise ignore them
    if nag[i_level].has_edges:
        is_isolated = nag[i_level].is_isolated()
    else:
        is_isolated = torch.ones(
            nag[i_level].num_nodes, dtype=torch.bool, device=nag.device)
        is_isolated[edges_0.unique()] = False
    idx_isolated_point = is_isolated[super_index]

    # Combine the point indices into a point mask
    mask = idx_isolated_point
    mask[idx_edge_point] = True

    # Sample points among the clusters. These will be used to compute
    # cluster adjacency graph and edge features. Note we sample more
    # generously here than for cluster features, because we need to
    # capture fine-grained adjacency
    idx_samples, ptr_samples = sample_clusters(
        i_level, nag, low=0, n_max=n_max_edge, n_min=n_min,
        mask=mask, pointers=True)

    print()
    print(f'idx_samples: {idx_samples.shape}')
    print(f'idx_samples.min(): {idx_samples.min()}')
    print(f'idx_samples.max(): {idx_samples.max()}')
    from superpoint_transformer.utils import has_duplicates
    print(f'idx_samples duplicates: {has_duplicates(idx_samples)}')
    print()
    print(f'ptr_samples: {ptr_samples.shape}')
    print(f'ptr_samples.min(): {ptr_samples.min()}')
    print(f'ptr_samples.max(): {ptr_samples.max()}')
    print(f'ptr_samples delta size min: {(ptr_samples[1:] - ptr_samples[:-1]).min()}')
    print(f'ptr_samples delta size max: {(ptr_samples[1:] - ptr_samples[:-1]).max()}')
    print(f'ptr_samples is_sorted: {is_sorted(ptr_samples)}')
    print(f'all clusters have a ptr: {ptr_samples.shape[0] - 1 == nag[i_level].num_nodes}')
    print(f'all clusters received n_min+ samples: {(ptr_samples[1:] - ptr_samples[:-1]).ge(n_min).all()}')
    print(f'some clusters received receive no sample: {torch.where(ptr_samples[1:] == ptr_samples[:-1])[0].shape[0]}/{nag[i_level].num_nodes}')
    #TODO: ********temp
    a = torch.repeat_interleave(torch.arange(nag[i_level].num_nodes), ptr_samples[1:] - ptr_samples[:-1])
    print(f'all points belong to the correct clusters: {torch.equal(super_index[idx_samples], a)}')
    # TODO: ********temp

    # Delaunay triangulation on the sampled points. The tetrahedra edges
    # are voronoi graph edges

    pos = nag[0].pos[idx_samples]
    tri = Delaunay(pos.numpy())

    # Concatenate all edges of the triangulation. For now, we do not
    # worry about directed/undirected graphs to mitigate memory and
    # compute
    pairs = torch.LongTensor(list(itertools.combinations(range(4), 2)))
    edges = torch.from_numpy(np.hstack([
        np.vstack((tri.simplices[:, i], tri.simplices[:, j]))
        for i, j in pairs])).long()

    # Remove edges whose distance is too large
    # TODO: careful that max_dist does not re-isolate the de-isolated
    #  nodes
    if max_dist > 0:
        dist = torch.linalg.norm(pos[edges[1]] - pos[edges[0]], dim=1)
        edges = edges[:, dist <= max_dist]
        del dist

    # Now we are only interested in the edges connecting two different
    # clusters and not in the intra-cluster connections. Select only
    # inter-cluster edges and compute the corresponding source and
    # target point and cluster indices
    se, idx_se, edges_inter, _ = edge_to_superedge(
        idx_samples[edges], super_index)

    # Direction are the pointwise source->target vectors, based on which
    # we will compute superedge descriptors
    direction = nag[0].pos[edges_inter[1]] - nag[0].pos[edges_inter[0]]
    dist = torch.linalg.norm(direction, dim=1)

    # We can now use torch_scatter operations to compute superedge
    # features
    se_direction = scatter_mean(direction.cuda(), idx_se.cuda(), dim=0).cpu()
    se_dist = scatter_mean(dist.cuda(), idx_se.cuda(), dim=0).cpu()
    se_min_dist = scatter_min(dist.cuda(), idx_se.cuda(), dim=0)[0].cpu()
    se_std_dist = scatter_std(dist.cuda(), idx_se.cuda(), dim=0).cpu()

    se_centroid_direction = data.pos[se[1]] - data.pos[se[0]]
    se_centroid_dist = torch.linalg.norm(se_centroid_direction, dim=1)

    se_normal_source = data.normal[se[0]]
    se_normal_target = data.normal[se[1]]
    se_normal_angle = (se_normal_source * se_normal_target).sum(dim=1)
    se_angle_source = (se_direction * se_normal_source).sum(dim=1)
    se_angle_target = (se_direction * se_normal_target).sum(dim=1)

    se_length_ratio = data.length[se[0]] / (data.length[se[1]] + 1e-6)
    se_surface_ratio = data.surface[se[0]] / (data.surface[se[1]] + 1e-6)
    se_volume_ratio = data.volume[se[0]] / (data.volume[se[1]] + 1e-6)

    sub_size = nag.get_sub_size(i_level)
    se_size_ratio = sub_size[se[0]] / (sub_size[se[1]] + 1e-6)

    # The superedges we have created so far are oriented. We need to
    # create the edges and corresponding features for the Target->Source
    # direction now
    se = torch.cat((se, se.roll(1, 1)), dim=1)

    #TODO : visualize/control the SP and SE features
    se_feat = [
        torch.cat((se_dist, se_dist)),
        torch.cat((se_min_dist, se_min_dist)),
        torch.cat((se_std_dist, se_std_dist)),
        torch.cat((se_centroid_dist, se_centroid_dist)),
        torch.cat((se_normal_angle, se_normal_angle)),
        torch.cat((se_angle_source, se_angle_target)),
        torch.cat((se_angle_target, se_angle_source)),
        torch.cat((se_length_ratio, 1 / (se_length_ratio + 1e-6))),
        torch.cat((se_surface_ratio, 1 / (se_surface_ratio + 1e-6))),
        torch.cat((se_volume_ratio, 1 / (se_volume_ratio + 1e-6))),
        torch.cat((se_size_ratio, 1 / (se_size_ratio + 1e-6)))]

    # Aggregate all edge features in a single tensor
    se_feat = torch.vstack(se_feat).T

    # Save superedges and superedge features in the Data object
    data.edge_index = se
    data.edge_attr = se_feat

    # Restore the i_level Data object, if need be
    nag._list[i_level] = data

    return nag


def compute_cluster_graph(
        nag, n_max_node=32, n_max_edge=64, n_min=5, max_dist=-1):
    assert isinstance(nag, NAG)
    assert isinstance(max_dist, (int, float, list)), "Expected a scalar or a List"

    if not isinstance(max_dist, list):
        max_dist = [max_dist] * (nag.num_levels - 1)

    for i_level, md in zip(range(1, nag.num_levels), max_dist):
        nag = _compute_cluster_graph(
            i_level, nag, n_max_node=n_max_node, n_max_edge=n_max_edge,
            n_min=n_min, max_dist=md)

    return nag
