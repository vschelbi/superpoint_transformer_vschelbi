import torch
import numpy as np
import itertools
from scipy.spatial import Delaunay
from torch_scatter import scatter_mean, scatter_std, scatter_min, scatter_sum
from superpoint_transformer.data import NAG
import superpoint_transformer.partition.utils.libpoint_utils as point_utils
from superpoint_transformer.transforms.sampling import sample_clusters
from torch_geometric.nn.pool.consecutive import consecutive_cluster


def compute_ajacency_graph(data, k_adjacency, lambda_edge_weight):
    """Create the adjacency graph edges based on the 'Data.neighbors'.

    NB: this graph is directed wrt Pytorch Geometric, but cut-pursuit
    happily takes this as an input.
    """
    source = torch.arange(data.num_nodes).repeat_interleave(k_adjacency)
    target = data.neighbors[:, :k_adjacency].flatten()
    data.edge_index = torch.stack((source, target))

    # Create the edge features for the adjacency graph
    distances = data.distances[:, :k_adjacency].flatten()
    data.edge_attr = 1 / (lambda_edge_weight + distances / distances.mean())

    return data


def _compute_cluster_graph(
        i_level, nag, high_node=32, high_edge=64, low=5):
    # TODO: WARNING the cluster geometric features will only work if we
    #  enforced a cutoff on the minimum superpoint size ! Make sure you
    #  enforce this

    # TODO: define recursive sampling for super(n)point features
    # TODO: define recursive edge for super(n)edge features
    # TODO: return all eigenvectors from the C++ geometric features, for
    #  superborder features computation
    # TODO: other superedge ideas to better describe how 2 clusters
    # relate and the geometry of their border (S=source, T=target):
    # - avg distance S/T points in border to centroid S/T (how far
    #   is the border from the cluster center)
    # - angle of mean S->T direction wrt S/T principal components (is
    #   the border along the long of short side of objects ?)
    # - PCA of points in S/T cloud (is it linear border or surfacic
    #   border ?)
    # - mean dist of S->T along S/T normal (offset along the objects
    #   normals, eg offsets between steps)

    assert isinstance(nag, NAG)
    assert i_level > 0

    # Recover the i_level Data object we will be working on
    data = nag[i_level]

    # Aggregate some point attributes into the clusters. This is not
    # performed dynamically since not all attributes can be aggregated
    # (eg 'neighbors', 'distances', 'edge_index', 'edge_attr'...)
    data_sub = nag[i_level - 1]

    if 'pos' in data_sub.keys:
        data.pos = scatter_mean(
            data_sub.pos.cuda(), data_sub.super_index.cuda(), dim=0).cpu()
        torch.cuda.empty_cache()

    if 'rgb' in data_sub.keys:
        data.rgb = scatter_mean(
            data_sub.rgb.cuda(), data_sub.super_index.cuda(), dim=0).cpu()
        torch.cuda.empty_cache()

    if 'y' in data_sub.keys:
        assert data_sub.y.dim() == 2, \
            "Expected Data.y to hold `(num_nodes, num_classes)` " \
            "histograms, not single labels"
        data.y = scatter_sum(
            data_sub.y.cuda(), data_sub.super_index.cuda(), dim=0).cpu()
        torch.cuda.empty_cache()

    # Sample points among the clusters. These will be used to compute
    # cluster geometric features as well as cluster adjacency graph and
    # edge features
    idx_samples, ptr_samples = sample_clusters(
        data, high=high_node, low=low, pointers=True)

    # Compute cluster geometric features
    xyz = nag[0].pos[idx_samples].cpu().numpy()
    nn = np.arange(idx_samples.shape[0]).astype('uint32')  # !!!! IMPORTANT CAREFUL WITH UINT32 = 4 BILLION points MAXIMUM !!!!
    nn_ptr = ptr_samples.cpu().numpy().astype('uint32')  # !!!! IMPORTANT CAREFUL WITH UINT32 = 4 BILLION points MAXIMUM !!!!

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

    # Sample points among the clusters. These will be used to compute
    # cluster adjacency graph and edge features. Note we sample more
    # generously here than for cluster features, because we need to
    # capture fine-grained adjacency
    idx_samples, ptr_samples = sample_clusters(
        data, high=high_edge, low=low, pointers=True)

    # Delaunay triangulation on the sampled points
    tri = Delaunay(nag[0].pos[idx_samples].numpy())

    # Concatenate all edges of the triangulation. For now, we do not
    # worry about directed/undirected graphs to mitigate memory and
    # compute
    pairs = torch.LongTensor(list(itertools.combinations(range(4), 2)))
    edges = torch.from_numpy(np.hstack([
        np.vstack((tri.vertices[:, i], tri.vertices[:, j]))
        for i, j in pairs]).T).long()

    # Now we are only interested in the edges connecting two different
    # clusters and not in the intra-cluster connections. So we first
    # identify the edges of interest. This step requires having access
    # to the whole NAG, since we need to convert level-0 point indices
    # into their corresponding level-i superpoint indices
    idx_point_source = idx_samples[edges[:, 0]]
    idx_point_target = idx_samples[edges[:, 1]]
    idx_source = idx_point_source
    idx_target = idx_point_target
    for i in range(i_level):
        idx_source = nag[i].super_index[idx_source]
        idx_target = nag[i].super_index[idx_target]
    inter_cluster = torch.where(idx_source != idx_target)[0]

    # Now only consider the edges of interest (ie inter-cluster edges)
    idx_point_source = idx_point_source[inter_cluster]
    idx_point_target = idx_point_target[inter_cluster]
    idx_source = idx_source[inter_cluster]
    idx_target = idx_target[inter_cluster]

    # Direction are the pointwise source->target vectors, based on which
    # we will compute superedge descriptors. So far we are manipulating
    # inter-cluster edges, but their may be multiple of those for a
    # given source-target pair. Next, we want to aggregate those into
    # "superegdes" and compute corresponding features (designated with
    # 'se_')
    direction = nag[0].pos[idx_point_target] - nag[0].pos[idx_point_source]
    dist = torch.linalg.norm(direction, dim=1)

    # Create unique and consecutive inter-cluster edge identifiers for
    # torch_scatter operations. We use 'se' to designate 'superedge' (ie
    # an edge between two clusters)
    idx_se = idx_source + data.num_nodes * idx_target
    idx_se, perm = consecutive_cluster(idx_se)
    idx_se_source = idx_source[perm]
    idx_se_target = idx_target[perm]
    se = torch.vstack((idx_se_source, idx_se_target))

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
    se_size_ratio = data.sub_size[se[0]] / (data.sub_size[se[1]] + 1e-6)

    # The superedges we have created so far are oriented. We need to
    # create the edges and corresponding features for the Target->Source
    # direction now
    se = torch.cat((se, se.roll(1, 1)))

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


def compute_cluster_graph(nag, high_node=32, high_edge=64, low=5):
    assert isinstance(nag, NAG)
    for i_level in range(1, nag.num_levels):
        nag = _compute_cluster_graph(
            i_level, nag, high_node=high_node, high_edge=high_edge, low=low)
    return nag
