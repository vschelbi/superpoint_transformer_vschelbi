import torch
import numpy as np
import itertools
from scipy.spatial import Delaunay
from torch_scatter import scatter_mean, scatter_std, scatter_min
from torch_geometric.nn.pool.consecutive import consecutive_cluster
import superpoint_transformer
from superpoint_transformer.data import Data, NAG
import superpoint_transformer.partition.utils.libpoint_utils as point_utils
from superpoint_transformer.transforms.sampling import sample_clusters
from superpoint_transformer.utils import print_tensor_info, isolated_nodes


def compute_adjacency_graph(data, k_adjacency, lambda_edge_weight):
    """Create the adjacency graph edges based on the 'Data.neighbors'
    and 'Data.distances'.

    NB: this graph is directed wrt Pytorch Geometric, but cut-pursuit
    happily takes this as an input.
    """
    assert data.has_neighbors, \
        "Level-0 must have 'neighbors' attribute to allow superpoint " \
        "features construction."
    assert getattr(data, 'distances', None) is not None, \
        "Level-0 must have 'distances' attribute to allow superpoint " \
        "features construction."

    # Compute source and target indices based on neighbors
    source = torch.arange(
        data.num_nodes, device=data.device).repeat_interleave(k_adjacency)
    target = data.neighbors[:, :k_adjacency].flatten()

    # Recover the neighbors distances
    distances = data.distances[:, :k_adjacency].flatten()

    # Account for -1 neighbors and delete corresponding edges
    mask = target >= 0
    source = source[mask]
    target = target[mask]
    distances = distances[mask]

    # Save edges and edge features in data
    data.edge_index = torch.stack((source, target))
    data.edge_attr = 1 / (lambda_edge_weight + distances / distances.mean())

    return data


def edge_to_superedge(edges, super_index, edge_attr=None):
    """Convert point-level edges into superedges between clusters, based
    on point-to-cluster indexing 'super_index'. Optionally 'edge_attr'
    can be passed to describe edge attributes that will be returned
    filtered and ordered to describe the superedges.
    """
    # We are only interested in the edges connecting two different
    # clusters and not in the intra-cluster connections. So we first
    # identify the edges of interest. This step requires having access
    # to the 'super_index' to convert point indices into their
    # corresponding cluster indices
    idx_source = super_index[edges[0]]
    idx_target = super_index[edges[1]]
    inter_cluster = torch.where(idx_source != idx_target)[0]

    # Now only consider the edges of interest (ie inter-cluster edges)
    edges_inter = edges[:, inter_cluster]
    edge_attr = edge_attr[inter_cluster] if edge_attr is not None else None
    idx_source = idx_source[inter_cluster]
    idx_target = idx_target[inter_cluster]

    # So far we are manipulating inter-cluster edges, but there may be
    # multiple of those for a given source-target pair. Next, we want to
    # aggregate those into 'superedge" and compute corresponding
    # features (designated with 'se_'). Here, we create unique and
    # consecutive inter-cluster edge identifiers for torch_scatter
    # operations. We use 'se' to designate 'superedge' (ie an edge
    # between two clusters)
    se_id = \
        idx_source * (max(idx_source.max(), idx_target.max()) + 1) + idx_target
    se_id, perm = consecutive_cluster(se_id)
    se_id_source = idx_source[perm]
    se_id_target = idx_target[perm]
    se = torch.vstack((se_id_source, se_id_target))

    return se, se_id, edges_inter, edge_attr


def _compute_cluster_graph(
        i_level, nag, n_max_node=32, n_max_edge=64, n_min=5, max_dist=-1):
    assert isinstance(nag, NAG)
    assert i_level > 0, "Cannot compute cluster graph on level 0"
    assert nag[0].has_edges, \
        "Level-0 must have an adjacency structure in 'edge_index' to allow " \
        "guided sampling for superedges construction."
    assert nag[0].num_nodes < np.iinfo(np.uint32).max, \
        "Too many nodes for `uint32` indices"
    assert nag[0].num_edges < np.iinfo(np.uint32).max, \
        "Too many edges for `uint32` indices"

    # Recover the i_level Data object we will be working on
    data = nag[i_level]
    num_nodes = data.num_nodes

    # Compute how many level-0 points each i_level cluster contains
    sub_size = nag.get_sub_size(i_level, low=0)

    # Sample points among the clusters. These will be used to compute
    # cluster geometric features as well as cluster adjacency graph and
    # edge features
    idx_samples, ptr_samples = sample_clusters(
        i_level, nag, low=0, n_max=n_max_node, n_min=n_min,
        return_pointers=True)

    # Compute cluster geometric features
    xyz = nag[0].pos[idx_samples].cpu().numpy()
    nn = np.arange(idx_samples.shape[0]).astype('uint32')
    nn_ptr = ptr_samples.cpu().numpy().astype('uint32')

    # Heuristic to avoid issues when a cluster sampling is such that
    # it produces singular covariance matrix (eg the sampling only
    # contains the same point repeated multiple times)
    xyz = xyz + torch.rand(xyz.shape).numpy() * 1e-5

    # C++ geometric features computation on CPU
    f = point_utils.compute_geometric_features(xyz, nn, nn_ptr, 5, False)
    f = torch.from_numpy(f.astype('float32'))

    # Recover length, surface and volume
    data.linearity = f[:, 0].to(data.pos.device)
    data.planarity = f[:, 1].to(data.pos.device)
    data.scattering = f[:, 2].to(data.pos.device)
    data.verticality = f[:, 3].to(data.pos.device)
    data.log_length = torch.log(f[:, 7] + 1).to(data.pos.device)
    data.log_surface = torch.log(f[:, 8] + 1).to(data.pos.device)
    data.log_volume = torch.log(f[:, 9] + 1).to(data.pos.device)
    data.normal = f[:, 4:7].view(-1, 3).to(data.pos.device)
    data.log_size = (torch.log(sub_size + 1) - np.log(2)) / 10

    # As a way to "stabilize" the normals' orientation, we choose to
    # express them as oriented in the z+ half-space
    data.normal[data.normal[:, 2] < 0] *= -1

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

    # To debug sampling
    if superpoint_transformer.is_debug_enabled():
        data.super_super_index = super_index
        data.node_idx_samples = idx_samples
        data.node_xyz_samples = torch.from_numpy(xyz)
        data.node_nn_samples = torch.from_numpy(nn.astype('int64'))
        data.node_nn_ptr_samples = torch.from_numpy(nn_ptr.astype('int64'))

        end = ptr_samples[1:]
        start = ptr_samples[:-1]
        super_index_samples = torch.repeat_interleave(
            torch.arange(num_nodes), end - start)
        print('\n\n' + '*' * 50)
        print(f'        cluster graph for i_level={i_level}')
        print('*' * 50 + '\n')
        print(f'nag: {nag}')
        print(f'data: {data}')
        print('\n* Sampling for superpoint features')
        print_tensor_info(idx_samples, 'idx_samples')
        print_tensor_info(ptr_samples, 'ptr_samples')
        print(f'all clusters have a ptr:                   '
              f'{ptr_samples.shape[0] - 1 == num_nodes}')
        print(f'all clusters received n_min+ samples:      '
              f'{(end - start).ge(n_min).all()}')
        print(f'clusters which received no sample:         '
              f'{torch.where(end == start)[0].shape[0]}/{num_nodes}')
        print(f'all points belong to the correct clusters: '
              f'{torch.equal(super_index[idx_samples], super_index_samples)}')

    # Exit in case the i_level graph contains only one node
    if num_nodes < 2:
        data.edge_index = None
        data.edge_attr = None
        nag._list[i_level] = data
        return nag

    # Once we know the i_level cluster each level-0 point belongs to,
    # we can search for level-0 edges between i_level clusters. These
    # in turn tell us which level-0 points to sample from
    edges_point_adj = super_index[nag[0].edge_index]
    inter_cluster = torch.where(edges_point_adj[0] != edges_point_adj[1])[0]
    edges_point_adj_inter = edges_point_adj[:, inter_cluster]
    idx_edge_point = nag[0].edge_index[:, inter_cluster].unique()

    # Some nodes may be isolated and not be connected to the other nodes
    # in the level-0 adjacency graph. For that reason, we need to look
    # for such isolated nodes and sample point inside them, since the
    # above approach will otherwise ignore them
    # TODO: This approach has 2 downsizes: it samples points anywhere in
    #  the isolated cluster and does not drive the other clusters
    #  sampling around the isolated clusters. This means that edges may
    #  not be the true visibility-based, simplest edges of the isolated
    #  clusters. Could be solved by dense sampling inside the clusters
    #  with points near the isolated clusters, but requires a bit of
    #  effort...
    is_isolated = isolated_nodes(edges_point_adj_inter, num_nodes=num_nodes)
    is_isolated_point = is_isolated[super_index]

    # Combine the point indices into a point mask
    mask = is_isolated_point
    mask[idx_edge_point] = True

    # Sample points among the clusters. These will be used to compute
    # cluster adjacency graph and edge features. Note we sample more
    # generously here than for cluster features, because we need to
    # capture fine-grained adjacency
    idx_samples, ptr_samples = sample_clusters(
        i_level, nag, low=0, n_max=n_max_edge, n_min=n_min, mask=mask,
        return_pointers=True)

    # To debug sampling
    if superpoint_transformer.is_debug_enabled():
        data.edge_idx_samples = idx_samples

        end = ptr_samples[1:]
        start = ptr_samples[:-1]
        super_index_samples = torch.repeat_interleave(
            torch.arange(num_nodes), end - start)

        print('\n* Sampling for superedge features')
        print_tensor_info(idx_samples, 'idx_samples')
        print_tensor_info(ptr_samples, 'ptr_samples')
        print(f'all clusters have a ptr:                   '
              f'{ptr_samples.shape[0] - 1 == num_nodes}')
        print(f'all clusters received n_min+ samples:      '
              f'{(end - start).ge(n_min).all()}')
        print(f'clusters which received no sample:         '
              f'{torch.where(end == start)[0].shape[0]}/{num_nodes}')
        print(f'all points belong to the correct clusters: '
              f'{torch.equal(super_index[idx_samples], super_index_samples)}')

    # Delaunay triangulation on the sampled points. The tetrahedra edges
    # are voronoi graph edges. This is the bottleneck of this function,
    # may be worth investigating alternatives if speedups are needed
    pos = nag[0].pos[idx_samples]
    tri = Delaunay(pos.numpy())

    # Concatenate all edges of the triangulation
    pairs = torch.LongTensor(list(itertools.combinations(range(4), 2)))
    edges_point = torch.from_numpy(np.hstack([
        np.vstack((tri.simplices[:, i], tri.simplices[:, j]))
        for i, j in pairs])).long()
    edges_point = idx_samples[edges_point]

    # Remove duplicate edges. For now, (i,j) and (j,i) are considered
    # to be duplicates. We remove do not care about directed graphs at
    # this point to mitigate memory use. The symmetric edges and edge
    # features will be created at the very end. To remove duplicates,
    # we leverage the Data.clean_graph() method
    edges_point = Data(edge_index=edges_point).clean_graph().edge_index

    # Now we are only interested in the edges connecting two different
    # clusters and not in the intra-cluster connections. Select only
    # inter-cluster edges and compute the corresponding source and
    # target point and cluster indices
    se, se_id, edges_point, _ = edge_to_superedge(edges_point, super_index)

    # Remove edges whose distance is too large. We pay articular
    # attention here to avoid isolating nodes by distance filtering. If
    # a node would be isolated by max_dist filtering, we preserve its
    # shortest edge to avoid it, even if it is larger than max_dist
    if max_dist > 0:
        # Identify the edges that are too long
        dist = torch.linalg.norm(
            nag[0].pos[edges_point[1]] - nag[0].pos[edges_point[0]], dim=1)
        too_far = dist > max_dist

        # Recover the corresponding cluster indices for each edge
        edges_super = super_index[edges_point]

        # Identify the clusters which would be isolated if all edges
        # beyond max_dist were removed
        potential_isolated = isolated_nodes(
            edges_super[:, ~too_far], num_nodes=num_nodes)

        # For those clusters, we will tolerate 1 edge larger than
        # max_dist and that connects to another cluster
        source_isolated = potential_isolated[edges_super[0]]
        target_isolated = potential_isolated[edges_super[1]]
        tricky_edge = too_far & (source_isolated | target_isolated) & \
                      (edges_super[0] != edges_super[1])

        # Sort tricky edges by distance in descending order and sort the
        # edge indices and cluster indices consequently. By populating a
        # 'shortest edge index' tensor for the clusters using the sorted
        # edge indices, we can ensure the last edge is the shortest.
        order = dist[tricky_edge].sort(descending=True).indices
        idx = edges_super[:, tricky_edge][:, order]
        val = torch.where(tricky_edge)[0][order]
        cluster_shortest_edge = -torch.ones(
            num_nodes, dtype=torch.long, device=nag.device)
        cluster_shortest_edge[idx[0]] = val
        cluster_shortest_edge[idx[1]] = val
        idx_edge_to_keep = cluster_shortest_edge[potential_isolated]

        # Update the too-far mask so as to preserve at least one edge
        # for each cluster
        too_far[idx_edge_to_keep] = False
        edges_point = edges_point[:, ~too_far]

        # Since this filtering might have affected edges_point, we
        # recompute the super edges indices and ids
        se, se_id, edges_point, _ = edge_to_superedge(edges_point, super_index)

        del dist

    # Direction are the pointwise source->target vectors, based on which
    # we will compute superedge descriptors
    direction = nag[0].pos[edges_point[1]] - nag[0].pos[edges_point[0]]

    # To stabilize the distance-based features' distribution, we use the
    # sqrt of the metric distance. This assumes coordinates are in meter
    # and that we are mostly interested in the range [1, 100]. Might
    # want to change this if your dataset is different
    dist_sqrt = torch.linalg.norm(direction, dim=1).sqrt() / 10

    # We can now use torch_scatter operations to compute superedge
    # features
    se_direction = scatter_mean(direction.cuda(), se_id.cuda(), dim=0).to(data.pos.device)
    se_direction = se_direction / torch.linalg.norm(se_direction, dim=0)
    se_dist = scatter_mean(dist_sqrt.cuda(), se_id.cuda(), dim=0).to(data.pos.device)
    se_min_dist = scatter_min(dist_sqrt.cuda(), se_id.cuda(), dim=0)[0].to(data.pos.device)
    se_std_dist = scatter_std(dist_sqrt.cuda(), se_id.cuda(), dim=0).to(data.pos.device)

    se_centroid_direction = data.pos[se[1]] - data.pos[se[0]]
    se_centroid_dist = torch.linalg.norm(se_centroid_direction, dim=1).sqrt() / 10

    se_normal_source = data.normal[se[0]]
    se_normal_target = data.normal[se[1]]
    se_normal_angle = (se_normal_source * se_normal_target).sum(dim=1).abs()
    se_angle_source = (se_direction * se_normal_source).sum(dim=1).abs()
    se_angle_target = (se_direction * se_normal_target).sum(dim=1).abs()

    se_log_length_ratio = data.log_length[se[0]] - data.log_length[se[1]]
    se_log_surface_ratio = data.log_surface[se[0]] - data.log_surface[se[1]]
    se_log_volume_ratio = data.log_volume[se[0]] - data.log_volume[se[1]]
    se_log_size_ratio = data.log_size[se[0]] - data.log_size[se[1]]

    se_is_artificial = is_isolated[se[0]] | is_isolated[se[1]]

    # TODO: other superedge ideas to better describe how 2 clusters
    #  relate and the geometry of their border (S=source, T=target):
    #  - matrix that transforms unary_vector_source into
    #   unary_vector_target ? Should express, differences in pose, size
    #   and shape as it requires rotation, translation and scaling. Poses
    #   questions regarding the canonical base, PCA orientation ±π
    #  - current SE direction is not the axis/plane of the edge but
    #   rather its normal... build it with PCA and for points sampled in
    #    each side ? careful with single-point edges...
    #  - avg distance S/T points in border to centroid S/T (how far
    #    is the border from the cluster center)
    #  - angle of mean S->T direction wrt S/T principal components (is
    #    the border along the long of short side of objects ?)
    #  - PCA of points in S/T cloud (is it linear border or surfacic
    #    border ?)
    #  - mean dist of S->T along S/T normal (offset along the objects
    #    normals, eg offsets between steps)

    # TODO: if scatter operations are bottleneck, use scatter_csr

    # The superedges we have created so far are oriented. We need to
    # create the edges and corresponding features for the Target->Source
    # direction now
    se = torch.cat((se, se.flip(0)), dim=1)
    se_feat = torch.vstack([
        torch.cat((se_dist, se_dist)),
        torch.cat((se_min_dist, se_min_dist)),
        torch.cat((se_std_dist, se_std_dist)),
        torch.cat((se_centroid_dist, se_centroid_dist)),
        torch.cat((se_normal_angle, se_normal_angle)),
        torch.cat((se_angle_source, se_angle_target)),
        torch.cat((se_angle_target, se_angle_source)),
        torch.cat((se_log_length_ratio, -se_log_length_ratio)),
        torch.cat((se_log_surface_ratio, -se_log_surface_ratio)),
        torch.cat((se_log_volume_ratio, -se_log_volume_ratio)),
        torch.cat((se_log_size_ratio, -se_log_size_ratio)),
        torch.cat((se_is_artificial, se_is_artificial))]).T

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
