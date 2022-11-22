import torch
import numpy as np
import itertools
from scipy.spatial import Delaunay
from torch_scatter import scatter_mean, scatter_std, scatter_min
import src
from src.data import Data, NAG
from src.transforms import Transform
import src.partition.utils.libpoint_utils as point_utils
from src.utils import print_tensor_info, isolated_nodes, \
    edge_to_superedge, tensor_idx


__all__ = [
    'AdjacencyGraph', 'SegmentFeatures', 'EdgeFeatures', 'ConnectIsolated',
    'NodeSize', 'JitterEdgeFeatures', 'SelectEdgeAttr', 'NAGSelectEdgeAttr']


class AdjacencyGraph(Transform):
    """Create the adjacency graph in `edge_index` and `edge_attr` based
    on the `Data.neighbor_index` and `Data.neighbor_distance`.

    NB: this graph is directed wrt Pytorch Geometric, but cut-pursuit
    happily takes this as an input.

    :param k: int
        Number of neighbors to consider for the adjacency graph
    :param w: float
        Scalar used to modulate the edge weight. If `w <= 0`, all edges
        will have a weight of 1. Otherwise, edges weights will follow:
        ```1 / (w + neighbor_distance / neighbor_distance.mean())```
    """

    def __init__(self, k=10, w=-1):
        self.k = k
        self.w = w

    def _process(self, data):
        assert data.has_neighbors, \
            "Data must have 'neighbor_index' attribute to allow adjacency " \
            "graph construction."
        assert self.w <= 0 or getattr(data, 'neighbor_distance', None) is not None, \
            "Data must have 'neighbor_distance' attribute to allow adjacency graph " \
            "construction."
        assert self.k <= data.neighbor_index.shape[1]

        # Compute source and target indices based on neighbors
        source = torch.arange(
            data.num_nodes, device=data.device).repeat_interleave(self.k)
        target = data.neighbor_index[:, :self.k].flatten()

        # Account for -1 neighbors and delete corresponding edges
        mask = target >= 0
        source = source[mask]
        target = target[mask]

        # Save edges and edge features in data
        data.edge_index = torch.stack((source, target))
        if self.w > 0:
            # Recover the neighbor distances and apply the masking
            distances = data.neighbor_distance[:, :self.k].flatten()[mask]
            data.edge_attr = 1 / (self.w + distances / distances.mean())
        else:
            data.edge_attr = torch.ones_like(source, dtype=torch.float)

        return data


class SegmentFeatures(Transform):
    """Compute segment features for all the NAG levels except its first
    (ie the 0-level). These are handcrafted node features that will be
    saved in the node attributes. To make use of those at training time,
    remember to move them to the `x` attribute using `AddKeyToX` and
    `NAGAddKeyToX`.

    :param n_max_node: int
        Maximum number of level-0 points to sample in each cluster to
        when building node features
    :param n_min: int
        Minimum number of level-0 points to sample in each cluster,
        unless it contains fewer points
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, n_max_node=32, n_min=5):
        self.n_max_node = n_max_node
        self.n_min = n_min

    def _process(self, nag):
        for i_level in range(1, nag.num_levels):
            nag = _compute_cluster_features(
                i_level, nag, n_max_node=self.n_max_node, n_min=self.n_min)
        return nag


def _compute_cluster_features(
        i_level, nag, n_max_node=32, n_min=5):
    assert isinstance(nag, NAG)
    assert i_level > 0, "Cannot compute cluster features on level-0"
    assert nag[0].num_nodes < np.iinfo(np.uint32).max, \
        "Too many nodes for `uint32` indices"

    # Recover the i_level Data object we will be working on
    data = nag[i_level]
    num_nodes = data.num_nodes
    device = nag.device

    # Compute how many level-0 points each i_level cluster contains
    sub_size = nag.get_sub_size(i_level, low=0)

    # Sample points among the clusters. These will be used to compute
    # cluster geometric features
    idx_samples, ptr_samples = nag.get_sampling(
        high=i_level, low=0, n_max=n_max_node, n_min=n_min,
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
    data.linearity = f[:, 0].to(device)
    data.planarity = f[:, 1].to(device)
    data.scattering = f[:, 2].to(device)
    data.verticality = f[:, 3].to(device)
    data.curvature = f[:, 10].to(device)
    data.log_length = torch.log(f[:, 7] + 1).to(device)
    data.log_surface = torch.log(f[:, 8] + 1).to(device)
    data.log_volume = torch.log(f[:, 9] + 1).to(device)
    data.normal = f[:, 4:7].view(-1, 3).to(device)
    data.log_size = (torch.log(sub_size + 1) - np.log(2)) / 10

    # As a way to "stabilize" the normals' orientation, we choose to
    # express them as oriented in the z+ half-space
    data.normal[data.normal[:, 2] < 0] *= -1

    # TODO: augment with Rep-SURF umbrella features ?
    # TODO: Random PointNet + PCA features ?

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
    if src.is_debug_enabled():
        data.super_super_index = super_index.to(device)
        data.node_idx_samples = idx_samples.to(device)
        data.node_xyz_samples = torch.from_numpy(xyz).to(device)
        data.node_nn_samples = torch.from_numpy(nn.astype('int64')).to(device)
        data.node_nn_ptr_samples = torch.from_numpy(nn_ptr.astype('int64')).to(device)

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

    # Update the i_level Data in the NAG
    nag._list[i_level] = data

    return nag


class EdgeFeatures(Transform):
    """Compute horizontal edges for all NAG levels except its first
    (ie the 0-level). These are the edges connecting the segments at
    each level, equipped with handcrafted edge features.

    By default, a series of handcrafted edge attributes are computed and
    stored in the corresponding `Data.edge_attr`. However, if one only
    needs a subset of those at train time, one may make use of
    `SelectEdgeAttr` and `NAGSelectEdgeAttr`.

    :param n_max_edge: int
        Maximum number of level-0 points to sample in each cluster to
        when building edges and edge features from Delaunay
        triangulation and edge features
    :param n_min: int
        Minimum number of level-0 points to sample in each cluster,
        unless it contains fewer points
    :param max_dist: float or List(float)
        Maximum distance allowed for edges. If zero, this is ignored.
        Otherwise, edges whose distance is larger than max_dist. We pay
        particular attention here to avoid isolating nodes by distance
        filtering. If a node was isolated by max_dist filtering, we
        preserve its shortest edge to avoid it, even if it is larger
        than max_dist
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, n_max_edge=64, n_min=5, max_dist=-1):
        self.n_max_edge = n_max_edge
        self.n_min = n_min
        self.max_dist = max_dist

    def _process(self, nag):
        assert isinstance(self.max_dist, (int, float, list)), \
            "Expected a scalar or a List"

        max_dist = self.max_dist
        if not isinstance(max_dist, list):
            max_dist = [max_dist] * (nag.num_levels - 1)

        for i_level, md in zip(range(1, nag.num_levels), max_dist):
            nag = _compute_edge_features(
                i_level, nag, n_max_edge=self.n_max_edge, n_min=self.n_min,
                max_dist=md)

        return nag


def _compute_edge_features(
        i_level, nag, n_max_edge=64, n_min=5, max_dist=-1):
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
    device = nag.device

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
    # TODO: This approach has 2 downsides: it samples points anywhere in
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
    mask = torch.where(mask)[0]

    # Sample points among the clusters. These will be used to compute
    # cluster adjacency graph and edge features. Note we sample more
    # generously here than for cluster features, because we need to
    # capture fine-grained adjacency
    idx_samples, ptr_samples = nag.get_sampling(
        high=i_level, low=0, n_max=n_max_edge, n_min=n_min, mask=mask,
        return_pointers=True)

    # To debug sampling
    if src.is_debug_enabled():
        data.edge_idx_samples = idx_samples

        end = ptr_samples[1:]
        start = ptr_samples[:-1]
        super_index_samples = torch.arange(
            num_nodes, device=device).repeat_interleave(end - start)

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
    tri = Delaunay(pos.cpu().numpy())

    # TODO: alternative to Delaunay: search all clusters with at least 1
    #  point within R of 1 point of the cluster:
    #  - dumb: radius search on segment centroids
    #  - better: search intersection between self bbox+R and all bboxes
    #    (can this be vectorized ?) -> https://math.stackexchange.com/questions/2651710/simplest-way-to-determine-if-two-3d-boxes-intersect

    # Concatenate all edges of the triangulation
    pairs = torch.tensor(
        list(itertools.combinations(range(4), 2)), device=device,
        dtype=torch.long)
    edges_point = torch.from_numpy(np.hstack([
        np.vstack((tri.simplices[:, i], tri.simplices[:, j]))
        for i, j in pairs])).long().to(device)
    edges_point = idx_samples[edges_point]

    # Remove duplicate edges. For now, (i,j) and (j,i) are considered
    # to be duplicates. We remove duplicate point-wise graph edges at
    # this point to mitigate memory use. The symmetric edges and edge
    # features will be created at the very end. To remove duplicates,
    # we leverage the Data.clean_graph() method
    edges_point = Data(edge_index=edges_point).clean_graph().edge_index

    # Now we are only interested in the edges connecting two different
    # clusters and not in the intra-cluster connections. Select only
    # inter-cluster edges and compute the corresponding source and
    # target point and cluster indices.
    se, se_id, edges_point, _ = edge_to_superedge(edges_point, super_index)

    # Remove edges whose distance is too large. We pay articular
    # attention here to avoid isolating nodes by distance filtering. If
    # a node was isolated by max_dist filtering, we preserve its
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
        tricky_edge = too_far & (source_isolated | target_isolated) \
                      & (edges_super[0] != edges_super[1])

        # Sort tricky edges by distance in descending order and sort the
        # edge indices and cluster indices consequently. By populating a
        # 'shortest edge index' tensor for the clusters using the sorted
        # edge indices, we can ensure the last edge is the shortest.
        order = dist[tricky_edge].sort(descending=True).indices
        idx = edges_super[:, tricky_edge][:, order]
        val = torch.where(tricky_edge)[0][order]
        cluster_shortest_edge = -torch.ones(
            num_nodes, dtype=torch.long, device=device)
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
    se_direction = scatter_mean(direction.cuda(), se_id.cuda(), dim=0).to(device)
    se_direction = se_direction / torch.linalg.norm(se_direction, dim=0)
    se_dist = scatter_mean(dist_sqrt.cuda(), se_id.cuda(), dim=0).to(device)
    se_min_dist = scatter_min(dist_sqrt.cuda(), se_id.cuda(), dim=0)[0].to(device)
    se_std_dist = scatter_std(dist_sqrt.cuda(), se_id.cuda(), dim=0).to(device)

    se_centroid_direction = data.pos[se[1]] - data.pos[se[0]]
    se_centroid_dist = torch.linalg.norm(se_centroid_direction, dim=1).sqrt() / 10

    se_normal_source = data.normal[se[0]]
    se_normal_target = data.normal[se[1]]
    se_normal_angle = (se_normal_source * se_normal_target).sum(dim=1).abs()

    # These do not seem useful: all edges are ~0. Would be more useful
    # with the 1st PCA component of the edge points...
    # se_angle_source = (se_direction * se_normal_source).sum(dim=1).abs()
    # se_angle_target = (se_direction * se_normal_target).sum(dim=1).abs()

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
        # torch.cat((se_angle_source, se_angle_target)),
        # torch.cat((se_angle_target, se_angle_source)),
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


class ConnectIsolated(Transform):
    """Creates edges for isolated nodes. Each isolated node is connected
    to the `k` nearest nodes. If the Data graph contains edge features
    in `Data.edge_attr`, the new edges will receive features based on
    their length and a linear regression of the relation between
    existing edge features and their corresponding edge length.

    NB: this is an inplace operation that will modify the input data.

    :param k: int
        Number of neighbors the isolated nodes should be connected to
    """

    def __init__(self, k=1):
        self.k = k

    def _process(self, data):
        return data.connect_isolated(k=self.k)


class NodeSize(Transform):
    """Compute the number of `low`-level elements are contained in each
    segment, at each above-level. Results are save in the `node_size`
    attribute of the corresponding Data objects.

    Note: `low=-1` is accepted when level-0 has a `sub` attribute
    (ie level-0 points are themselves segments of `-1` level absent
    from the NAG object).

    :param low: int
        Level whose elements we want to count
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, low=0):
        assert isinstance(low, int) and low >= -1
        self.low = low

    def _process(self, nag):
        for i_level in range(self.low + 1, nag.num_levels):
            nag[i_level].node_size = nag.get_sub_size(i_level, low=self.low)
        return nag


class JitterEdgeFeatures(Transform):
    """Add some gaussian noise to data.edge_attr for all data in a NAG.

    :param sigma: float or List(float)
        Standard deviation of the gaussian noise. A list may be passed
        to transform NAG levels with different parameters. Passing
        sigma <= 0 will prevent any jittering.
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, sigma=0.01):
        assert isinstance(sigma, (int, float, list))
        self.sigma = float(sigma)

    def _process(self, nag):
        device = nag.device

        if not isinstance(self.sigma, list):
            sigma = [self.sigma] * nag.num_levels
        else:
            sigma = self.sigma

        for i_level in range(nag.num_levels):

            if sigma[i_level] <= 0 \
                    or getattr(nag[i_level], 'edge_attr', None) is None:
                continue

            noise = torch.randn_like(
                nag[i_level].edge_attr, device=device) * self.sigma
            nag[i_level].edge_attr += noise

        return nag


class SelectEdgeAttr(Transform):
    """Select edge attributes from their indices.

    :param idx: int, Tensor or list
        The indices of the edge features to keep. If None, this
        transform will have no effect and edge features will be left
        untouched
    """

    def __init__(self, idx=None):
        self.idx = tensor_idx(idx) if idx is not None else None

    def _process(self, data):
        if self.idx is None:
            return data
        data.edge_attr = data.edge_attr[:, self.idx.to(device=data.device)]
        return data


class NAGSelectEdgeAttr(Transform):
    """Select edge attributes from their indices.

    :param level: int or str
        Level at which to select attributes. Can be an int or a str. If
        the latter, 'all' will apply on all levels, 'i+' will apply on
        level-i and above, 'i-' will apply on level-i and below
    :param idx: int, Tensor or list
        The indices of the edge features to keep. If None, this
        transform will have no effect and edge features will be left
        untouched
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, level='all', idx=None):
        self.level = level
        self.idx = idx

    def _process(self, nag):

        level_idx = [[]] * nag.num_levels
        if isinstance(self.level, int):
            level_idx[self.level] = self.idx
        elif self.level == 'all':
            level_idx = [self.idx] * nag.num_levels
        elif self.level[-1] == '+':
            i = int(self.level[:-1])
            level_idx[i:] = [self.idx] * (nag.num_levels - i)
        elif self.level[-1] == '-':
            i = int(self.level[:-1])
            level_idx[:i] = [self.idx] * i
        else:
            raise ValueError(f'Unsupported level={self.level}')

        transforms = [SelectEdgeAttr(idx=idx) for idx in level_idx]

        for i_level in range(nag.num_levels):
            nag._list[i_level] = transforms[i_level](nag._list[i_level])

        return nag
