import torch
import numpy as np
import itertools
from scipy.spatial import Delaunay
from torch_scatter import scatter_mean, scatter_std, scatter_min, segment_csr
import src
from src.data import NAG
from src.transforms import Transform
import src.partition.utils.libpoint_utils as point_utils
from src.utils import print_tensor_info, isolated_nodes, edge_to_superedge, \
    subedges, to_trimmed, cluster_radius_nn, is_trimmed

__all__ = [
    'AdjacencyGraph', 'SegmentFeatures', 'DelaunayHorizontalGraph',
    'RadiusHorizontalGraph', 'OnTheFlyEdgeFeatures', 'ConnectIsolated',
    'NodeSize', 'JitterEdgeFeatures']


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

    :param n_max: int
        Maximum number of level-0 points to sample in each cluster to
        when building node features
    :param n_min: int
        Minimum number of level-0 points to sample in each cluster,
        unless it contains fewer points
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, n_max=32, n_min=5):
        self.n_max = n_max
        self.n_min = n_min

    def _process(self, nag):
        for i_level in range(1, nag.num_levels):
            nag = _compute_cluster_features(
                i_level, nag, n_max=self.n_max, n_min=self.n_min)
        return nag


def _compute_cluster_features(
        i_level, nag, n_max=32, n_min=5):
    assert isinstance(nag, NAG)
    assert i_level > 0, "Cannot compute cluster features on level-0"
    assert nag[0].num_nodes < np.iinfo(np.uint32).max, \
        "Too many nodes for `uint32` indices"

    # Recover the i_level Data object we will be working on
    data = nag[i_level]
    num_nodes = data.num_nodes
    device = nag.device

    # Compute how many level-0 points each level cluster contains
    sub_size = nag.get_sub_size(i_level, low=0)

    # Sample points among the clusters. These will be used to compute
    # cluster geometric features
    idx_samples, ptr_samples = nag.get_sampling(
        high=i_level, low=0, n_max=n_max, n_min=n_min,
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

    # Add elevation if present in the points, raise an error if not
    # found
    if getattr(nag[0], 'elevation', None) is not None:
        data.elevation = segment_csr(nag[0].elevation, ptr_samples, reduce='mean')

    # TODO: augment with Rep-SURF umbrella features ?
    # TODO: Random PointNet + PCA features ?

    # To guide the sampling for superedges, we want to sample among
    # points whose neighbors in the level-0 adjacency graph belong to
    # a different cluster in the i_level graph. To this end, we first
    # need to tell whose i_level cluster each level-0 point belongs to.
    # This step requires having access to the whole NAG, since we need
    # to convert level-0 point indices into their corresponding level-i
    # superpoint indices
    super_index = nag.get_super_index(i_level)

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
        print(f'        cluster graph for level={i_level}')
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


class DelaunayHorizontalGraph(Transform):
    """Compute horizontal edges for all NAG levels except its first
    (ie the 0-level). These are the edges connecting the segments at
    each level, equipped with handcrafted edge features.

    This approach relies on the dual graph of the Delaunay triangulation
    of the point cloud. To reduce computation, each segment is susampled
    based on its size. This sampling still has downsides and the
    triangulation remains fairly long for large clouds, due to its O(N²)
    complexity. Besides, the horizontal graph induced by the
    triangulation is a visibility-based graph, meaning neighboring
    segments may not be connected if a large enough segment separates
    them. A faster alternative is `RadiusHorizontalGraph`.

    By default, a series of handcrafted edge attributes are computed and
    stored in the corresponding `Data.edge_attr`. However, if one only
    needs a subset of those at train time, one may make use of
    `SelectColumns` and `NAGSelectColumns`.

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
            nag = _horizontal_graph_by_delaunay(
                i_level, nag, n_max_edge=self.n_max_edge, n_min=self.n_min,
                max_dist=md)

        return nag


def _horizontal_graph_by_delaunay(
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

    # Exit in case the i_level graph contains only one node
    if num_nodes < 2:
        data.edge_index = None
        data.edge_attr = None
        nag._list[i_level] = data
        return nag

    # To guide the sampling for superedges, we want to sample among
    # points whose neighbors in the level-0 adjacency graph belong to
    # a different cluster in the i_level graph. To this end, we first
    # need to tell whose i_level cluster each level-0 point belongs to.
    # This step requires having access to the whole NAG, since we need
    # to convert level-0 point indices into their corresponding level-i
    # superpoint indices
    super_index = nag.get_super_index(i_level)

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
    # features will be created at the very end
    edges_point, _ = to_trimmed(edges_point)

    # Now we are only interested in the edges connecting two different
    # clusters and not in the intra-cluster connections. Select only
    # inter-cluster edges and compute the corresponding source and
    # target point and cluster indices
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

    # Prepare data attributes before computing edge features
    data.edge_index = se
    data.is_artificial = is_isolated

    # Edge feature computation. NB: operates on trimmed graphs only.
    # Features for all undirected edges can be computed later using
    # `_on_the_fly_horizontal_edge_features()`
    data = _minimalistic_horizontal_edge_features(
        data, nag[0].pos, edges_point, se_id)

    # Restore the i_level Data object, if need be
    nag._list[i_level] = data

    return nag


class RadiusHorizontalGraph(Transform):
    """Compute horizontal edges for all NAG levels except its first
    (ie the 0-level). These are the edges connecting the segments at
    each level, equipped with handcrafted edge features.

    This approach relies on a fast heuristics to search neighboring
    segments as well as to identify level-0 points making up the
    'subedges' between the segments.

    By default, a series of handcrafted edge attributes are computed and
    stored in the corresponding `Data.edge_attr`. However, if one only
    needs a subset of those at train time, one may make use of
    `SelectColumns` and `NAGSelectColumns`.

    :param k_max: int, List(int)
        Maximum number of neighbors per segment
    :param gap: float, List(float)
        Two segments A and B are considered neighbors if there is a in A
        and b in B such that dist(a, b) < gap
    :param k_ratio: float
        Maximum ratio of a segment's points than can be used in a
        superedge's subedges
    :param k_min: int
        Minimum of subedges per superedge
    :param cycles: int
        Number of iterations for nearest neighbor search between
        segments
    :param margin: float
        Tolerance margin used for selecting subedges points and
        excluding segment points from potential subedge candidates
    :param chunk_size: int, float
        Allows mitigating memory use. If `chunk_size > 1`,
        `edge_index` will be processed into chunks of `chunk_size`. If
        `0 < chunk_size < 1`, then `edge_index` will be divided into
        parts of `edge_index.shape[1] * chunk_size` or less
    :param halfspace_filter: bool
        Whether the halfspace filtering should be applied
    :param bbox_filter: bool
        Whether the bounding box filtering should be applied
    :param target_pc_flip: bool
        Whether the subedge point pairs should be carefully ordered
    :param source_pc_sort: bool
        Whether the source and target subedge point pairs should be
        ordered along the same vector
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG
    _NO_REPR = ['chunk_size']

    def __init__(
            self, k_max=100, gap=0, k_ratio=0.2, k_min=20, cycles=3,
            margin=0.2, chunk_size=100000, halfspace_filter=True,
            bbox_filter=True, target_pc_flip=True, source_pc_sort=False):
        self.k_max = k_max
        self.gap = gap
        self.k_ratio = k_ratio
        self.k_min = k_min
        self.cycles = cycles
        self.margin = margin
        self.chunk_size = chunk_size
        self.halfspace_filter = halfspace_filter
        self.bbox_filter = bbox_filter
        self.target_pc_flip = target_pc_flip
        self.source_pc_sort = source_pc_sort

    def _process(self, nag):
        # Convert parameters to list for each NAG level, if need be
        k_ratio = self.k_ratio if isinstance(self.k_ratio, list) \
            else [self.k_ratio] * (nag.num_levels - 1)
        k_min = self.k_min if isinstance(self.k_min, list) \
            else [self.k_min] * (nag.num_levels - 1)
        cycles = self.cycles if isinstance(self.cycles, list) \
            else [self.cycles] * (nag.num_levels - 1)
        margin = self.margin if isinstance(self.margin, list) \
            else [self.margin] * (nag.num_levels - 1)
        chunk_size = self.chunk_size if isinstance(self.chunk_size, list) \
            else [self.chunk_size] * (nag.num_levels - 1)

        # Compute the horizontal graph, without edge features
        nag = _horizontal_graph_by_radius(
            nag, k_max=self.k_max, gap=self.gap, trim=True,
            cycles=cycles, chunk_size=chunk_size)

        # Compute the edge features, level by level
        for i_level, kr, km, cy, mg, cs in zip(
                range(1, nag.num_levels), k_ratio, k_min, cycles, margin,
                chunk_size):
            nag = self._process_edge_features_for_single_level(
                nag, i_level, kr, km, cy, mg, cs)

        return nag

    def _process_edge_features_for_single_level(
            self, nag, i_level, k_ratio, k_min, cycles, margin, chunk_size):
        # Compute 'subedges', ie edges between level-0 points making up
        # the edges between the segments. These will be used for edge
        # features computation. NB: this operation simplifies the
        # edge_index graph into a trimmed graph. To restore
        # the bidirectional edges, we will need to reconstruct the j<i
        # edges later on (done in `_horizontal_edge_features`)
        edge_index, se_point_index, se_id = subedges(
            nag[0].pos, nag.get_super_index(i_level), nag[i_level].edge_index,
            k_ratio=k_ratio, k_min=k_min, cycles=cycles, pca_on_cpu=True,
            margin=margin, halfspace_filter=self.halfspace_filter,
            bbox_filter=self.bbox_filter, target_pc_flip=self.target_pc_flip,
            source_pc_sort=self.source_pc_sort, chunk_size=chunk_size)

        # Prepare for edge feature computation
        data = nag[i_level]
        data.edge_index = edge_index

        # Edge feature computation. NB: operates on trimmed graph only
        # to alleviate memory and compute. Features for all undirected
        # edges can be computed later using
        # `_on_the_fly_horizontal_edge_features()`
        data = _minimalistic_horizontal_edge_features(
            data, nag[0].pos, se_point_index, se_id)

        # Restore the i_level Data object
        nag._list[i_level] = data

        return nag


def _horizontal_graph_by_radius(
        nag, k_max=100, gap=0, trim=True, cycles=3, chunk_size=None):
    """Search neighboring segments with points distant from `gap`or
    less.

    :param nag: NAG
        Hierarchical structure
    :param k_max: int, List(int)
        Maximum number of neighbors per segment
    :param gap: float, List(float)
        Two segments A and B are considered neighbors if there is a in A
        and b in B such that dist(a, b) < gap
    :param trim: bool
        Whether the returned horizontal graph should be trimmed. If
        True, `to_trimmed()` will be called and all edges will be
        expressed with source_index < target_index, self-loops and
        redundant edges will be removed. This may be necessary to
        alleviate memory consumption before computing edge features
    :param cycles int
        Number of iterations. Starting from a point X in set A, one
        cycle accounts for searching the nearest neighbor, in A, of the
        nearest neighbor of X in set B
    :param chunk_size: int, float
        Allows mitigating memory use when computing the subedges. If
        `chunk_size > 1`, `edge_index` will be processed into chunks of
        `chunk_size`. If `0 < chunk_size < 1`, then `edge_index` will be
        divided into parts of `edge_index.shape[1] * chunk_size` or less
    :return:
    """
    assert isinstance(nag, NAG)
    if not isinstance(k_max, list):
        k_max = [k_max] * (nag.num_levels - 1)
    if not isinstance(gap, list):
        gap = [gap] * (nag.num_levels - 1)
    if not isinstance(cycles, list):
        cycles = [cycles] * (nag.num_levels - 1)
    if not isinstance(chunk_size, list):
        chunk_size = [chunk_size] * (nag.num_levels - 1)

    for i_level, k, g, cy, cs in zip(
            range(1, nag.num_levels), k_max, gap, cycles, chunk_size):
        nag = _horizontal_graph_by_radius_for_single_level(
            nag, i_level, k_max=k, gap=g, trim=trim,
            cycles=cy, chunk_size=cs)

    return nag


def _horizontal_graph_by_radius_for_single_level(
        nag, i_level, k_max=100, gap=0, trim=True, cycles=3,
        chunk_size=100000):
    """

    :param nag:
    :param i_level:
    :param k_max:
    :param gap:
    :param trim:
    :param cycles:
    :param chunk_size:
    :return:
    """
    assert isinstance(nag, NAG)
    assert i_level > 0, "Cannot compute cluster graph on level 0"
    assert nag[0].num_nodes < np.iinfo(np.uint32).max, \
        "Too many nodes for `uint32` indices"
    assert nag[0].num_edges < np.iinfo(np.uint32).max, \
        "Too many edges for `uint32` indices"

    # Recover the i_level Data object we will be working on
    data = nag[i_level]
    num_nodes = data.num_nodes

    # Remove any already-existing horizontal graph
    data.edge_index = None
    data.edge_attr = None

    # Exit in case the i_level graph contains only one node
    if num_nodes < 2:
        nag._list[i_level] = data
        return nag

    # Compute the super_index for level-0 points wrt the target level
    super_index = nag.get_super_index(i_level)

    # Search neighboring clusters
    edge_index, distances = cluster_radius_nn(
        nag[0].pos, super_index, k_max=k_max, gap=gap, trim=trim,
        cycles=cycles, chunk_size=chunk_size)

    # Save the graph in the Data object
    data.edge_index = edge_index
    data.edge_attr = distances

    # Search for nodes which received no edges and connect them to their
    # nearest neighbor
    data.connect_isolated(k=1)

    # Trim the graph. This is temporary, to alleviate edge features
    # computation
    if trim:
        data.to_trimmed(reduce='min')

    # Store the updated Data object in the NAG
    nag._list[i_level] = data

    return nag


def _minimalistic_horizontal_edge_features(data, points, se_point_index, se_id):
    """Compute the features for horizontal edges, given the edge graph
    and the level-0 'subedges' making up each edge.

    The features computed here are partly based on:
    https://github.com/loicland/superpoint_graph

    :param data:
    :param points:
    :param se_point_index:
    :param se_id:
    :return:
    """
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

    # Recover the edges between the segments
    se = data.edge_index

    assert is_trimmed(se), \
        "Expects the graph to be trimmed, consider using " \
        "`src.utils.to_trimmed()` before computing the features"
    assert getattr(data, 'normal', None) is not None, \
        "Expects input Data object to have a 'normal' attribute, holding the " \
        "segment normal vectors. See `src.utils.scatter_pca` to efficiently " \
        "compute PCA on segments"

    # Direction are the pointwise source->target vectors, based on which
    # we will compute superedge descriptors
    offset = points[se_point_index[1]] - points[se_point_index[0]]

    # To stabilize the distance-based features' distribution, we use the
    # sqrt of the metric distance. This assumes coordinates are in meter
    # and that we are mostly interested in the range [1, 100]. Might
    # want to change this if your dataset is different
    dist = torch.linalg.norm(offset, dim=1)

    # Compute mean, min and std subedge direction
    se_mean_off = scatter_mean(offset, se_id, dim=0)
    se_std_off = scatter_std(offset, se_id, dim=0)

    # Compute mean subedge distance
    se_mean_dist = scatter_mean(dist, se_id, dim=0).sqrt()

    # The superedges we have created so far are oriented. We need to
    # create the edges and corresponding features for the Target->Source
    # direction now
    se_feat = torch.vstack([se_mean_off.T, se_std_off.T, se_mean_dist]).T

    # Save superedges and superedge features in the Data object
    data.edge_index = se
    data.edge_attr = se_feat

    return data


class OnTheFlyEdgeFeatures(Transform):
    """Compute edge features "on-the-fly" for all i->j and j->i
    horizontal edges of the NAG levels except its first (ie the
    0-level).

    Expects only trimmed edges as input, along with some edge-specific
    attributes that cannot be recovered from the corresponding source
    and target node attributes (see `src.utils.to_trimmed`).

    Accepts input edge_attr to be float16, to alleviate memory use and
    accelerate data loading and transforms. Output edge_attr will,
    however, be in float32.

    Optionally adds some edge features that can be recovered from the
    source and target node attributes.

    Builds the j->i edges and corresponding features based on their i->j
    counterparts in the trimmed graph.

    Equips the output NAG with all i->j and j->i nodes and corresponding
    features.

    Note: this transform is intended to be called after all sampling
    transforms, to mitigate compute and memory impact of horizontal
    edges. Besides, it expects the input `Data.edge_attr` to hold 5
    features precomputed with `_minimalistic_horizontal_edge_features`.

    :param mean_dist:
    :param min_dist:
    :param std_dist:
    :param angle_source:
    :param angle_target:
    :param centroid_direction:
    :param centroid_dist:
    :param normal_angle:
    :param log_length:
    :param log_surface:
    :param log_volume:
    :param log_size:
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(
            self, mean_offset=True, std_offset=True, mean_dist=True,
            angle_source=True, angle_target=True, centroid_direction=True,
            centroid_dist=True, normal_angle=True, log_length=True,
            log_surface=True, log_volume=True, log_size=True):
        self.mean_offset = mean_offset
        self.std_offset = std_offset
        self.mean_dist = mean_dist
        self.angle_source = angle_source
        self.angle_target = angle_target
        self.centroid_direction = centroid_direction
        self.centroid_dist = centroid_dist
        self.normal_angle = normal_angle
        self.log_length = log_length
        self.log_surface = log_surface
        self.log_volume = log_volume
        self.log_size = log_size

    def _process(self, nag):
        for i_level in range(1, nag.num_levels):
            nag._list[i_level] = _on_the_fly_horizontal_edge_features(
                nag[i_level],
                mean_offset=self.mean_offset,
                std_offset=self.std_offset,
                mean_dist=self.mean_dist,
                angle_source=self.angle_source,
                angle_target=self.angle_target,
                centroid_direction=self.centroid_direction,
                centroid_dist=self.centroid_dist,
                normal_angle=self.normal_angle,
                log_length=self.log_length,
                log_surface=self.log_surface,
                log_volume=self.log_volume,
                log_size=self.log_size)
        return nag


def _on_the_fly_horizontal_edge_features(
        data, mean_offset=True, std_offset=True, mean_dist=True, angle_source=True,
        angle_target=True, centroid_direction=True, centroid_dist=True,
        normal_angle=True, log_length=True, log_surface=True, log_volume=True,
        log_size=True):
    """Compute all edges and edge features for a horizontal graph, given
    a trimmed graph and some precomputed edge attributes.

    For each edge i->j, this will build additional edge features from
    the node attributes, as well as the symmetric j->i edge and
    corresponding features.

    :param data:
    :param mean_offset:
    :param std_offset:
    :param mean_dist:
    :param angle_source:
    :param angle_target:
    :param centroid_direction:
    :param centroid_dist:
    :param normal_angle:
    :param log_length:
    :param log_surface:
    :param log_volume:
    :param log_size:
    :return:
    """

    # Recover the edges between the segments
    se = data.edge_index

    assert is_trimmed(se), \
        "Expects the graph to be trimmed, consider using " \
        "`src.utils.to_trimmed()` before computing the features"
    assert not angle_source or getattr(data, 'normal', None) is not None, \
        "Expects input Data to have a 'normal' attribute"
    assert not angle_target or getattr(data, 'normal', None) is not None, \
        "Expects input Data to have a 'normal' attribute"
    assert not normal_angle or getattr(data, 'normal', None) is not None, \
        "Expects input Data to have a 'normal' attribute"
    assert not log_length or getattr(data, 'log_length', None) is not None, \
        "Expects input Data to have a 'log_length' attribute"
    assert not log_surface or getattr(data, 'log_surface', None) is not None, \
        "Expects input Data to have a 'log_surface' attribute"
    assert not log_volume or getattr(data, 'log_volume', None) is not None, \
        "Expects input Data to have a 'log_volume' attribute"
    assert not log_size or getattr(data, 'log_size', None) is not None, \
        "Expects input Data to have a 'log_size' attribute"
    assert getattr(data, 'edge_attr', None) is not None \
           and data.edge_attr.shape[1] == 7, \
        "Expects input Data 'edge_attr' to hold a Ex7 tensor of edge features" \
        " precomputed using `_minimalistic_horizontal_edge_features`: " \
        "se_mean_off, se_std_off and se_mean_dist"

    # Recover already-existing features from the Data.edge_attr.
    # IMPORTANT: these are assumed to have been generated using
    # `_minimalistic_horizontal_edge_features` and to be the following:
    #   - se_mean_off: mean subedge offset
    #   - se_std_off: std subedge offset
    #   - se_mean_dist: mean subedge distance
    # Precomputed edge features might be expressed in float16, so we
    # convert them to float32 here
    se_feat_precomputed = data.edge_attr.float()
    se_mean_off = se_feat_precomputed[:, :3]
    se_std_off = se_feat_precomputed[:, 3:6]
    se_mean_dist = se_feat_precomputed[:, 6]

    # Compute the distance and direction between the segments' centroids
    se_centroid_direction = data.pos[se[1]] - data.pos[se[0]]
    se_centroid_dist = torch.linalg.norm(se_centroid_direction, dim=1)
    se_centroid_direction /= se_centroid_dist.view(-1, 1)
    se_centroid_dist = se_centroid_dist.sqrt()

    # Compute the mean subedge (normalized) direction
    se_direction = se_mean_off / torch.linalg.norm(
        se_mean_off, dim=1).view(-1, 1)

    # Compute some edge features based on segment attributes
    normal = getattr(data, 'normal', None)
    if angle_source and normal is not None:
        se_angle_s = (se_direction * normal[se[0]]).sum(dim=1).abs()
    else:
        se_angle_s = torch.zeros_like(se_centroid_dist)

    if angle_target and normal is not None:
        se_angle_t = (se_direction * normal[se[1]]).sum(dim=1).abs()
    else:
        se_angle_t = torch.zeros_like(se_centroid_dist)

    if normal_angle and normal is not None:
        se_normal_angle = (normal[se[0]] * normal[se[1]]).sum(dim=1).abs()
    else:
        se_normal_angle = torch.zeros_like(se_centroid_dist)

    if log_length and getattr(data, 'log_length', None) is not None:
        se_log_length_ratio = data.log_length[se[0]] - data.log_length[se[1]]
    else:
        se_log_length_ratio = torch.zeros_like(se_centroid_dist)

    if log_surface and getattr(data, 'log_surface', None) is not None:
        se_log_surface_ratio = data.log_surface[se[0]] - data.log_surface[se[1]]
    else:
        se_log_surface_ratio = torch.zeros_like(se_centroid_dist)

    if log_volume and getattr(data, 'log_volume', None) is not None:
        se_log_volume_ratio = data.log_volume[se[0]] - data.log_volume[se[1]]
    else:
        se_log_volume_ratio = torch.zeros_like(se_centroid_dist)

    if log_size and getattr(data, 'log_size', None) is not None:
        se_log_size_ratio = data.log_size[se[0]] - data.log_size[se[1]]
    else:
        se_log_size_ratio = torch.zeros_like(se_centroid_dist)

    # The features we have created so far are only for the trimmed
    # graph. For each edge i->j, need to create the j->i edge and
    # corresponding features
    se = torch.cat((se, se.flip(0)), dim=1)
    se_feat = torch.vstack([  # 14 TOT
        torch.cat((se_mean_off, -se_mean_off)).T,  # 3
        torch.cat((se_std_off, -se_std_off)).T,  # 3
        torch.cat((se_mean_dist, se_mean_dist)),  # 1

        torch.cat((se_angle_s, se_angle_t)),  # 1
        torch.cat((se_angle_t, se_angle_s)),  # 1
        torch.cat((se_centroid_direction, -se_centroid_direction)).T,  # 3
        torch.cat((se_centroid_dist, se_centroid_dist)),  # 1
        torch.cat((se_normal_angle, se_normal_angle)),  # 1
        torch.cat((se_log_length_ratio, -se_log_length_ratio)),  # 1
        torch.cat((se_log_surface_ratio, -se_log_surface_ratio)),  # 1
        torch.cat((se_log_volume_ratio, -se_log_volume_ratio)),  # 1
        torch.cat((se_log_size_ratio, -se_log_size_ratio))]).T  # 1

    # Only keep the required edge attributes
    mask = torch.tensor([
        *[mean_offset] * 3, *[std_offset] * 3, mean_dist, angle_source,
        angle_target, *[centroid_direction] * 3, centroid_dist, normal_angle,
        log_length, log_surface, log_volume, log_size], device=data.device)
    se_feat = se_feat[:, mask]

    # Save superedges and superedge features in the Data object
    data.edge_index = se
    data.edge_attr = se_feat

    return data


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
