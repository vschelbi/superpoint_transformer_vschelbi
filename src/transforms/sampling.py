import re
import torch
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.utils import k_hop_subgraph, to_undirected
from torch_cluster import grid_cluster
from torch_scatter import scatter_mean
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from src.utils import fast_randperm, sparse_sample, knn_2
from src.transforms import Transform
from src.data import NAG, NAGBatch
from src.utils.metrics import atomic_to_histogram


__all__ = [
    'Shuffle', 'SaveNodeIndex', 'NAGSaveNodeIndex', 'GridSampling3D',
    'SampleSubNodes', 'SampleKHopSubgraphs', 'SampleRadiusSubgraphs',
    'SampleSegments', 'SampleEdges']


class Shuffle(Transform):
    """Shuffle the order of points in a Data object."""

    def _process(self, data):
        idx = fast_randperm(data.num_points, device=data.device)
        return data.select(idx, update_sub=False, update_super=False)


class SaveNodeIndex(Transform):
    """Adds the index of the nodes to the Data object attributes. This
    allows tracking nodes from the output back to the input Data object.
    """

    KEY = 'node_id'

    def __init__(self, key=None):
        self.KEY = key if key is not None else self.KEY

    def _process(self, data):
        if hasattr(data, self.KEY):
            return data

        setattr(data, self.KEY, torch.arange(0, data.pos.shape[0]))
        return data


class NAGSaveNodeIndex(SaveNodeIndex):
    """SaveNodeIndex, applied to each NAG level.
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def _process(self, nag):
        transform = SaveNodeIndex(key=self.KEY)
        for i_level in range(nag.num_levels):
            nag._list[i_level] = transform(nag._list[i_level])
        return nag


class GridSampling3D(Transform):
    """ Clusters 3D points into voxels with size :attr:`size`.
    Parameters
    ----------
    size: float
        Size of a voxel (in each dimension).
    quantize_coords: bool
        If True, it will convert the points into their associated sparse
        coordinates within the grid and store the value into a new
        `coords` attribute.
    mode: string:
        The mode can be either `last` or `mean`.
        If mode is `mean`, all the points and their features within a
        cell will be averaged. If mode is `last`, one random points per
        cell will be selected with its associated features.
    hist_key: str or List(str)
        Data attributes for which we would like to aggregate values into
        an histogram. This is typically needed when we want to aggregate
        points labels without losing the distribution, as opposed to
        majority voting.
    hist_size: str or List(str)
        Must be of same size as `hist_key`, indicates the number of
        bins for each key-histogram. This is typically needed when we
        want to aggregate points labels without losing the distribution,
        as opposed to majority voting.
    inplace: bool
        Whether the input Data object should be modified in-place
    verbose: bool
        Verbosity
    """

    _NO_REPR = ['verbose', 'inplace']

    def __init__(
            self, size, quantize_coords=False, mode="mean", hist_key=None,
            hist_size=None, inplace=False, verbose=False):

        hist_key = [] if hist_key is None else hist_key
        hist_size = [] if hist_size is None else hist_size
        hist_key = [hist_key] if isinstance(hist_key, str) else hist_key
        hist_size = [hist_size] if isinstance(hist_size, int) else hist_size

        assert isinstance(hist_key, list)
        assert isinstance(hist_size, list)
        assert len(hist_key) == len(hist_size)

        self.grid_size = size
        self.quantize_coords = quantize_coords
        self.mode = mode
        self.bins = {k: v for k, v in zip(hist_key, hist_size)}
        self.inplace = inplace

        if verbose:
            print(
                "If you need to keep track of the position of your points, use "
                "SaveNodeIndex transform before using GridSampling3D.")

            if self.mode == "last":
                print(
                    "The tensors within data will be shuffled each time this "
                    "transform is applied. Be careful that if an attribute "
                    "doesn't have the size of num_nodes, it won't be shuffled")

    def _process(self, data_in):
        # In-place option will modify the input Data object directly
        data = data_in if self.inplace else data_in.clone()

        # If the aggregation mode is 'last', shuffle the points order.
        # Note that voxelization of point attributes will be stochastic
        if self.mode == 'last':
            data = Shuffle()(data)

        # Convert point coordinates to the voxel grid coordinates
        coords = torch.round((data.pos) / self.grid_size)

        # Match each point with a voxel identifier
        if 'batch' not in data:
            cluster = grid_cluster(coords, torch.ones(3, device=coords.device))
        else:
            cluster = voxel_grid(coords, data.batch, 1)

        # Reindex the clusters to make sure the indices used are
        # consecutive. Basically, we do not want cluster indices to span
        # [0, i_max] without all in-between indices to be used, because
        # this will affect the speed and output size of torch_scatter
        # operations
        cluster, unique_pos_indices = consecutive_cluster(cluster)

        # Perform voxel aggregation
        data = _group_data(
            data, cluster, unique_pos_indices, mode=self.mode, bins=self.bins)

        # Optionally convert quantize the coordinates. This is useful
        # for sparse convolution models
        if self.quantize_coords:
            data.coords = coords[unique_pos_indices].int()

        # Save the grid size in the Data attributes
        data.grid_size = torch.tensor([self.grid_size])

        return data


def _group_data(
        data, cluster=None, unique_pos_indices=None, mode="mean", skip_keys=[],
        bins={}):
    """Group data based on indices in cluster. The option ``mode``
    controls how data gets aggregated within each cluster.

    Warning: this modifies the input Data object in-place

    :param data : Data
    :param cluster : torch.Tensor
        Tensor of the same size as the number of points in data. Each
        element is the cluster index of that point.
    :param unique_pos_indices : torch.tensor
        Tensor containing one index per cluster, this index will be used
        to select features and labels
    :param mode : str
        Option to select how the features and labels for each voxel is
        computed. Can be ``last`` or ``mean``. ``last`` selects the last
        point falling in a voxel as the representative, ``mean`` takes
        the average.
    :param skip_keys: list
        Keys of attributes to skip in the grouping
    :param bins: dict
        Dictionary holding ``{'key': n_bins}`` where ``key`` is a Data
        attribute for which we would like to aggregate values into an
        histogram and ``n_bins`` accounts for the corresponding number
        of bins. This is typically needed when we want to aggregate
        point labels without losing the distribution, as opposed to
        majority voting.
    """

    # Keys for which voxel aggregation will be based on majority voting
    _VOTING_KEYS = ['y', 'instance_labels', 'super_index', 'is_val']

    # Keys for which voxel aggregation will be based on majority voting
    _LAST_KEYS = ['batch', SaveNodeIndex.KEY]

    # Supported mode for aggregation
    _MODES = ['mean', 'last']
    assert mode in _MODES
    if mode == "mean" and cluster is None:
        raise ValueError(
            "In mean mode the cluster argument needs to be specified")
    if mode == "last" and unique_pos_indices is None:
        raise ValueError(
            "In last mode the unique_pos_indices argument needs to be specified")

    # Save the number of nodes here because the subsequent in-place
    # modifications will affect it
    num_nodes = data.num_nodes

    # Aggregate Data attributes for same-cluster points
    for key, item in data:

        # `skip_keys` are not aggregated
        if key in skip_keys:
            continue

        # Edges cannot be aggregated
        if bool(re.search('edge', key)):
            raise ValueError("Edges not supported. Wrong data type.")

        # TODO: adapt 'sub' to make use of CSRData batching ?
        if key == 'sub':
            raise ValueError("'sub' not supported. Wrong data type.")

        # Only torch.Tensor attributes of size Data.num_nodes are
        # considered for aggregation
        if not torch.is_tensor(item) or item.size(0) != num_nodes:
            continue

        # For 'last' mode, use unique_pos_indices to pick values
        # from a single point within each cluster. The same behavior
        # is expected for the _LAST_KEYS
        if mode == 'last' or key in _LAST_KEYS:
            data[key] = item[unique_pos_indices]
            continue

        # For 'mean' mode, the attributes will be aggregated
        # depending on their nature.

        # If the attribute is a boolean, temporarily convert to integer
        # to facilitate aggregation
        is_item_bool = item.dtype == torch.bool
        if is_item_bool:
            item = item.int()

        # For keys requiring a voting scheme or a histogram
        if key in _VOTING_KEYS or key in bins.keys():
            voting = key not in bins.keys()
            n_bins = item.max() + 1 if voting else bins[key]
            hist = atomic_to_histogram(item, cluster, n_bins=n_bins)
            data[key] = hist.argmax(dim=-1) if voting else hist

        # Standard behavior, where attributes are simply
        # averaged across the clusters
        else:
            data[key] = scatter_mean(item, cluster, dim=0)

        # Convert back to boolean if need be
        if is_item_bool:
            data[key] = data[key].bool()

    return data


class SampleSubNodes(Transform):
    """Sample elements at `low`-level, based on which segment they
    belong to at `high`-level.

    The sampling operation is run without replacement and each segment
    is sampled at least `n_min` and at most `n_max` times, within the
    limits allowed by its actual size.

    Optionally, a `mask` can be passed to filter out some `low`-level
    points.

    :param high: int
        Partition level of the segments we want to sample. By default,
        `high=1` to sample the level-1 segments
    :param low: int
        Partition level we will sample from, guided by the `high`
        segments. By default, `high=0` to sample the level-0 points.
        `low=-1` is accepted when level-0 has a `sub` attribute (ie
        level-0 points are themselves segments of `-1` level absent
        from the NAG object).
    :param n_max: int
        Maximum number of `low`-level elements to sample in each
        `high`-level segment
    :param n_min: int
        Minimum number of `low`-level elements to sample in each
        `high`-level segment, within the limits of its size (ie no
        oversampling)
    :param mask: list, np.ndarray, torch.LongTensor, torch.BoolTensor
        Indicates a subset of `low`-level elements to consider. This
        allows ignoring
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(
            self, high=1, low=0, n_max=32, n_min=16, mask=None):
        assert isinstance(high, int)
        assert isinstance(low, int)
        assert isinstance(n_max, int)
        assert isinstance(n_min, int)
        self.high = high
        self.low = low
        self.n_max = n_max
        self.n_min = n_min
        self.mask = mask

    def _process(self, nag):
        idx = nag.get_sampling(
            high=self.high, low=self.low, n_max=self.n_max, n_min=self.n_min,
            return_pointers=False)
        return nag.select(self.low, idx)


class SampleSegments(Transform):
    """Remove randomly-picked nodes from each level 1+ of the NAG. This
    operation relies on `NAG.select()` to maintain index consistency
    across the NAG levels.

    Note: we do not directly prune level-0 points, see `SampleSubNodes`
    for that. For speed consideration, it is recommended to use
    `SampleSubNodes` first before `SampleSegments`, to minimize the
    number of level-0 points to manipulate.

    :param ratio: float or list(float)
        Portion of nodes to be dropped. A list may be passed to prune
        NAG 1+ levels with different probabilities
    :param by_size: bool
        If True, the segment size will affect the chances of being
        dropped out. The smaller the segment, the greater its chances
        to be dropped
    :param by_class: bool
        If True, the classes will affect the chances of being
        dropped out. The more frequent the segment class, the greater
        its chances to be dropped
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, ratio=0.2, by_size=False, by_class=False):
        assert isinstance(ratio, list) and all(0 <= r < 1 for r in ratio) \
               or (0 <= ratio < 1)
        self.ratio = ratio
        self.by_size = by_size
        self.by_class = by_class

    def _process(self, nag):
        if not isinstance(self.ratio, list):
            ratio = [self.ratio] * (nag.num_levels - 1)
        else:
            ratio = self.ratio

        # Drop some nodes from each NAG level. Note that we start
        # dropping from the highest to the lowest level, to accelerate
        # sampling
        device = nag.device
        for i_level in range(nag.num_levels - 1, 0, -1):

            # Negative max_ratios prevent dropout
            if ratio[i_level - 1] <= 0:
                continue

            # Prepare sampling
            num_nodes = nag[i_level].num_nodes
            num_keep = num_nodes - int(num_nodes * ratio[i_level - 1])

            # Initialize all segments with the same weights
            weights = torch.ones(num_nodes, device=device)

            # Compute per-segment weights solely based on the segment
            # size. This is biased towards preserving large segments in
            # the sampling
            if self.by_size:
                node_size = nag.get_sub_size(i_level, low=0)
                size_weights = node_size ** 0.333
                size_weights /= size_weights.sum()
                weights += size_weights

            # Compute per-class weights based on class frequencies in
            # the current NAG and give a weight to each segment
            # based on the rarest class it contains. This is biased
            # towards sampling rare classes
            if self.by_class and nag[i_level].y is not None:
                counts = nag[i_level].y.sum(dim=0).sqrt()
                scores = 1 / (counts + 1)
                scores /= scores.sum()
                mask = nag[i_level].y.gt(0)
                class_weights = (mask * scores.view(1, -1)).max(dim=1).values
                class_weights /= class_weights.sum()
                weights += class_weights.squeeze()

            # Normalize the weights again, in case size or class weights
            # were added
            weights /= weights.sum()

            # Generate sampling indices
            idx = torch.multinomial(weights, num_keep, replacement=False)

            # Select the nodes and update the NAG structure accordingly
            nag = nag.select(i_level, idx)

        return nag


class BaseSampleSubgraphs(Transform):
    """Base class for sampling subgraphs from a NAG. It randomly picks
    `k` seed nodes from `i_level`, from which `k` subgraphs can be
    grown. Child classes must implement `_sample_subgraphs()` to
    describe how these subgraphs are built. Optionally, the see sampling
    can be driven by their class, or their size, using `by_class` and
    `by_size`, respectively.

    This operation relies on `NAG.select()` to maintain index
    consistency across the NAG levels.

    :param i_level: int
        Partition level we want to pick from. By default, `i_level=-1`
        will sample the highest level of the input NAG
    :param k: int
        Number of sub-graphs/seeds to pick
    :param by_size: bool
        If True, the segment size will affect the chances of being
        selected as a seed. The larger the segment, the greater its
        chances to be picked
    :param by_class: bool
        If True, the classes will affect the chances of being
        selected as a seed. The scarcer the segment class, the greater
        its chances to be selected
    :param use_batch: bool
        If True, the 'Data.batch' attribute will be used to guide seed
        sampling across batches. More specifically, if the input NAG is
        a NAGBatch made up of multiple NAGs, the subgraphs will be
        sampled in a way that guarantees each NAG is sampled from.
        Obviously enough, if `k < batch.max() + 1`, not all NAGs will be
        sampled from
    :param disjoint: bool
        If True, subgraphs sampled from the same NAG will be separated
        as distinct NAGs themselves. Instead, when `disjoint=False`,
        subgraphs sampled in the same NAG will be long the same NAG.
        Hence, if two subgraphs share a node, they will be connected
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(
            self, i_level=1, k=1, by_size=False, by_class=False,
            use_batch=True, disjoint=True):
        self.i_level = i_level
        self.k = k
        self.by_size = by_size
        self.by_class = by_class
        self.use_batch = use_batch
        self.disjoint = disjoint

    def _process(self, nag):
        device = nag.device

        # Skip if i_level is None or k<=0. This may be useful to turn
        # this transform into an Identity, if need be
        if self.i_level is None or self.k <= 0:
            return nag

        # Initialization
        i_level = self.i_level if 0 <= self.i_level < nag.num_levels \
            else nag.num_levels - 1
        k = self.k if self.k < nag[i_level].num_nodes \
            else 1

        # Initialize all segments with the same weights
        weights = torch.ones(nag[i_level].num_nodes, device=device)

        # Compute per-segment weights solely based on the segment
        # size. This is biased towards preserving large segments in
        # the sampling
        if self.by_size:
            node_size = nag.get_sub_size(i_level, low=0)
            size_weights = node_size ** 0.333
            size_weights /= size_weights.sum()
            weights += size_weights

        # Compute per-class weights based on class frequencies in
        # the current NAG and give a weight to each segment
        # based on the rarest class it contains. This is biased
        # towards sampling rare classes
        if self.by_class and nag[i_level].y is not None:
            counts = nag[i_level].y.sum(dim=0).sqrt()
            scores = 1 / (counts + 1)
            scores /= scores.sum()
            mask = nag[i_level].y.gt(0)
            class_weights = (mask * scores.view(1, -1)).max(dim=1).values
            class_weights /= class_weights.sum()
            weights += class_weights.squeeze()

        # Normalize the weights again, in case size or class weights
        # were added
        weights /= weights.sum()

        # Generate sampling indices. If the Data object has a 'batch'
        # attribute and 'self.use_batch', use it to guide the sampling
        # across the batches
        batch = getattr(nag[i_level], 'batch', None)
        if batch is not None and self.use_batch:
            idx_list = []
            num_batch = batch.max() + 1
            num_sampled = 0
            k_batch = torch.div(k, num_batch, rounding_mode='floor')
            k_batch = k_batch.maximum(torch.ones_like(k_batch))
            for i_batch in range(num_batch):

                # Try to sample all NAGs in the batch as evenly as
                # possible, within the constraints of k and
                # num_batch
                if i_batch >= num_batch - 1:
                    k_batch = k - num_sampled

                # Compute the sampling indices for the NAG at hand
                mask = torch.where(i_batch == batch)[0]
                idx_ = torch.multinomial(
                    weights[mask], k_batch, replacement=False)
                idx_list.append(mask[idx_])

                # Update number of sampled subgraphs
                num_sampled += k_batch
                if num_sampled >= k:
                    break

            # Aggregate sampling indices
            idx = torch.cat(idx_list)
        else:
            idx = torch.multinomial(weights, k, replacement=False)

        # Sample the NAG and allow subgraphs sharing the same nodes to
        # be connected
        if not self.disjoint:
            return self._sample_subgraphs(nag, i_level, idx)

        # All sampled subgraphs are disjoint
        return NAGBatch.from_nag_list([
            self._sample_subgraphs(nag, i_level, i.view(1)) for i in idx])

    def _sample_subgraphs(self, nag, i_level, idx):
        raise NotImplementedError


class SampleKHopSubgraphs(BaseSampleSubgraphs):
    """Randomly pick segments from `i_level`, along with their `hops`
    neighbors. This can be thought as a spherical sampling in the graph
    of i_level.

    This operation relies on `NAG.select()` to maintain index
    consistency across the NAG levels.

    Note: we do not directly sample level-0 points, see `SampleSubNodes`
    for that. For speed consideration, it is recommended to use
    `SampleSubNodes` first before `SampleKHopSubgraphs`, to minimize the
    number of level-0 points to manipulate.

    :param hops: int
        Number of hops ruling the neighborhood size selected around the
        seed nodes
    :param i_level: int
        Partition level we want to pick from. By default, `i_level=-1`
        will sample the highest level of the input NAG
    :param k: int
        Number of sub-graphs/seeds to pick
    :param by_size: bool
        If True, the segment size will affect the chances of being
        selected as a seed. The larger the segment, the greater its
        chances to be picked
    :param by_class: bool
        If True, the classes will affect the chances of being
        selected as a seed. The scarcer the segment class, the greater
        its chances to be selected
    :param use_batch: bool
        If True, the 'Data.batch' attribute will be used to guide seed
        sampling across batches. More specifically, if the input NAG is
        a NAGBatch made up of multiple NAGs, the subgraphs will be
        sampled in a way that guarantees each NAG is sampled from.
        Obviously enough, if `k < batch.max() + 1`, not all NAGs will be
        sampled from
    :param disjoint: bool
        If True, subgraphs sampled from the same NAG will be separated
        as distinct NAGs themselves. Instead, when `disjoint=False`,
        subgraphs sampled in the same NAG will be long the same NAG.
        Hence, if two subgraphs share a node, they will be connected
    """
    def __init__(
            self, hops=2, i_level=1, k=1, by_size=False, by_class=False,
            use_batch=True, disjoint=False):
        super().__init__(
            i_level=i_level, k=k, by_size=by_size, by_class=by_class,
            use_batch=use_batch, disjoint=disjoint)
        self.hops = hops

    def _sample_subgraphs(self, nag, i_level, idx):
        assert nag[i_level].has_edges, \
            "Expected Data object to have edges for k-hop subgraph sampling"

        # Convert the graph to undirected graph. This is needed because
        # it is likely that the graph has been trimmed (see
        # `src.utils.to_trimmed`), in which case the trimmed edge
        # direction would affect the k-hop search
        edge_index = to_undirected(nag[i_level].edge_index)

        # Search the k-hop neighbors of the sampled nodes
        idx = k_hop_subgraph(
            idx, self.hops, edge_index, num_nodes=nag[i_level].num_nodes)[0]

        # Select the nodes and update the NAG structure accordingly
        return nag.select(i_level, idx)


class SampleRadiusSubgraphs(BaseSampleSubgraphs):
    """Randomly pick segments from `i_level`, along with their
    spherical neighborhood of fixed radius.

    This operation relies on `NAG.select()` to maintain index
    consistency across the NAG levels.

    Note: we do not directly sample level-0 points, see `SampleSubNodes`
    for that. For speed consideration, it is recommended to use
    `SampleSubNodes` first before `SampleRadiusSubgraphs`, to minimize
    the number of level-0 points to manipulate.

    :param r: float
        Radius for spherical sampling
    :param i_level: int
        Partition level we want to pick from. By default, `i_level=-1`
        will sample the highest level of the input NAG
    :param k: int
        Number of sub-graphs/seeds to pick
    :param by_size: bool
        If True, the segment size will affect the chances of being
        selected as a seed. The larger the segment, the greater its
        chances to be picked
    :param by_class: bool
        If True, the classes will affect the chances of being
        selected as a seed. The scarcer the segment class, the greater
        its chances to be selected
    :param use_batch: bool
        If True, the 'Data.batch' attribute will be used to guide seed
        sampling across batches. More specifically, if the input NAG is
        a NAGBatch made up of multiple NAGs, the subgraphs will be
        sampled in a way that guarantees each NAG is sampled from.
        Obviously enough, if `k < batch.max() + 1`, not all NAGs will be
        sampled from
    :param disjoint: bool
        If True, subgraphs sampled from the same NAG will be separated
        as distinct NAGs themselves. Instead, when `disjoint=False`,
        subgraphs sampled in the same NAG will be long the same NAG.
        Hence, if two subgraphs share a node, they will be connected
    """
    def __init__(
            self, r=2, i_level=1, k=1, by_size=False, by_class=False,
            use_batch=True, disjoint=False):
        super().__init__(
            i_level=i_level, k=k, by_size=by_size, by_class=by_class,
            use_batch=use_batch, disjoint=disjoint)
        self.r = r

    def _sample_subgraphs(self, nag, i_level, idx):
        # Neighbors are searched using the node coordinates. This is not
        # the optimal search for cluster-cluster distances, but is the
        # fastest for our needs here. If need be, one could make this
        # search more accurate using something like:
        # `src.utils.neighbors.cluster_radius_nn`

        # TODO: searching using knn_2 was sluggish, switching to brute
        #  force for now. If bottleneck, need to investigate alternative
        #  search approaches
        # # Search using radius knn utils
        # search_mask = torch.ones_like(nag[i_level].pos[:, 0], dtype=torch.bool)
        # search_mask[idx] = False
        # x_search = nag[i_level].pos
        # x_query = nag[i_level].pos[idx]
        # k = x_search.shape[0]
        # neighbors = knn_2(x_search, x_query, k, r_max=self.r)[0]
        #
        # # Convert neighborhoods to node indices for `NAG.select()`
        # neighbors = neighbors.flatten()
        # idx = neighbors[neighbors != -1].unique()

        # TODO: Assuming idx.shape[0] is small, we search spherical
        #  samplings one by one, without any fancy KNN search tool,
        #  because it seems faster that way, probably due to the large
        #  number of neighbors
        idx_select_list = []
        pos = nag[i_level].pos
        for i in idx:
            distance = torch.linalg.norm(pos - pos[i].view(1, -1), dim=1)
            idx_select_list.append(torch.where(distance < self.r)[0])
        idx_select = torch.cat(idx_select_list).unique()

        # Select the nodes and update the NAG structure accordingly
        return nag.select(i_level, idx_select)


class SampleSubNodes(Transform):
    """Sample elements at `low`-level, based on which segment they
    belong to at `high`-level.

    The sampling operation is run without replacement and each segment
    is sampled at least `n_min` and at most `n_max` times, within the
    limits allowed by its actual size.

    Optionally, a `mask` can be passed to filter out some `low`-level
    points.

    :param high: int
        Partition level of the segments we want to sample. By default,
        `high=1` to sample the level-1 segments
    :param low: int
        Partition level we will sample from, guided by the `high`
        segments. By default, `high=0` to sample the level-0 points.
        `low=-1` is accepted when level-0 has a `sub` attribute (ie
        level-0 points are themselves segments of `-1` level absent
        from the NAG object).
    :param n_max: int
        Maximum number of `low`-level elements to sample in each
        `high`-level segment
    :param n_min: int
        Minimum number of `low`-level elements to sample in each
        `high`-level segment, within the limits of its size (ie no
        oversampling)
    :param mask: list, np.ndarray, torch.LongTensor, torch.BoolTensor
        Indicates a subset of `low`-level elements to consider. This
        allows ignoring
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(
            self, high=1, low=0, n_max=32, n_min=16, mask=None):
        assert isinstance(high, int)
        assert isinstance(low, int)
        assert isinstance(n_max, int)
        assert isinstance(n_min, int)
        self.high = high
        self.low = low
        self.n_max = n_max
        self.n_min = n_min
        self.mask = mask

    def _process(self, nag):
        idx = nag.get_sampling(
            high=self.high, low=self.low, n_max=self.n_max, n_min=self.n_min,
            return_pointers=False)
        return nag.select(self.low, idx)


class SampleEdges(Transform):
    """Sample edges based on which source node they belong to.

    The sampling operation is run without replacement and each source
    segment has at least `n_min` and at most `n_max` edges, within the
    limits allowed by its actual number of edges.

    :param level: int or str
        Level at which to sample edges. Can be an int or a str. If the
        latter, 'all' will apply on all levels, 'i+' will apply on
        level-i and above, 'i-' will apply on level-i and below
    :param n_min: int or List(int)
        Minimum number of edges for each node, within the limits of its
        input number of edges
    :param n_max: int or List(int)
        Maximum number of edges for each node
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, level='1+', n_min=16, n_max=32):
        assert isinstance(level, (int, str))
        assert isinstance(n_min, (int, list))
        assert isinstance(n_max, (int, list))
        self.level = level
        self.n_min = n_min
        self.n_max = n_max

    def _process(self, nag):

        # If 'level' is an int, we only need to process a single level
        if isinstance(self.level, int):
            nag._list[self.level] = self._process_single_level(
                nag[self.level], self.n_min, self.n_max)
            return nag

        # If 'level' covers multiple levels, iteratively process levels
        level_n_min = [-1] * nag.num_levels
        level_n_max = [-1] * nag.num_levels

        if self.level == 'all':
            level_n_min = self.n_min if isinstance(self.n_min, list) \
                else [self.n_min] * nag.num_levels
            level_n_max = self.n_max if isinstance(self.n_max, list) \
                else [self.n_max] * nag.num_levels
        elif self.level[-1] == '+':
            i = int(self.level[:-1])
            level_n_min[i:] = self.n_min if isinstance(self.n_min, list) \
                else [self.n_min] * (nag.num_levels - i)
            level_n_max[i:] = self.n_max if isinstance(self.n_max, list) \
                else [self.n_max] * (nag.num_levels - i)
        elif self.level[-1] == '-':
            i = int(self.level[:-1])
            level_n_min[:i] = self.n_min if isinstance(self.n_min, list) \
                else [self.n_min] * i
            level_n_max[:i] = self.n_max if isinstance(self.n_max, list) \
                else [self.n_max] * i
        else:
            raise ValueError(f'Unsupported level={self.level}')

        for i_level, (n_min, n_max) in enumerate(zip(level_n_min, level_n_max)):
            nag._list[i_level] = self._process_single_level(
                nag[i_level], n_min, n_max)

        return nag

    @staticmethod
    def _process_single_level(data, n_min, n_max):
        # Skip process if n_min or n_max is negative or if in put Data
        # has not edges
        if n_min < 0 or n_max < 0 or not data.has_edges:
            return data

        # Compute a sampling for the edges, based on the source node
        # they belong to
        idx = sparse_sample(
            data.edge_index[0], n_max=n_max, n_min=n_min, return_pointers=False)

        # Select edges and their attributes, if relevant
        data.edge_index = data.edge_index[:, idx]
        if data.has_edge_attr:
            data.edge_attr = data.edge_attr[idx]
        for key in data.edge_keys:
            data[key] = data[key][idx]

        return data
