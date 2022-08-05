import torch
import re
from torch_geometric.nn.pool import voxel_grid
from torch_cluster import grid_cluster
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn.pool.consecutive import consecutive_cluster


def shuffle_data(data):
    """ Shuffle the order of nodes in Data. Only `torch.Tensor`
    attributes of size `Data.num_nodes` are affected.

    Warning: this modifies the input Data object in-place

    Parameters
    ----------
    data : Data
    """
    num_points = data.pos.shape[0]
    shuffle_idx = torch.randperm(num_points)
    for key in set(data.keys):
        item = data[key]
        if torch.is_tensor(item) and num_points == item.shape[0]:
            data[key] = item[shuffle_idx]
    return data


def group_data(
        data, cluster=None, unique_pos_indices=None, mode="mean", skip_keys=[],
        bins={}):
    """ Group data based on indices in cluster. The option ``mode``
    controls how data gets aggregated within each cluster.

    Warning: this modifies the input Data object in-place

    Parameters
    ----------
    data : Data
        [description]
    cluster : torch.Tensor
        Tensor of the same size as the number of points in data. Each
        element is the cluster index of that point.
    unique_pos_indices : torch.tensor
        Tensor containing one index per cluster, this index will be used
        to select features and labels
    mode : str
        Option to select how the features and labels for each voxel is
        computed. Can be ``last`` or ``mean``. ``last`` selects the last
        point falling in a voxel as the representative, ``mean`` takes
        the average.
    skip_keys: list
        Keys of attributes to skip in the grouping
    bins: dict
        Dictionary holding ``{'key': n_bins}`` where ``key`` is a Data
        attribute for which we would like to aggregate values into an
        histogram and ``n_bins`` accounts for the corresponding number
        of bins. This is typically needed when we want to aggregate
        point labels without losing the distribution, as opposed to
        majority voting.
    """

    # TODO: adapt 'sub' to make use of CSRData batching ?
    # Keys for which voxel aggregation will be based on majority voting
    _VOTING_KEYS = ['y', 'instance_labels', 'super_index']

    # Keys for which voxel aggregation will be based on majority voting
    _LAST_KEYS = ['batch', SaveOriginalPosId.KEY]

    assert mode in ["mean", "last"]
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
        if bool(re.search("edge", key)):
            raise ValueError("Edges not supported. Wrong data type.")

        # Only torch.Tensor attributes of size Data.num_nodes are
        # considered for aggregation
        if torch.is_tensor(item) and item.size(0) == num_nodes:

            # For 'last' mode, use unique_pos_indices to pick values
            # from a single point within each cluster. The same behavior
            # is expected for the _LAST_KEYS
            if mode == "last" or key in _LAST_KEYS:
                data[key] = item[unique_pos_indices]

            # For 'mean' mode, the attributes will be aggregated
            # depending on their nature
            elif mode == "mean":

                # If the attribute is a boolean, temporarily convert is
                # to integer to facilitate aggregation
                is_item_bool = item.dtype == torch.bool
                if is_item_bool:
                    item = item.int()

                # For keys requiring a voting scheme or a histogram
                if key in _VOTING_KEYS or key in bins.keys():

                    assert item.ge(0).all(),\
                        "Mean aggregation only supports positive integers"
                    assert item.dtype in [torch.uint8, torch.int, torch.long], \
                        "Mean aggregation only supports positive integers"

                    # Initialization
                    voting = key not in bins.keys()
                    n_bins = item.max() if voting else bins[key]

                    # Convert values to one-hot encoding. Values are
                    # temporarily offset to 0 to save some memory and
                    # compute in one-hot encoding and scatter_add
                    offset = item.min()
                    item = torch.nn.functional.one_hot(item - offset)

                    # Count number of occurrence of each value
                    hist = scatter_add(item, cluster, dim=0)
                    N = hist.shape[0]
                    device = hist.device

                    # Prepend 0 columns to the histogram for bins
                    # removed due to offsetting
                    bins_before = torch.zeros(
                        (N, offset), device=device).long()
                    hist = torch.cat((bins_before, hist), dim=1)

                    # Append columns to the histogram for unobserved
                    # classes/bins
                    bins_after = torch.zeros(
                        (N, n_bins - hist.shape[1]), device=device).long()
                    hist = torch.cat((hist, bins_after), dim=1)

                    # Either save the histogram or the majority vote
                    data[key] = hist.argmax(dim=-1) if voting else hist

                # Standard behavior, where attributes are simply
                # averaged across the clusters
                else:
                    data[key] = scatter_mean(item, cluster, dim=0)

                # Convert back to boolean if need be
                if is_item_bool:
                    data[key] = data[key].bool()

    return data


class SaveOriginalPosId:
    """Adds the index of the point to the Data object attributes. This
    allows tracking this point from the output back to the input
    data object
    """

    KEY = "origin_id"

    def __init__(self, key=None):
        self.KEY = key if key is not None else self.KEY

    def _process(self, data):
        if hasattr(data, self.KEY):
            return data

        setattr(data, self.KEY, torch.arange(0, data.pos.shape[0]))
        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return self.__class__.__name__


class GridSampling3D:
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
     bins: dict
        Dictionary holding ``{'key': n_bins}`` where ``key`` is a Data
        attribute for which we would like to aggregate values into an
        histogram and ``n_bins`` accounts for the corresponding number
        of bins. This is typically needed when we want to aggregate
        point labels without losing the distribution, as opposed to
        majority voting.
    inplace: bool
        Whether the input Data object should be modified in-place
    verbose: bool
        Verbosity
    """

    def __init__(
            self, size, quantize_coords=False, mode="mean", bins={},
            inplace=False, verbose=False):
        self.grid_size = size
        self.quantize_coords = quantize_coords
        self.mode = mode
        self.bins = bins
        self.inplace = inplace
        if verbose:
            print(
                "If you need to keep track of the position of your points, use "
                "SaveOriginalPosId transform before using GridSampling3D.")

            if self.mode == "last":
                print(
                    "The tensors within data will be shuffled each time this "
                    "transform is applied. Be careful that if an attribute "
                    "doesn't have the size of num_nodes, it won't be shuffled")

    def _process(self, data_in):
        # In-place option will modify the input Data object directly
        data = data_in if self.inplace else data_in.clone()

        # If the aggregation mode is 'last', shuffle the point order.
        # Note that voxelization of point attributes will be stochastic
        if self.mode == "last":
            data = shuffle_data(data)

        # Convert point coordinates to the voxel grid coordinates
        coords = torch.round((data.pos) / self.grid_size)

        # Match each point with a voxel identifier
        if "batch" not in data:
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
        data = group_data(
            data, cluster, unique_pos_indices, mode=self.mode, bins=self.bins)

        # Optionally convert quantize the coordinates. This is useful
        # for sparse convolution models
        if self.quantize_coords:
            data.coords = coords[unique_pos_indices].int()

        # Save the grid size in the Data attributes
        data.grid_size = torch.tensor([self.grid_size])

        return data

    def __call__(self, data):
        if isinstance(data, list):
            data = [self._process(d) for d in data]
        else:
            data = self._process(data)
        return data

    def __repr__(self):
        return "{}(grid_size={}, quantize_coords={}, mode={})".format(
            self.__class__.__name__, self.grid_size, self.quantize_coords,
            self.mode)


def sample_clusters(data, high=32, low=1, pointers=False):
    """Compute point indices for sampling points inside clusters saved
    in a CSR format.
    """
    #TODO: operate on NAG and careful with indices !
    #TODO: must be able to sample on subset/mask of sub too (for edge sampling)
    #TODO: could optionally sample from i into j with i > j > 0...
    #TODO: modify sampling to prevent duplicates using "choice = torch.randperm(num_nodes)[:self.num]" ?

    # Compute the number of points that will be sampled from each
    # cluster
    if high > 0:
        # k * tanh(x / k) is bounded by k, is ~x for low x and starts
        # saturating at x~k
        n_samples = (high * torch.tanh(
            data.sub.sizes / high)).floor().long()
    else:
        # Fallback to sqrt sampling
        n_samples = data.sub.sizes.sqrt().round().long()

    # Make sure each cluster is sampled at least 'low' times
    n_samples = n_samples.clamp(min=low)

    # Sample values in [0, 1], these will be used to compute
    # corresponding integer indices for vectorized sampling of the
    # points directly from the CSR-format cluster-to-point data
    samples = torch.rand(n_samples.sum())

    # Convert the [0, 1] samples to integer indices in [0, n_samples]
    # depending on the corresponding cluster sampling size. The 0.9999
    # factor is a security to handle the case where torch.rand is to
    # close to 1.0, which would yield incorrect sampling coordinates
    n_bins = n_samples.repeat_interleave(n_samples)
    samples = (samples * 0.9999 * n_bins).floor().long()

    # As they are now, sampling indices are expressed in [0, sub_size].
    # If we want to sample directly from the cluster-to-point CSR data,
    # we need to express those coordinates in the aggregated CSR values,
    # that is to say we need to apply the cumulative cluster sizes as
    # offsets to the indices. This is typically what is stored in
    # CSRData.pointers
    offsets = data.sub.pointers[:-1].repeat_interleave(n_samples)
    samples = samples + offsets

    # Now we can get the sampled point indices
    idx_samples = data.sub.points[samples]

    # Return here if sampling pointers are not required
    if not pointers:
        return idx_samples

    # Compute the pointers
    ptr_samples = torch.cat([torch.LongTensor([0]), n_samples.cumsum(dim=0)])

    return idx_samples, ptr_samples.contiguous()
