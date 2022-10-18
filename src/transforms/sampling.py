import re
import torch
from torch_geometric.nn.pool import voxel_grid
from torch_cluster import grid_cluster
from torch_scatter import scatter_mean, scatter_add, scatter_sum
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from src.utils import fast_randperm
from src.transforms import Transform


__all__ = ['Shuffle', 'SaveOriginalPosId', 'GridSampling3D']


class Shuffle(Transform):
    """Shuffle the order of points in a Data object."""

    def _process(self, data):
        idx = fast_randperm(data.num_points, device=data.device)
        return data.select(idx, update_sub=False, update_super=False)


class SaveOriginalPosId(Transform):
    """Adds the index of the point to the Data object attributes. This
    allows tracking this point from the output back to the input
    data object
    """

    KEY = 'origin_id'

    def __init__(self, key=None):
        self.KEY = key if key is not None else self.KEY

    def _process(self, data):
        if hasattr(data, self.KEY):
            return data

        setattr(data, self.KEY, torch.arange(0, data.pos.shape[0]))
        return data


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
    _VOTING_KEYS = ['y', 'instance_labels', 'super_index']

    # Keys for which voxel aggregation will be based on majority voting
    _LAST_KEYS = ['batch', SaveOriginalPosId.KEY]

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

            assert item.ge(0).all(),\
                "Mean aggregation only supports positive integers"
            assert item.dtype in [torch.uint8, torch.int, torch.long], \
                "Mean aggregation only supports positive integers"
            assert item.ndim <= 2, \
                "Voting and histograms are only supported for 1D and " \
                "2D tensors"

            # Initialization
            voting = key not in bins.keys()
            n_bins = item.max() if voting else bins[key]

            # Important: if values are already 2D, we consider them to
            # be histograms and will simply scatter_add them
            if item.ndim == 2:
                # Aggregate the histograms
                hist = scatter_add(item, cluster, dim=0)
                data[key] = hist

                # Either save the histogram or the majority vote
                data[key] = hist.argmax(dim=-1) if voting else hist
                continue

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
                N, offset, device=device, dtype=torch.long)
            hist = torch.cat((bins_before, hist), dim=1)

            # Append columns to the histogram for unobserved
            # classes/bins
            bins_after = torch.zeros(
                N, n_bins - hist.shape[1], device=device,
                dtype=torch.long)
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
