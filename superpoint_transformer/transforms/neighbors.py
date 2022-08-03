import torch
from superpoint_transformer.data import Data
from superpoint_transformer.partition.FRNN import frnn


def _search_outliers(
        xyz_query, xyz_search, k_min, r_max=1, recursive=False, q_in_s=False):
    """Optionally recursive outlier search. The `xyz_query` and
    `xyz_search`. Search for points with less than `k_min` neighbors
    within a radius of `r_max`.

    Since removing outliers may cause some points to become outliers
    themselves, this problem can be tackled with the `recursive` option.
    Note that this recursive search holds no guarantee of reasonable
    convergence as one could design a point cloud for given `k_min` and
    `r_max` whose points would all recursively end up as outliers.
    """
    # Data initialization
    xyz_query = xyz_query.view(1, -1, 3)
    xyz_search = xyz_search.view(1, -1, 3)
    device = xyz_query.device

    # KNN on GPU. Actual neighbor search now
    neighbors = frnn.frnn_grid_points(
        xyz_query, xyz_search, K=k_min + q_in_s, r=r_max)[1]

    # If the Query points are included in the Search points, remove each
    # point from its own neighborhood
    if q_in_s:
        neighbors = neighbors[0][:, 1:]

    # Get the number of found neighbors for each point. Indeed,
    # depending on the cloud properties and the chosen K and radius,
    # some points may receive "-1" neighbors
    n_found_nn = (neighbors != -1).sum(dim=1)

    # Identify points which have less than k_min neighbor. Those are
    # treated as outliers
    mask_outliers = n_found_nn < k_min
    idx_outliers = torch.where(mask_outliers)[0]
    idx_inliers = torch.where(~mask_outliers)[0]

    # Exit here if not recursively searching for outliers
    if not recursive:
        return idx_outliers, idx_inliers

    # Identify the points affected by the removal of the outliers. Those
    # inliers are potential outliers
    idx_potential = torch.where(
        torch.isin(neighbors[idx_inliers], idx_outliers).any(dim=1))[0]

    # Exit here if there are no potential new outliers among the inliers
    if idx_potential.shape[0] == 0:
        return idx_outliers, idx_inliers

    # Recursviely search actual outliers among the potential
    xyz_query_sub = xyz_query[0, idx_inliers[idx_potential]]
    xyz_search_sub = xyz_search[0, idx_inliers]
    idx_outliers_sub, idx_inliers_sub = _search_outliers(
        xyz_query_sub, xyz_search_sub, k_min, r_max=r_max, recursive=True,
        q_in_s=True)

    # Update the outliers mask
    mask_outliers[idx_inliers[idx_potential][idx_outliers_sub]] = True
    idx_outliers = torch.where(mask_outliers)[0]
    idx_inliers = torch.where(~mask_outliers)[0]

    return idx_outliers, idx_inliers


def search_outliers(data, k_min, r_max=1, recursive=False):
    """Search for points with less than `k_min` neighbors within a
    radius of `r_max`.

    Since removing outliers may cause some points to become outliers
    themselves, this problem can be tackled with the `recursive` option.
    Note that this recursive search holds no guarantee of reasonable
    convergence as one could design a point cloud for given `k_min` and
    `r_max` whose points would all recursively end up as outliers.
    """
    # Actual outlier search, optionally recursive
    idx_outliers, idx_inliers = _search_outliers(
        data.pos, data.pos, k_min, r_max=r_max, recursive=recursive,
        q_in_s=True)

    # Create a Data object for the inliers and outliers
    # Save the index for these isolated points in the Data object. This
    # will help properly handle neighborhoods, features and adjacency
    # graph for those specific points.
    # NB: it is important this attribute follows the "*index" naming
    # convention, see:
    # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
    data_in = Data()
    data_out = Data(outliers_index=idx_outliers)
    for key, item in data:
        if torch.is_tensor(item) and item.size(0) == data.num_nodes:
            data_in[key] = data[key][idx_inliers]
            data_out[key] = data[key][idx_outliers]

    return data_in, data_out


def oversample_partial_neighborhoods(neighbors, distances, k):
    """Oversample partial neighborhoods with less than k points. Missing
    neighbors are indicated by the "-1" index.

    Remarks
      - Neighbors and distances are assumed to be sorted in order of
      increasing distance
      - All neighbors are assumed to have at least one valid neighbor.
      See `search_outliers` to remove points with not enough neighbors
    """
    # Initialization
    assert neighbors.dim() == distances.dim() == 2
    device = neighbors.device

    # Get the number of found neighbors for each point. Indeed,
    # depending on the cloud properties and the chosen K and radius,
    # some points may receive `-1` neighbors
    n_found_nn = (neighbors != -1).sum(dim=1)

    # Identify points which have more than k_min and less than k
    # neighbors within R. For those, we oversample the neighbors to
    # reach k
    idx_partial = torch.where(n_found_nn < k)[0]
    neighbors_partial = neighbors[idx_partial]
    distances_partial = distances[idx_partial]

    # Since the neighbors are sorted by increasing distance, the missing
    # neighbors will always be the last ones. This helps finding their
    # number and position, for oversampling.

    # *******************************************************************
    # The above statement is actually INCORRECT because the outlier
    # removal may produce "-1" neighbors at unexpected positions. So
    # either we manage to treat this in a clean vectorized way, or we
    # fall back to the 2-searches solution...
    # Honestly, this feels like it is getting out of hand, let's keep
    # things simple, since we are not going to save so much computation
    # time with KNN wrt the partition.
    # *******************************************************************

    # For each missing neighbor, compute the size of the discrete set to
    # oversample from.
    n_valid = n_found_nn[idx_partial].repeat_interleave(
        k - n_found_nn[idx_partial])

    # Compute the oversampling row indices.
    idx_x_sampling = torch.arange(neighbors_partial.shape[0], device=device
                                  ).repeat_interleave(k - n_found_nn[idx_partial])

    # Compute the oversampling column indices. The 0.9999 factor is a
    # security to handle the case where torch.rand is to close to 1.0,
    # which would yield incorrect sampling coordinates that would in
    # result in sampling '-1' indices (ie all we try to avoid here)
    idx_y_sampling = (n_valid * torch.rand(
        n_valid.shape[0], device=device) * 0.9999).floor().long()

    # Apply the oversampling
    idx_missing = torch.where(neighbors_partial == -1)
    neighbors_partial[idx_missing] = neighbors_partial[
        idx_x_sampling, idx_y_sampling]
    distances_partial[idx_missing] = distances_partial[
        idx_x_sampling, idx_y_sampling]

    # Restore the oversampled neighborhods with the rest
    neighbors[idx_partial] = neighbors_partial
    distances[idx_partial] = distances_partial

    return neighbors, distances


def search_neighbors(data, k, r_max=1):
    # Data initialization
    xyz_query = data.pos.view(1, -1, 3)
    xyz_search = data.pos.view(1, -1, 3)

    #     #--------------------------------
    #     # KNN on GPU. Search for outliers first
    #     _, neighbors, _, _ = frnn.frnn_grid_points(
    #         xyz_query, xyz_search, K=k_min + 1, r=r_max)

    #     # Remove each point from its own neighborhood
    #     neighbors = neighbors[0][:, 1:]

    #     # Get the number of found neighbors for each point. Indeed,
    #     # depending on the cloud properties and the chosen K and radius,
    #     # some points may receive `-1` neighbors
    #     n_found_nn = (neighbors != -1).sum(dim=1)

    #     # Identify points which have less than k_min neighbors within R.
    #     # Those are treated as outliers and will be discarded
    #     idx_isolated = torch.where(n_found_nn < k_min)[0]

    #     # Save the outliers in a separate Data object
    #     outliers = Data(
    #         pos=data.pos[idx_isolated], rgb=data.rgb[idx_isolated],
    #         y=data.y[idx_isolated], idx_isolated=idx_isolated)

    #     # KNN on GPU. Search for outliers first
    #     _, neighbors, _, _ = frnn.frnn_grid_points(
    #         xyz_query, xyz_search, K=k_min + 1, r=r_max)
    #     #--------------------------------

    # KNN on GPU. Actual neighbor search now
    distances, neighbors, _, _ = frnn.frnn_grid_points(
        xyz_query, xyz_search, K=k + 1, r=r_max)

    # Remove each point from its own neighborhood
    neighbors = neighbors[0][:, 1:]
    distances = distances[0][:, 1:]

    # Oversample the neighborhoods where less than k points were found
    neighbors, distances = oversample_partial_neighborhoods(
        neighbors, distances, k)

    # Store the neighbors and distances as a Data object attribute
    data.neighbors = neighbors.cpu()
    data.distances = distances.cpu()

    # Save the index for these isolated points in the Data object. This
    # will help properly handle neighborhoods, features and adjacency
    # graph for those specific points.
    # NB: it is important this attribute follows the "*index" naming
    # convention, see:
    # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
    # data.isolated_index = idx_isolated.cpu()

    return data

# IMPORTANT !!!
#   - points with no neighbors within radius -> set to 0-feature !