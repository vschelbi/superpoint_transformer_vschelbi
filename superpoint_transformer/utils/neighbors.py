import torch
import superpoint_transformer
from superpoint_transformer.partition.FRNN import frnn


__all__ = [
    'knn_1', 'knn_2', 'inliers_split', 'outliers_split',
    'inliers_outliers_splits']


def knn_1(
        xyz, k, r_max=1, oversample=False, self_is_neighbor=False,
        verbose=False):
    """Search k-NN inside for a 3D point cloud xyz. This search differs
    from `knn_2` in that it operates on a single cloud input (search and
    query are the same) and it allows oversampling the neighbors when
    less than `k` neighbors are found within `r_max`
    """
    assert isinstance(xyz, torch.Tensor)
    assert k >= 1
    assert xyz.dim() == 2

    # Data initialization
    device = xyz.device
    xyz_query = xyz.view(1, -1, 3).cuda()
    xyz_search = xyz.view(1, -1, 3).cuda()

    # KNN on GPU. Actual neighbor search now
    k_search = k if self_is_neighbor else k + 1
    distances, neighbors, _, _ = frnn.frnn_grid_points(
        xyz_query, xyz_search, K=k_search, r=r_max)

    # Remove each point from its own neighborhood
    neighbors = neighbors[0] if self_is_neighbor else neighbors[0][:, 1:]
    distances = distances[0] if self_is_neighbor else distances[0][:, 1:]

    # Oversample the neighborhoods where less than k points were found
    if oversample:
        neighbors, distances = oversample_partial_neighborhoods(
            neighbors, distances, k)

    # Restore the neighbors and distances to the input device
    neighbors = neighbors.to(device)
    distances = distances.to(device)

    if not verbose and not superpoint_transformer.is_debug_enabled():
        return neighbors, distances

    # Warn the user of partial and empty neighborhoods
    num_nodes = neighbors.shape[0]
    n_missing = (neighbors < 0).sum(dim=1)
    n_partial = (n_missing > 0).sum()
    n_empty = (n_missing == k).sum()
    if n_partial == 0:
        return neighbors, distances

    print(
        f"\nWarning: {n_partial}/{num_nodes} points have partial "
        f"neighborhoods and {n_empty}/{num_nodes} have empty "
        f"neighborhoods (missing neighbors are indicated by -1 indices).")

    return neighbors, distances


def knn_2(x_search, x_query, k, r_max=1):
    """Search k-NN of x_query inside x_search, within radius `r_max`.
    """
    assert isinstance(x_search, torch.Tensor)
    assert isinstance(x_query, torch.Tensor)
    assert k >= 1
    assert x_search.dim() == 2
    assert x_query.dim() == 2
    assert x_query.shape[1] == x_search.shape[1]

    k = torch.Tensor([k])
    r_max = torch.Tensor([r_max])

    # Data initialization
    device = x_search.device
    xyz_query = x_query.view(1, -1, 3).cuda()
    xyz_search = x_search.view(1, -1, 3).cuda()

    # KNN on GPU. Actual neighbor search now
    distances, neighbors, _, _ = frnn.frnn_grid_points(
        xyz_query, xyz_search, K=k, r=r_max)

    # Remove each point from its own neighborhood
    neighbors = neighbors[0].to(device)
    distances = distances[0].to(device)
    if k == 1:
        neighbors = neighbors[:, 0]
        distances = distances[:, 0]

    return distances, neighbors


def inliers_split(
        xyz_query, xyz_search, k_min, r_max=1, recursive=False, q_in_s=False):
    """Optionally recursive inlier search. The `xyz_query` and
    `xyz_search`. Search for points with less than `k_min` neighbors
    within a radius of `r_max`.

    Since removing outliers may cause some points to become outliers
    themselves, this problem can be tackled with the `recursive` option.
    Note that this recursive search holds no guarantee of reasonable
    convergence as one could design a point cloud for given `k_min` and
    `r_max` whose points would all recursively end up as outliers.
    """
    return inliers_outliers_splits(
        xyz_query, xyz_search, k_min, r_max=r_max, recursive=recursive,
        q_in_s=q_in_s)[0]


def outliers_split(
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
    return inliers_outliers_splits(
        xyz_query, xyz_search, k_min, r_max=r_max, recursive=recursive,
        q_in_s=q_in_s)[1]


def inliers_outliers_splits(
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
    device = xyz_query.device
    xyz_query = xyz_query.view(1, -1, 3).cuda()
    xyz_search = xyz_search.view(1, -1, 3).cuda()

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
        return idx_outliers.to(device), idx_inliers.to(device)

    # Identify the points affected by the removal of the outliers. Those
    # inliers are potential outliers
    idx_potential = torch.where(
        torch.isin(neighbors[idx_inliers], idx_outliers).any(dim=1))[0]

    # Exit here if there are no potential new outliers among the inliers
    if idx_potential.shape[0] == 0:
        return idx_outliers.to(device), idx_inliers.to(device)

    # Recursively search actual outliers among the potential
    xyz_query_sub = xyz_query[0, idx_inliers[idx_potential]]
    xyz_search_sub = xyz_search[0, idx_inliers]
    idx_outliers_sub, idx_inliers_sub = inliers_outliers_splits(
        xyz_query_sub, xyz_search_sub, k_min, r_max=r_max, recursive=True,
        q_in_s=True)

    # Update the outliers mask
    mask_outliers[idx_inliers[idx_potential][idx_outliers_sub]] = True
    idx_outliers = torch.where(mask_outliers)[0]
    idx_inliers = torch.where(~mask_outliers)[0]

    return idx_outliers.to(device), idx_inliers.to(device)


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
    idx_x_sampling = torch.arange(
        neighbors_partial.shape[0], device=device).repeat_interleave(
        k - n_found_nn[idx_partial])

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

    # Restore the oversampled neighborhoods with the rest
    neighbors[idx_partial] = neighbors_partial
    distances[idx_partial] = distances_partial

    return neighbors, distances
