import math
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_min
from itertools import combinations_with_replacement
from src.utils.edge import edge_wise_points
from torch_geometric.utils import coalesce


__all__ = ['scatter_mean_weighted', 'scatter_pca', 'scatter_nearest_neighbor']


def scatter_mean_weighted(x, idx, w, dim_size=None):
    """Helper for scatter_mean with weights"""
    assert w.ge(0).all(), "Only positive weights are accepted"
    assert w.dim() == idx.dim() == 1, "w and idx should be 1D Tensors"
    assert x.shape[0] == w.shape[0] == idx.shape[0], \
        "Only supports weighted mean along the first dimension"

    # Concatenate w and x in the same tensor to only call scatter once
    w = w.view(-1, 1).float()
    wx = torch.cat((w, x * w), dim=1)

    # Scatter sum the wx tensor to obtain
    wx_segment = scatter_add(wx, idx, dim=0, dim_size=dim_size)

    # Extract the weighted mean from the result
    w_segment = wx_segment[:, 0]
    x_segment = wx_segment[:, 1:]
    w_segment[w_segment == 0] = 1
    mean_segment = x_segment / w_segment.view(-1, 1)

    return mean_segment


def scatter_pca(x, idx, on_cpu=True):
    """Scatter implementation for PCA.

    Returns aigenvalues and eigenvectors for each group in idx.
    If x has shape N1xD and idx covers indices in [0, N2], the
    eigenvalues will have shape N2xD and the eigenvectors will
    have shape N2xDxD. The eigenvalues and eigenvectors are
    sorted by increasing eigenvalue.
    """
    assert idx.dim() == 1
    assert x.dim() == 2
    assert idx.shape[0] == x.shape[0]
    assert x.shape[1] > 1

    d = x.shape[1]
    device = x.device

    # Substract mean
    mean = scatter_mean(x, idx, dim=0)
    x = x - mean[idx]

    # Compute pointwise covariance as a N_1x(DxD) matrix
    ij = torch.tensor(list(combinations_with_replacement(range(d), 2)), device=device)
    upper_triangle = x[:, ij[:, 0]] * x[:, ij[:, 1]]

    # Aggregate the covariances as a N_2x(DxD) with scatter_sum
    # and convert it to a N_2xDxD batch of matrices
    upper_triangle = scatter_add(upper_triangle, idx, dim=0) / d
    cov = torch.empty((upper_triangle.shape[0], d, d), device=device)
    cov[:, ij[:, 0], ij[:, 1]] = upper_triangle

    # Eigendecompostion
    if on_cpu:
        device = cov.device
        cov = cov.cpu()
        eval, evec = torch.linalg.eigh(cov, UPLO='U')
        eval = eval.to(device)
        evec = evec.to(device)
    else:
        eval, evec = torch.linalg.eigh(cov, UPLO='U')

    # If Nan values are computed, return equal eigenvalues and
    # Identity eigenvectors
    idx_nan = torch.where(torch.logical_and(
        eval.isnan().any(1), evec.flatten(1).isnan().any(1)))
    eval[idx_nan] = torch.ones(3, dtype=eval.dtype, device=device)
    evec[idx_nan] = torch.eye(3, dtype=evec.dtype, device=device)

    # Precision errors may cause close-to-zero eigenvalues to be
    # negative. Hard-code these to zero
    eval[torch.where(eval < 0)] = 0

    return eval, evec


def scatter_nearest_neighbor(
        points, index, edge_index, cycles=3, chunk_size=None):
    """For each pair of segments indicated in edge_index, find the 2
    closest points between the two segments.

    NB: this is an approximate, iterative process.

    :param points: (N, D) tensor
        Points
    :param index: (N) LongTensor
        Segment index, for each point
    :param edge_index: (2, E) LongTensor
        Segment pairs for which to compute the nearest neighbors
    :param cycles int
        Number of iterations. Starting from a point X in set A, one
        cycle accounts for searching the nearest neighbor, in A, of the
        nearest neighbor of X in set B
    :param chunk_size: int, float
        Allows mitigating memory use when computing the neighbors. If
        `chunk_size > 1`, `edge_index` will be processed into chunks of
        `chunk_size`. If `0 < chunk_size < 1`, then `edge_index` will be
        divided into parts of `edge_index.shape[1] * chunk_size` or less
    """
    assert edge_index.shape == coalesce(edge_index).shape, \
        "Does not support duplicate edges, please coalesce the edges" \
        " before calling this function"

    # Recursive call in case chunk is specified. Chunk allows limiting
    # the number of edges processed at once. This might alleviate
    # memory use
    if chunk_size is not None and chunk_size > 0:

        # Recursive call on smaller edge_index chunks
        chunk_size = int(chunk_size) if chunk_size > 1 \
            else math.ceil(edge_index.shape[1] * chunk_size)
        num_chunks = math.ceil(edge_index.shape[1] / chunk_size)
        out_list = []
        for i_chunk in range(num_chunks):
            start = i_chunk * chunk_size
            end = (i_chunk + 1) * chunk_size
            out_list.append(scatter_nearest_neighbor(
                points, index, edge_index[:, start:end], cycles=cycles,
                chunk_size=None))

        # Combine outputs
        candidate = torch.cat([elt[0] for elt in out_list], dim=0)
        candidate_idx = torch.cat([elt[1] for elt in out_list], dim=1)

        return candidate, candidate_idx

    # We define the segments in the first row of edge_index as 'source'
    # segments, while the elements of the second row are 'target'
    # segments. The corresponding variables are prepended with 's_' and
    # 't_' for clarity
    s_idx = edge_index[0]
    t_idx = edge_index[1]

    # Expand the edge variables to point-edge values. That is, the
    # concatenation of all the source --or target-- points for each
    # edge. The corresponding variables are prepended with 'S_' and 'T_'
    # for clarity
    (S_points, S_points_idx, S_uid), (T_points, T_points_idx, T_uid) = \
        edge_wise_points(points, index, edge_index)

    # Initialize the candidate points as the centroid of each segment
    segment_centroid = scatter_mean(points, index, dim=0)
    segment_size = index.bincount()
    s_candidate = segment_centroid[s_idx]
    t_candidate = segment_centroid[t_idx]
    s_candidate_idx = -torch.ones_like(s_idx)
    t_candidate_idx = -torch.ones_like(s_idx)

    # Step operation will update the source --target, respectively--
    # candidate based on the current target --source, respectively--
    # candidate
    def step(source=True):
        if source:
             x_idx, y_candidate, X_points, X_points_idx, X_uid = \
                 s_idx, t_candidate, S_points, S_points_idx, S_uid
        else:
            x_idx, y_candidate, X_points, X_points_idx, X_uid = \
                t_idx, s_candidate, T_points, T_points_idx, T_uid

        # Expand the other segments' candidates to point-edge values
        size = segment_size[x_idx]
        Y_candidate = y_candidate.repeat_interleave(size, dim=0)

        # Compute the distance between the points and the other segment's
        # candidate and update the segment's candidate as the point with
        # the smallest distance to the candidate
        X_dist = torch.linalg.norm(X_points - Y_candidate, dim=1)

        # Update the candidate as the point with the smallest distance
        # for each edge
        # TODO: this is the bottleneck of scatter_nearest_neighbor
        _, X_argmin = scatter_min(X_dist, X_uid)
        x_candidate_idx = X_points_idx[X_argmin]
        x_candidate = points[x_candidate_idx]

        return x_candidate, x_candidate_idx

    # Iteratively update the target and source candidates
    for _ in range(cycles):
        t_candidate, t_candidate_idx = step(source=False)
        s_candidate, s_candidate_idx = step(source=True)

    # Stack for output
    candidate = torch.vstack((s_candidate, t_candidate))
    candidate_idx = torch.vstack((s_candidate_idx, t_candidate_idx))

    return candidate, candidate_idx
