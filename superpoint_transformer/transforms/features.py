import torch
import numpy as np
import superpoint_transformer.partition.utils.libpoint_utils as point_utils


def compute_point_features(
        data, pos=False, radius=5, rgb=True, linearity=True, planarity=True,
        scattering=True, verticality=True, normal=True, length=False,
        surface=False, volume=False, k_min=5):
    """ Compute the pointwise features that will be used for the
    partition.

    All local geometric features assume the input ``Data`` has a
    ``neighbors`` attribute, holding a ``(num_nodes, k)`` tensor of
    indices. All k neighbors will be used for local geometric features
    computation, unless some are missing (indicated by -1 indices). If
    the latter, only positive indices will be used.

    Parameters
    ----------
    pos: bool
        Use point position.
    rgb: bool
        Use rgb color. Assumes Data.rgb holds either [0, 1] floats or
        [0, 255] integers
    linearity: bool
        Use local linearity. Assumes ``Data.neighbors``.
    planarity: bool
        Use local planarity. Assumes ``Data.neighbors``.
    scattering: bool
        Use local scattering. Assumes ``Data.neighbors``.
    verticality: bool
        Use local verticality. Assumes ``Data.neighbors``.
    normal: bool
        Use local normal. Assumes ``Data.neighbors``.
    length: bool
        Use local length. Assumes ``Data.neighbors``.
    surface: bool
        Use local surface. Assumes ``Data.neighbors``.
    volume: bool
        Use local volume. Assumes ``Data.neighbors``.
    k_min: int
        Minimum number of neighbors to consider for geometric features
        computation. Points with less than k_min neighbors will receive
        0-features. Assumes ``Data.neighbors``.
    """
    assert data.has_neighbors, "Data is expected to have a 'neighbors' attribute"
    assert data.num_nodes < np.iinfo(np.uint32).max, "Too many nodes for `uint32` indices"
    assert data.neighbors.max() < np.iinfo(np.uint32).max, "Too high 'neighbors' indices for `uint32` indices"

    features = []

    # Add xyz normalized. The scaling factor drives the maximum cluster
    # size the partition may produce
    if pos and data.pos is not None:
        features.append((data.pos - data.pos.mean(dim=0)) / radius)

    # Add rgb to the features. If colors are stored in int, we assume
    # they are encoded in  [0, 255] and normalize them. Otherwise, we
    # assume they have already been [0, 1] normalized
    if rgb and data.rgb is not None:
        f = data.rgb
        if f.type in [torch.uint8, torch.int, torch.long]:
            f = f.float() / 255
        features.append(f)

    # Add local geometric features
    needs_geof = any((linearity, planarity, scattering, verticality, normal))
    if needs_geof and data.pos is not None:

        # Prepare data for numpy boost interface. Note: we add each
        # point to its own neighborhood before computation
        xyz = data.pos.cpu().numpy()
        nn = torch.cat(
            (torch.arange(xyz.shape[0]).view(-1, 1), data.neighbors), dim=1)
        k = nn.shape[1]

        # Check for missing neighbors (indicated by -1 indices)
        n_missing = (nn < 0).sum(dim=1)
        if (n_missing > 0).any():
            sizes = k - n_missing
            nn = nn[nn >= 0]
            nn_ptr = torch.cat((torch.zeros(1), sizes.cumsum(dim=0).cpu()))
        else:
            nn = nn.flatten().cpu()
            nn_ptr = torch.arange(xyz.shape[0] + 1) * k
        nn = nn.numpy().astype('uint32')
        nn_ptr = nn_ptr.numpy().astype('uint32')

        # Make sure array are contiguous before moving to C++
        xyz = np.ascontiguousarray(xyz)
        nn = np.ascontiguousarray(nn)
        nn_ptr = np.ascontiguousarray(nn_ptr)

        # C++ geometric features computation on CPU
        f = point_utils.compute_geometric_features(xyz, nn, nn_ptr, k_min, False)
        f = torch.from_numpy(f.astype('float32'))

        # Heuristic to increase the importance of verticality
        f[:, 3] *= 2

        # Select only required features
        mask = (
            [linearity, planarity, scattering, verticality]
            + [normal] * 3
            + [length, surface, volume])
        features.append(f[:, mask].to(data.pos.device))

    # Save all features in the Data.x attribute
    data.x = torch.cat(features, dim=1).to(data.pos.device)

    return data
