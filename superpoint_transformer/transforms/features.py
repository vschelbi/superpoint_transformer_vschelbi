import torch
import numpy as np
import superpoint_transformer.partition.utils.libpoint_utils as point_utils


def compute_pointfeatures(
        data, pos=True, radius=5, rgb=True, linearity=True, planarity=True,
        scattering=True, verticality=True, normal=True, length=False,
        surface=False, volume=False):
    """ Compute the pointwise features that will be used for the
    partition.

    All local geometric features assume the input ``Data`` has a
    ``neighbors`` attribute, holding a ``(num_nodes, k)`` tensor of
    indices. All k neighbors will be used for local geometric features
    computation.

    Parameters
    ----------
    pos: bool
        Use point position.
    radius: bool
        Radius used to scale the point position features, to mitigate
        the maximum superpoint size.
    rgb: bool
        Use rgb color. Assumes Data.rgb holds either [0, 1] floats or
        [0, 255] integers
    linearity: bool
        Use local linearity. Assumes ``Data.neighbors``.
    planarity: bool
        Use local lanarity. Assumes ``Data.neighbors``.
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
    """
    features = []

    # Add xyz normalized. The scaling factor drives the maximum cluster
    # size the partition may produce
    if 'pos' in data.keys:
        features.append(data.pos / radius)

    # Add rgb to the features. If colors are stored in int, we assume
    # they are encoded in  [0, 255] and normalize them. Otherwise, we
    # assume they have already been [0, 1] normalized
    if rgb:
        f = data.rgb
        if f.type in [torch.uint8, torch.int, torch.long]:
            f = f.float() / 255
        features.append(f)

    # Add local geometric features
    if any((linearity, planarity, scattering, verticality, normal)):
        # Prepare data for numpy boost interface
        xyz = data.pos.cpu().numpy()
        nn = data.neighbors.flatten().cpu().numpy().astype(
            'uint32')  # !!!! IMPORTANT CAREFUL WITH UINT32 = 4 BILLION points MAXIMUM !!!!
        k = data.neighbors.shape[1]
        nn_ptr = np.arange(xyz.shape[0] + 1).astype(
            'uint32') * k  # !!!! IMPORTANT CAREFUL WITH UINT32 = 4 BILLION points MAXIMUM !!!!

        # C++ geometric features computation on CPU
        f = point_utils.compute_geometric_features(xyz, nn, nn_ptr, False)
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