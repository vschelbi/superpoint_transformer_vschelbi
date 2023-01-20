import torch
import numpy as np


__all__ = [
    'cross_product_matrix', 'rodrigues_rotation_matrix', 'base_vectors_3d']


def cross_product_matrix(k):
    """Compute the cross-product matrix of a vector k.

    Credit: https://github.com/torch-points3d/torch-points3d
    """
    return torch.tensor(
        [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], device=k.device)


def rodrigues_rotation_matrix(axis, theta_degrees):
    """Given an axis and a rotation angle, compute the rotation matrix
    using the Rodrigues formula.

    Source : https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    Credit: https://github.com/torch-points3d/torch-points3d
    """
    axis = axis / axis.norm()
    K = cross_product_matrix(axis)
    t = torch.tensor([theta_degrees / 180. * np.pi], device=axis.device)
    R = torch.eye(3, device=axis.device) \
        + torch.sin(t) * K + (1 - torch.cos(t)) * K.mm(K)
    return R


def base_vectors_3d(x):
    """Compute orthonormal bases for a set of 3D vectors. The 1st base
    vector is the normalized input vector, while the 2nd and 3rd vectors
    are constructed in the corresponding orthogonal plane. Note that
    this problem is underconstrained and, as such, any rotation of the
    output base around the 1st vector is a valid orthonormal base.
    """
    assert x.dim() == 2
    assert x.shape[1] == 3

    a = x / torch.linalg.norm(x, dim=1).view(-1, 1)

    b = torch.vstack((a[:, 1] - a[:, 2], a[:, 2] - a[:, 0], a[:, 0] - a[:, 1])).T
    b /= torch.linalg.norm(b, dim=1).view(-1, 1)

    c = torch.linalg.cross(a, b)

    return torch.cat((a.unsqueeze(1), b.unsqueeze(1), c.unsqueeze(1)), dim=1)
