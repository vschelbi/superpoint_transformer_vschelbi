import torch
import numpy as np


__all__ = ['cross_product_matrix', 'rodrigues_rotation_matrix']


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
