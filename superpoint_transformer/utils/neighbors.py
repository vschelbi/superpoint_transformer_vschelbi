import torch
from superpoint_transformer.partition.FRNN import frnn


__all__ = ['knn']


def knn(x_search, x_query, k, r_max=1):
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
