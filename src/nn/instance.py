from torch import nn
from src.utils.instance import instance_cut_pursuit


__all__ = ['InstancePartitioner']


class InstancePartitioner(nn.Module):
    """Partition a graph into instances using cut-pursuit.
    More specifically, this step will group nodes together based on:
        - node offset position
        - node predicted classification logits
        - node size
        - edge affinity

    NB: This operation relies on the parallel cut-pursuit algorithm:
        https://gitlab.com/1a7r0ch3/parallel-cut-pursuit
        Currently, this implementation is non-differentiable and runs on
        CPU.

    :param regularization: float
        Regularization parameter for the partition
    :param spatial_weight: float
        Weight used to mitigate the impact of the point position in the
        partition. The larger, the less spatial coordinates matter. This
        can be loosely interpreted as the inverse of a maximum
        superpoint radius
    :param cutoff: float
        Minimum number of points in each cluster
    :param parallel: bool
        Whether cut-pursuit should run in parallel
    :param iterations: int
        Maximum number of iterations for each partition
    :param trim: bool
        Whether the input graph should be trimmed. See `to_trimmed()`
        documentation for more details on this operation
    :param discrepancy_epsilon: float
        Mitigates the maximum discrepancy. More precisely:
        `affinity=0 ⇒ discrepancy=discrepancy_epsilon`
    :return:
    """

    def __init__(
            self,
            regularization=1e-2,
            spatial_weight=1,
            cutoff=1,
            parallel=True,
            iterations=10,
            trim=False,
            discrepancy_epsilon=1e-3):
        super().__init__()
        self.regularization = regularization
        self.spatial_weight = spatial_weight
        self.cutoff = cutoff
        self.parallel = parallel
        self.iterations = iterations
        self.trim = trim
        self.discrepancy_epsilon = discrepancy_epsilon

    def forward(
            self,
            node_pos,
            node_offset,
            node_logits,
            node_size,
            edge_index,
            edge_affinity_logits):
        """

        :param node_pos: Tensor of shape [num_nodes, num_dim]
            Node positions
        :param node_offset: Tensor of shape [num_nodes, num_dim]
            Predicted instance centroid offset for each node
        :param node_logits: Tensor of shape [num_nodes, num_classes]
            Predicted classification logits for each node
        :param node_size: Tensor of shape [num_nodes]
            Size of each node
        :param edge_index: Tensor of shape [2, num_edges]
            Edges of the graph, in torch-geometric's format
        :param edge_affinity_logits: Tensor of shape [num_edges]
            Predicted affinity logits (ie in R+, before sigmoid) of each
            edge

        :return: instance_index: Tensor of shape [num_nodes]
            Indicates which predicted instance each node belongs to
        """
        return instance_cut_pursuit(
            node_pos,
            node_offset,
            node_logits,
            node_size,
            edge_index,
            edge_affinity_logits,
            do_sigmoid_affinity=True,
            regularization=self.regularization,
            spatial_weight=self.spatial_weight,
            cutoff=self.cutoff,
            parallel=self.parallel,
            iterations=self.iterations,
            trim=self.trim,
            discrepancy_epsilon=self.discrepancy_epsilon)
