import torch
from torch import nn
from src.utils.instance import instance_cut_pursuit
from src.utils.scatter import scatter_mean_weighted
from torch_geometric.nn.pool.consecutive import consecutive_cluster


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

    :param loss_type: str
        Rules the loss applied on the node features. Accepts one of
        'l2' (L2 loss on node features and probabilities),
        'l2_kl' (L2 loss on node features and Kullback-Leibler
        divergence on node probabilities)
    :param regularization: float
        Regularization parameter for the partition
    :param x_weight: float
        Weight used to mitigate the impact of the node position in the
        partition. The larger, the less spatial coordinates matter
    :param p_weight: float
        Weight used to mitigate the impact of the node probabilities in
        the partition. The larger, the greater the impact
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
        `affinity=1 ⇒ discrepancy=1/discrepancy_epsilon`
    :param temperature: float
        Temperature used in the softmax when converting node logits to
        probabilities
    :param dampening: float
        Dampening applied to the node probabilities to mitigate the
        impact of near-zero probabilities in the Kullback-Leibler
        divergence
    :return:
    """

    def __init__(
            self,
            loss_type='l2_kl',
            regularization=1e-2,
            x_weight=1,
            p_weight=1,
            cutoff=1,
            parallel=True,
            iterations=10,
            trim=False,
            discrepancy_epsilon=1e-3,
            temperature=1,
            dampening=0):
        super().__init__()
        self.loss_type = loss_type
        self.regularization = regularization
        self.x_weight = x_weight
        self.p_weight = p_weight
        self.cutoff = cutoff
        self.parallel = parallel
        self.iterations = iterations
        self.trim = trim
        self.discrepancy_epsilon = discrepancy_epsilon
        self.temperature = temperature
        self.dampening = dampening

    def forward(
            self,
            batch,
            node_x,
            node_logits,
            stuff_classes,
            node_size,
            edge_index,
            edge_affinity_logits):
        """The forward step will compute the partition on the instance
        graph, based on the node features, node logits, and edge
        affinities. The partition segments will then be further merged
        so that there is at most one instance of each stuff class per
        batch item (ie per scene).

        :param batch: Tensor of shape [num_nodes]
            Batch index of each node
        :param node_x: Tensor of shape [num_nodes, num_dim]
            Predicted node embeddings
        :param node_logits: Tensor of shape [num_nodes, num_classes]
            Predicted classification logits for each node
        :param stuff_classes: List or Tensor
            List of 'stuff' class labels. These are used for merging
            stuff segments together to ensure there is at most one
            predicted instance of each 'stuff' class per batch item
        :param node_size: Tensor of shape [num_nodes]
            Size of each node
        :param edge_index: Tensor of shape [2, num_edges]
            Edges of the graph, in torch-geometric's format
        :param edge_affinity_logits: Tensor of shape [num_edges]
            Predicted affinity logits (ie in R+, before sigmoid) of each
            edge

        :return: obj_index: Tensor of shape [num_nodes]
            Indicates which predicted instance each node belongs to
        """
        return instance_cut_pursuit(
            batch,
            node_x,
            node_logits,
            stuff_classes,
            node_size,
            edge_index,
            edge_affinity_logits,
            do_sigmoid_affinity=True,
            loss_type=self.loss_type,
            regularization=self.regularization,
            x_weight=self.x_weight,
            p_weight=self.p_weight,
            cutoff=self.cutoff,
            parallel=self.parallel,
            iterations=self.iterations,
            trim=self.trim,
            discrepancy_epsilon=self.discrepancy_epsilon,
            temperature=self.temperature,
            dampening=self.dampening)

    def extra_repr(self) -> str:
        keys = [
            'regularization',
            'x_weight',
            'cutoff',
            'parallel',
            'iterations',
            'trim',
            'discrepancy_epsilon']
        return ', '.join([f'{k}={getattr(self, k)}' for k in keys])
