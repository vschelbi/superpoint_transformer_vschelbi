import torch
import superpoint_transformer
from superpoint_transformer.data.csr import CSRData
from superpoint_transformer.utils import has_duplicates
from torch_geometric.nn.pool.consecutive import consecutive_cluster


class Cluster(CSRData):
    """Child class of CSRData to simplify some common operations
    dedicated to cluster-point indexing.
    """

    def __init__(self, pointers, points, dense=False):
        super().__init__(
            pointers, points, dense=dense, is_index_value=[True])

    @property
    def points(self):
        return self.values[0]

    @points.setter
    def points(self, points):
        assert points.device == self.device, \
            f"Points is on {points.device} while self is on {self.device}"
        self.values[0] = points
        if superpoint_transformer.is_debug_enabled():
            self.debug()

    @property
    def num_clusters(self):
        return self.num_groups

    @property
    def num_points(self):
        return self.num_items

    def to_super_index(self):
        """Return a 1D tensor of indices converting the CSR-formatted
        clustering structure in 'self' into the 'super_index' format.
        """
        # TODO: this assumes 'self.point' is a permutation, shall we
        #  check this (although it requires sorting) ?
        device = self.device
        out = torch.empty((self.num_items,), dtype=torch.long, device=device)
        cluster_idx = torch.arange(self.num_groups, device=device)
        out[self.points] = cluster_idx.repeat_interleave(self.size)
        return out

    def select(self, idx, update_sub=True):
        """Returns a new Cluster with updated clusters and points, which
        indexes `self` using entries in `idx`. Supports torch and numpy
        fancy indexing. `idx` must NOT contain duplicate entries, as
        this would cause ambiguities in super- and sub- indices.

        NB: if `self` belongs to a NAG, calling this function in
        isolation may break compatibility with point and cluster indices
        in the other hierarchy levels. If consistency matters, prefer
        using NAG indexing instead.

        :parameter
        idx: int or 1D torch.LongTensor or numpy.NDArray
            Cluster indices to select from 'self'. Must NOT contain
            duplicates
        update_sub: bool
            If True, the point (ie subpoint) indices will also be
            updated to maintain dense indices. The output will then
            contain '(idx_sub, sub_super)' which can help apply these
            changes to maintain consistency with lower hierarchy levels
            of a NAG.

        :returns cluster, (idx_sub, sub_super)
        clusters: Cluster
            indexed cluster
        idx_sub: torch.LongTensor
            to be used with 'Data.select()' on the sub-level
        sub_super: torch.LongTensor
            to replace 'Data.super_index' on the sub-level
        """
        # Normal CSRData indexing, creates a new object in memory
        cluster = self[idx]

        if not update_sub:
            return cluster, (None, None)

        # Convert subpoint indices, in case some subpoints have
        # disappeared. 'idx_sub' is intended to be used with
        # Data.select() on the level below
        new_cluster_points, perm = consecutive_cluster(cluster.points)
        idx_sub = cluster.points[perm]
        cluster.points = new_cluster_points

        # Selecting the subpoints with 'idx_sub' will not be
        # enough to maintain consistency with the current points. We
        # also need to update the sub-level's 'Data.super_index', which
        # can be computed from 'cluster'
        sub_super = cluster.to_super_index()

        return cluster, (idx_sub, sub_super)

    def debug(self):
        super().debug()
        # TODO: calling has_duplicates whenever we debug might be costly...
        assert not has_duplicates(self.points)

    def __repr__(self):
        info = [
            f"{key}={getattr(self, key)}"
            for key in ['num_clusters', 'num_points', 'device']]
        return f"{self.__class__.__name__}({', '.join(info)})"
