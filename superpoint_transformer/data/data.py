import torch
import copy
from torch_geometric.data import Data as PyGData
from torch_geometric.data import Batch as PyGBatch
from superpoint_transformer.data.cluster import Cluster
from superpoint_transformer.utils import tensor_idx, is_dense, has_duplicates
from torch_geometric.nn.pool.consecutive import consecutive_cluster


class Data(PyGData):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.debug()

    @property
    def neighbors(self):
        return self['neighbors'] if 'neighbors' in self._store else None

    @property
    def sub(self):
        """Cluster object indicating subpoint indices for each point."""
        return self['sub'] if 'sub' in self._store else None

    @property
    def super_index(self):
        """Index of the superpoint each point belongs to."""
        return self['super_index'] if 'super_index' in self._store else None

    @property
    def is_super(self):
        """Whether the points are superpoints for a denser sub-graph."""
        return self.sub is not None

    @property
    def is_sub(self):
        """Whether the points belong to a coarser super-graph."""
        return self.super_index is not None

    @property
    def has_neighbors(self):
        """Whether the points have neighbors."""
        return self.neighbors is not None

    @property
    def has_edges(self):
        """Whether the points have edges."""
        return self.edge_index is not None

    @property
    def num_points(self):
        return self.num_nodes

    @property
    def num_super(self):
        return self.super_index.max().cpu().item() + 1 if self.is_sub else 0

    @property
    def num_sub(self):
        return self.sub.points.max().cpu().item() + 1 if self.is_super else 0

    @property
    def device(self):
        """Device of the first-encountered tensor in 'self'."""
        for key, item in self:
            if torch.is_tensor(item):
                return item.device
        return torch.Tensor().device

    def debug(self):
        """Sanity checks."""
        if self.is_super:
            assert isinstance(self.sub, Cluster), \
                "Clusters must be expressed using a Cluster object"
            assert self.y is None or self.y.dim() == 2, \
                "Clusters must hold label histograms"
        if self.is_sub:
            assert is_dense(self.super_index), \
                "Point-to-cluster indices must be dense (ie all indices in " \
                "[0, super_index.max()] must be used"

    def __inc__(self, key, value, *args, **kwargs):
        """Extend the PyG.Data.__inc__ behavior on '*index*' and
        '*face*' attributes to our 'super_index'. This is needed for
        maintaining clusters when batching Data objects together.
        """
        return self.num_super if key in 'super_index' \
            else super().__inc__(key, value)

    def select(self, idx, update_sub=True, update_super=True):
        """Returns a new Data with updated clusters, which indexes
        `self` using entries in `idx`. Supports torch and numpy fancy
        indexing. `idx` must not contain duplicate entries, as this
        would cause ambiguities in edges and super- and sub- indices.

        This operations breaks neighborhoods, so if 'self.has_neighbors'
        the output Data will not.

        NB: if `self` belongs to a NAG, calling this function in
        isolation may break compatibility with point and cluster indices
        in the other hierarchy levels. If consistency matters, prefer
        using NAG indexing instead.

        :parameter
        idx: int or 1D torch.LongTensor or numpy.NDArray
            Data indices to select from 'self'. Must NOT contain
            duplicates
        update_sub: bool
            If True, the point (ie subpoint) indices will also be
            updated to maintain dense indices. The output will then
            contain '(idx_sub, sub_super)' which can help apply these
            changes to maintain consistency with lower hierarchy levels
            of a NAG.
        update_super: bool
            If True, the cluster (ie superpoint) indices will also be
            updated to maintain dense indices. The output will then
            contain '(idx_super, super_sub)' which can help apply these
            changes to maintain consistency with higher hierarchy levels
            of a NAG.

        :returns data, (idx_sub, sub_super), (idx_super, super_sub)
        data: Data
            indexed data
        idx_sub: torch.LongTensor
            to be used with 'Data.select()' on the sub-level
        sub_super: torch.LongTensor
            to replace 'Data.super_index' on the sub-level
        idx_super: torch.LongTensor
            to be used with 'Data.select()' on the super-level
        super_sub: Cluster
            to replace 'Data.sub' on the super-level
        """
        device = self.device

        # Convert idx to a torch.LongTensor
        idx = tensor_idx(idx).to(device)

        # Make sure idx contains no duplicate entries
        #TODO: calling this whenever we select points might be costly, is
        # there a workaround ?
        assert not has_duplicates(idx), \
            "Duplicate indices are not supported. This would cause " \
            "ambiguities in edges and super- and sub- indices."

        # Output Data will not share memory with input Data
        data = self.__class__(num_nodes=idx.shape[0])

        # If Data contains edges, we will want to update edge indices
        # and attributes with respect to the new point order. Edge
        # indices are updated here, so as to compute 'idx_edge', which
        # will be used to select edge attributes
        if self.has_edges:
            # To update edge indices, create a 'reindex' tensor so that
            # the desired output can be computed with simple indexation
            # 'reindex[edge_index]'. This avoids using map() or
            # numpy.vectorize alternatives.
            reindex = torch.full(
                (self.num_nodes,), -1, dtype=torch.int64, device=device)
            reindex = reindex.scatter_(
                0, idx, torch.arange(idx.shape[0], device=device))
            edge_index = reindex[self.edge_index]

            # Remove the obsolete edges (ie those involving a '-1' index)
            idx_edge = torch.where((edge_index != -1).all(dim=0))[0]
            data.edge_index = edge_index[:, idx_edge]

        # Selecting points may affect their order, if we need to
        # preserve subpoint consistency, we need to update the
        # 'Data.sub' of the current level and the 'Data.super_index'
        # of the level below
        out_sub = (None, None)
        if self.is_super:
            data.sub, out_sub = self.sub.select(idx, update_sub=update_sub)

        # Selecting points may affect their order, if we need to
        # preserve superpoint consistency, we need to update the
        # 'Data.super_index' of the current level along with the
        # 'Data.sub' of the level above
        out_super = (None, None)
        if self.is_sub:
            data.super_index = self.super_index[idx].clone()

        if self.is_sub and update_super:
            # Convert superpoint indices, in case some superpoints have
            # disappeared. 'idx_super' is intended to be used with
            # Data.select() on the level above
            data.super_index, perm = consecutive_cluster(data.super_index)
            idx_super = data.super_index[perm]

            # Selecting the superpoints with 'idx_super' will not be
            # enough to maintain consistency with the current points. We
            # also need to update the super-level's 'Data.sub', which
            # can be computed from 'super_index'
            super_sub = Cluster(
                data.super_index, torch.arange(data.num_nodes, device=device),
                dense=True)

            out_super = (idx_super, super_sub)

        # Index data items depending on their type
        skip_keys = ['edge_index', 'sub', 'super_index', 'neighbors']
        for key, item in self:

            # 'skip_keys' have already been dealt with earlier on, so we
            # can skip them here
            if key in skip_keys:
                continue

            is_tensor = torch.is_tensor(item)
            is_node_size = item.size(0) == self.num_nodes
            is_edge_size = item.size(0) == self.num_edges

            # Slice tensor elements containing num_edges elements. Note
            # we deal with edges first, to rule out the case where
            # num_edges = num_nodes.
            if self.has_edges and is_tensor and is_edge_size and 'edge' in key:
                data[key] = item[idx_edge].clone()

            # Slice other tensor elements containing num_nodes elements
            elif is_tensor and is_node_size:
                data[key] = item[idx].clone()

            # Other Data attributes are simply copied
            else:
                data[key] = copy.deepcopy(item)

        return data, out_sub, out_super





    #TODO
    # def GROUP SIZE !!!***************




class Batch(PyGBatch):
    pass
    #TODO
    # batching data.sub indices
