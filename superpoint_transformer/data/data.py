import torch
import copy
from torch_geometric.data import Data as PyGData
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.utils import coalesce, remove_self_loops
import superpoint_transformer
from superpoint_transformer.data.cluster import Cluster
from superpoint_transformer.utils import tensor_idx, is_dense, has_duplicates
from superpoint_transformer.utils.neighbors import knn


class Data(PyGData):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if superpoint_transformer.is_debug_enabled():
            self.debug()

    @property
    def pos(self):
        return self['pos'] if 'pos' in self._store else None

    @property
    def rgb(self):
        return self['rgb'] if 'rgb' in self._store else None

    @property
    def pred(self):
        return self['pred'] if 'pred' in self._store else None

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
    def num_edges(self):
        """Overwrite the torch_geometric initial definition, which
        somehow returns incorrect results, like:
            data.num_edges != data.edge_index.shape[1]
        """
        return self.edge_index.shape[1] if self.has_edges else 0

    @property
    def num_points(self):
        return self.num_nodes

    @property
    def num_super(self):
        return self.super_index.max().cpu().item() + 1 if self.is_sub else 0

    @property
    def num_sub(self):
        return self.sub.points.max().cpu().item() + 1 if self.is_super else 0

    def to(self, device):
        """Extend torch_geometric.Data.to to handle Cluster attributes.
        """
        self = super().to(device)
        if self.is_super:
            self.sub = self.sub.to(device)
        return self

    def cpu(self):
        """Move the NAG with all Data in it to CPU."""
        return self.to('cpu')

    def cuda(self):
        """Move the NAG with all Data in it to CUDA."""
        return self.to('cuda')

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

        # Output Data will not share memory with input Data.
        # NB: it is generally not recommended to instantiate en empty
        # Data like this, as it might cause issues when calling
        # 'data.num_nodes' later on. Need to be careful when calling
        # 'data.num_nodes' before having set any of the pointwise
        # attributes (eg 'x', 'pos', 'rgb', 'y', etc)
        data = self.__class__()

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

            # Remove obsolete edges (ie those involving a '-1' index)
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
            new_super_index, perm = consecutive_cluster(data.super_index)
            idx_super = data.super_index[perm]
            data.super_index = new_super_index

            # Selecting the superpoints with 'idx_super' will not be
            # enough to maintain consistency with the current points. We
            # also need to update the super-level's 'Data.sub', which
            # can be computed from 'super_index'
            super_sub = Cluster(
                data.super_index, torch.arange(idx.shape[0], device=device),
                dense=True)

            out_super = (idx_super, super_sub)

        # Index data items depending on their type
        warn_keys = ['neighbors', 'distances']
        skip_keys = ['edge_index', 'sub', 'super_index'] + warn_keys
        for key, item in self:

            # 'skip_keys' have already been dealt with earlier on, so we
            # can skip them here
            if key in warn_keys and superpoint_transformer.is_debug_enabled():
                print(
                    f"WARNING: Data.select does not support '{key}', this "
                    f"attribute will be absent from the output")
            if key in skip_keys:
                continue

            is_tensor = torch.is_tensor(item)
            is_node_size = item.size(0) == self.num_nodes
            is_edge_size = item.size(0) == self.num_edges

            # Slice tensor elements containing num_edges elements. Note
            # we deal with edges first, to rule out the case where
            # num_edges = num_nodes
            if self.has_edges and is_tensor and is_edge_size and 'edge' in key:
                data[key] = item[idx_edge].clone()

            # Slice other tensor elements containing num_nodes elements
            elif is_tensor and is_node_size:
                data[key] = item[idx].clone()

            # Other Data attributes are simply copied
            else:
                data[key] = copy.deepcopy(item)

        # Security just in case no node-level attribute was passed, Data
        # will not be able to properly infer its number of nodes
        if data.num_nodes != idx.shape[0]:
            data.num_nodes = idx.shape[0]

        return data, out_sub, out_super

    def is_isolated(self):
        """If self.has_edges, returns a boolean tensor of size
        self.num_nodes indicating which are absent from self.edge_index.
        Will raise an error if self.has_edges is False.
        """
        assert self.has_edges
        mask = torch.ones(self.num_nodes, dtype=torch.bool, device=self.device)
        mask[self.edge_index.unique()] = False
        return mask

    # def absorb(self, data, idx):
    #     """Integrate the points of another data object into self, based
    #     on the provided indices. This operation requires data to have
    #     the same attributes as self (any attribute of data absent from
    #     self will be silently discarded). Note that this operation
    #     assumes data.sub
    #     """
    #     # Sanity checks
    #     idx = tensor_idx(idx)
    #     assert isinstance(data, Data)
    #     assert data.num_points == idx.numel()
    #     assert idx.max() < self.num_points
    #     missing = [k for k in self.keys if k not in data.keys]
    #     if len(missing) > 0:
    #         raise ValueError(
    #             f"Keys in 'self' and 'data' do not match. Missing keys: "
    #             f"{missing}")
    #
    #     #TODO: DO NOT fuse attributes, only super and sub... ?

    def connect_isolated(self, k=1):
        """Search for nodes with no edges in the graph and connect them
        to their k nearest neighbors. Update self.edge_index and
        self.edge_attr accordingly.

        Will raise an error if self has no edges or no pos.

        Returns self updated with the newly-created edges.
        """
        assert self.has_edges
        assert self.pos is not None

        # Search for isolated nodes and exit if no node is isolated
        is_isolated = self.is_isolated()
        is_out = torch.where(is_isolated)[0]
        if not is_isolated.any():
            return self

        # Search the nearest nodes for isolated nodes, among all nodes
        # NB: we remove the nodes themselves from their own neighborhood
        high = self.pos.max(dim=0).values
        low = self.pos.min(dim=0).values
        r_max = torch.linalg.norm(high - low)
        distances, neighbors = knn(
            self.pos, self.pos[is_out], k + 1, r_max=r_max)
        distances = distances[:, 1:]
        neighbors = neighbors[:, 1:]

        # If the edges have attributes, we also create attributes for
        # the new edges. There is no trivial way of doing so, the
        # heuristic here simply attempts to linearly regress the edge
        # weights based on the corresponding node distances
        if self.edge_attr is not None:

            # Get existing edges attributes and associated distance
            w = self.edge_attr
            s = self.edge_index[0]
            t = self.edge_index[1]
            d = torch.linalg.norm(self.pos[s] - self.pos[t], dim=1)
            d_1 = torch.vstack((d, torch.ones_like(d))).T

            # Least square on d_1.x = w  (ie d.a + b = w)
            a, b = torch.linalg.lstsq(d_1, w).solution

            # Heuristic: linear approximation of w by d
            edge_attr_new = distances.flatten() * a + b

            # Append to existing self.edge_attr
            self.edge_attr = torch.cat((self.edge_attr, edge_attr_new))

        # Add new edges between the nodes
        source = is_out.repeat_interleave(k)
        target = neighbors.flatten()
        edge_index_new = torch.vstack((source, target))
        self.edge_index = torch.cat((self.edge_index, edge_index_new), dim=1)

        return self

    def clean_graph(self):
        """Remove self loops, redundant edges and undirected edges."""
        assert self.has_edges

        # Recover self edges and edge attributes
        edge_index = self.edge_index
        edge_attr = self.edge_attr if self.edge_attr is not None else None

        # Search for undirected edges, ie edges with (i,j) and (j,i)
        # both present in edge_index. Flip (j,i) into (i,j) to make them
        # redundant
        s_larger_t = edge_index[0] > edge_index[1]
        edge_index[:, s_larger_t] = edge_index[:, s_larger_t].flip(0)

        # Sort edges by row and remove duplicates
        if edge_attr is None:
            edge_index = coalesce(edge_index)
        else:
            edge_index, edge_attr = coalesce(
                edge_index, edge_attr=edge_attr, reduce='mean')

        # Remove self loops
        edge_index, edge_attr = remove_self_loops(
            edge_index, edge_attr=edge_attr)

        # Save new graph in self attributes
        self.edge_index = edge_index
        self.edge_attr = edge_attr

        return self


class Batch(PyGBatch):
    pass
    #TODO
    # batching data.sub indices
