import torch
from torch_scatter import scatter_max
from src.data import NAG, InstanceData
from src.transforms import Transform
from torch_geometric.nn.pool.consecutive import consecutive_cluster


__all__ = ['NAGPropagatePointInstances']


class NAGPropagatePointInstances(Transform):
    """Compute the instances contained in each superpoint of each level,
    provided that the first level has an 'obj' attribute holding an
    `InstanceData`.

    :param strict: bool
        If True, will raise an exception if the level Data does not have
        instance
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG
    _NO_REPR = ['strict']

    def __init__(self, strict=False):
        self.strict = strict

    def _process(self, nag):
        # Read the instances from the first data
        obj_0 = nag[0].obj
        if obj_0 is None or not isinstance(obj_0, InstanceData):
            if not self.strict:
                return nag
            raise ValueError(f"Could not find any InstanceData in `nag[0].obj`")

        for i_level in range(1, nag.num_levels):
            super_index = nag.get_super_index(i_level)
            nag[i_level].obj = obj_0.merge(super_index)

        return nag


class OnTheFlyInstanceGraph(Transform):
    """Compute the non-oriented graph used for instance and panoptic
    segmentation.
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG
    _MODES = ['adjacency', 'radius']

    def __init__(self, level=1, mode='adjacency'):
        assert mode in self._MODES, f"Expected 'mode' to be one of {self._MODES}"
        self.level = level
        self.mode = mode

    def _process(self, nag):
        data = nag[self.level]

        if data.obj is None:
            raise ValueError(f"Expected `Data.obj` to contain an InstanceData")

        # Build the edges on which the graph optimization will be run
        # for instance or panoptic segmentation
        if self.mode == 'adjacency':
            data.obj_edge_index = data.edge_index
        elif self.mode == 'radius':
            data.obj_edge_index = ...
        else:
            raise NotImplementedError

        # If, for some reason, the graph is None, we convert it to an
        # empty torch_geometric-friendly `edge_index`-like format
        empty_graph = torch.empty(2, 0, dtype=torch.long, device=data.device)
        data.obj_edge_index = data.obj_edge_index or empty_graph

        # Compute the superpoint target instance centroid position
        # NB: this is a proxy method assuming NAG[0] is pure-enough
        obj_pos, obj_idx = nag[0].obj.estimate_centroid(nag[0].pos)

        # Find the dominant target instance, for each node
        argmax = scatter_max(data.obj.count, data.obj.indices)[1]
        node_obj_idx = data.obj[argmax]

        # Distribute, for each node, the dominant object position
        obj_idx, perm = consecutive_cluster(obj_idx)

        # TODO: assign the obj centroid to each sp


        return nag
