from src.data import NAG, InstanceData
from src.transforms import Transform


__all__ = ['']


class NAGInstances(Transform):
    """Compute the instances contained in each superpoint of each level,
    provided that the first level actually has an 'obj' attribute
    holding an InstanceData`.

    :param strict: bool
        If True, will raise an exception if the level Data does not have
        instance
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG
    _OBJ_KEY = 'obj'
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
