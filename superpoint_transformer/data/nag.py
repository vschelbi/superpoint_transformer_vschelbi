import torch
from typing import List
import superpoint_transformer
from superpoint_transformer.data import Data
from superpoint_transformer.utils import tensor_idx, has_duplicates
from torch_scatter import scatter_sum


class NAG:
    """Holder for a Nested Asymmetric Graph, containing a list of
    nested partitions of the same point cloud.
    """

    def __init__(self, data_list: List[Data]):
        assert len(data_list) > 0,\
            "The NAG must have at least 1 level of hierarchy. Please " \
            "provide a minimum of 1 Data object."
        self._list = data_list
        if superpoint_transformer.is_debug_enabled():
            self.debug()

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self[i]

    def get_sub_size(self, high, low=0):
        """Compute the number of points of level 'low' contained in each
        superpoint of level 'high'.

        Note: 'low=-1' is accepted when level-0 has a 'sub' attribute
        (ie level-0 points are themselves clusters of '-1' level absent
        from the NAG object).
        """
        assert -1 <= low < high < self.num_levels
        assert 0 <= low or self[0].is_super

        # Sizes are computed in a bottom-up fashion. Note this scatter
        # operation assumes all levels of hierarchy use dense,
        # consecutive indices which are consistent between levels
        sub_size = self[low + 1].sub.size
        for i in range(low + 1, high):
            sub_size = scatter_sum(sub_size, self[i].super_index, dim=0)
        return sub_size

    @property
    def num_levels(self):
        """Number of levels of hierarchy in the nested graph."""
        return len(self)

    @property
    def num_points(self):
        """Number of points/nodes in the lower-level graph."""
        return [d.num_points for d in self] if len(self) > 0 else 0

    def to_list(self):
        """Return the Data list"""
        return self._list

    def clone(self):
        """Return a new NAG instance containing the Data clones."""
        return self.__class__([d.clone() for d in self])

    def to(self, device):
        """Move the NAG with all Data in it to device."""
        self._list = [d.to(device) for d in self]
        return self

    def cpu(self):
        """Move the NAG with all Data in it to CPU."""
        return self.to('cpu')

    def cuda(self):
        """Move the NAG with all Data in it to CUDA."""
        return self.to('cuda')

    @property
    def device(self):
        """Return device of first Data in NAG."""
        return self[0].device if len(self) > 0 else torch.Tensor().device

    @property
    def is_cuda(self):
        """Return True is one of the Data contains a CUDA Tensor."""
        for d in self:
            if isinstance(d, torch.Tensor) and d.is_cuda:
                return True
        return False

    def __getitem__(self, i_level):
        """Return a Data object from the hierarchy.

        Parameters
        ----------
        i_level: int
            The hierarchy level to return
        """
        return self._list[i_level]

    def select(self, i_level, idx):
        """Indexing mechanism on the NAG.

        Returns a new copy of the indexed NAG, with updated clusters.
        Supports int, torch and numpy indexing.

        Contrary to indexing 'Data' objects in isolation, this will
        maintain cluster indices compatibility across all levels of the
        hierarchy.

        Note that cluster indices in 'idx' must be duplicate-free.
        Indeed, duplicates would create ambiguous situations or lower
        and higher hierarchy level updates.

        Parameters
        ----------
        i_level: int
            The hierarchy level to index from.
        idx: int, np.NDArray, torch.Tensor
            Index to select nodes of the chosen hierarchy. Must be
            duplicate-free
        """
        assert isinstance(i_level, int)
        assert i_level < len(self)

        # Convert idx to a Tensor
        idx = tensor_idx(idx).to(self.device)

        # Make sure idx contains no duplicate entries
        #TODO: calling this whenever we select points might be costly, is
        # there a workaround ?
        assert not has_duplicates(idx), \
            "Duplicate indices are not supported. This would cause " \
            "ambiguities in edges and super- and sub- indices."

        # Prepare the output Data list
        data_list = [None] * self.num_levels

        # Select the nodes at level 'i_level' and update edges, subpoint
        # and superpoint indices accordingly. The returned 'out_sub' and
        # 'out_super' will help us update the lower and higher hierarchy
        # levels iteratively
        data_list[i_level], out_sub, out_super = self[i_level].select(
            idx, update_sub=True, update_super=True)

        # Iteratively update lower hierarchy levels
        for i in range(i_level - 1, -1, -1):

            # Unpack the 'out_sub' from the previous above level
            (idx_sub, sub_super) = out_sub

            # Select points but do not update 'super_index', it will be
            # directly provided by the above-level's 'sub_super'
            data_list[i], out_sub, _ = self[i].select(
                idx_sub, update_sub=True, update_super=False)

            # Directly update the 'super_index' using 'sub_super' from
            # the above level
            data_list[i].super_index = sub_super

        # Iteratively update higher hierarchy levels
        for i in range(i_level + 1, self.num_levels):

            # Unpack the 'out_super' from the previous below level
            (idx_super, super_sub) = out_super

            # Select points but do not update 'sub', it will be directly
            # provided by the above-level's 'super_sub'
            data_list[i], _, out_super = self[i].select(
                idx_super, update_sub=False, update_super=True)

            # Directly update the 'sub' using 'super_sub' from the above
            # level
            data_list[i].sub = super_sub

        # Create a new NAG with the list of indexed Data
        nag = self.__class__(data_list)

        return nag

    def debug(self):
        """Sanity checks."""
        assert len(self) > 0
        for i, d in enumerate(self):
            assert isinstance(d, Data)
            if i > 0:
                assert d.is_super
                assert d.num_points == self[i - 1].num_super
            if i < len(self) - 1:
                assert d.is_sub
                assert d.num_points == self[i + 1].num_sub

    def __repr__(self):
        info = [
            f"{key}={getattr(self, key)}"
            for key in ['num_levels', 'num_points', 'device']]
        return f"{self.__class__.__name__}({', '.join(info)})"
