import h5py
import torch
from typing import List
import superpoint_transformer
from superpoint_transformer.data import Data, Cluster
from superpoint_transformer.utils import tensor_idx, has_duplicates, numpyfy, \
    select_hdf5_data
from torch_geometric.nn.pool.consecutive import consecutive_cluster
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
        assert 0 <= i_level < len(self)
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
        if superpoint_transformer.is_debug_enabled():
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

    def save(self, path, x32=True):
        """Save NAG to HDF5 file.

        :param path:
        :param x32:
        :return:
        """
        with h5py.File(path, 'w') as f:
            for i_level, data in enumerate(self):
                group = f.create_group(f'partition_{i_level}')
                for k, val in data.items():
                    if isinstance(val, torch.Tensor):
                        val = numpyfy(val, x32=x32)
                        group.create_dataset(k, data=val, dtype=val.dtype)
                    elif isinstance(val, Cluster):
                        p = numpyfy(val.pointers, x32=x32)
                        group.create_dataset('sub_pointers', data=p, dtype=p.dtype)
                        p = numpyfy(val.points, x32=x32)
                        group.create_dataset('sub_points', data=p, dtype=p.dtype)
                    else:
                        raise NotImplementedError(f'Unsupported type={type(val)}')

    @staticmethod
    def load(
            path, low=0, high=-1, idx=None, idx_keys=None, keys=None,
            update_super=True, update_sub=False):
        """Load NAG from an HDF5 file. See `NAG.save` for writing such
        file. Options allow reading only part of the data.

        :param path: path the file
        :param low: lowest partition level to read
        :param high: highest partition level to read
        :param idx: index, slice, mask to select from level_min
        :param idx_keys: keys on which the indexing should be applied
        :param keys: keys to load. If None, all keys will be loaded
        update_sub: bool
            If True, the point (ie subpoint) indices will also be
            updated to maintain dense indices. The output will then
            contain '(idx_sub, sub_super)' which can help apply these
            changes to maintain consistency with lower hierarchy levels
            of a NAG. See Data.select
        :param update_super: If True, the cluster (ie superpoint)
            indices will also be updated to maintain dense indices. The
            output will then contain '(idx_super, super_sub)' which can
            help apply these changes to maintain consistency with higher
            hierarchy levels of a NAG. See Data.select
        :return:
        """
        idx_keys = Data._INDEXABLE if idx_keys is None else idx_keys
        keys = Data._READABLE if keys is None else keys
        data_list = []
        with h5py.File(path, 'r') as f:

            # Initialize partition levels min and max to read from the
            # file. This functionality is especially intended for
            # loading levels 1 and above when we want to avoid loading
            # the memory-costly level-0 points
            low = max(low, 0)
            high = len(f) - 1 if high < 0 else min(high, len(f) - 1)

            # Make sure all required partitions are present in the file
            assert all([
                f'partition_{k}' in f.keys()
                for k in range(low, high + 1)])

            for i in range(low, high + 1):

                # Apply index selection on the low only, if required.
                # For all subsequent levels, only keys selection is
                # available
                if i == low:
                    kwargs = select_hdf5_data(
                        f[f'partition_{i}'], keys=keys, idx=idx,
                        idx_keys=idx_keys)
                else:
                    kwargs = select_hdf5_data(
                        f[f'partition_{i}'], keys=keys)

                # Special treatment is required to gather the
                # 'sub_pointers' and 'sub_points' and convert them into
                # a Cluster object
                pointers = kwargs.pop('sub_pointers', None)
                points = kwargs.pop('sub_points', None)
                if points is not None and pointers is not None:
                    sub = Cluster(pointers, points)
                    if i == low and idx is not None:
                        sub = sub.select(idx, update_sub=update_sub)
                    kwargs['sub'] = sub

                data_list.append(Data(**kwargs))

        if idx is not None:
            return NAG(data_list)

        # In case the lowest level was indexed, we need to update the
        # above level too. Unfortunately, this is probably because we do
        # not want to load the whole low-level partition, so we
        # artificially create a Data object to simulate it, just to be
        # able to leverage the convenient NAG.select method.
        # NB: this may be a little slow for the CPU-based DataLoader
        # operations at train time, so we will prefer setting
        # update_super=False in this situation and do the necessary
        # later on on GPU
        if update_super:
            return NAG.cat_select(data_list[0], data_list[1:], idx=idx)

        return data_list, idx

    @staticmethod
    def cat_select(data, data_list, idx=None):
        """Does part of what Data.select does but in an ugly way. This
        is mostly intended for the DataLoader to be able to load NAG and
        sample level-0 points on CPU in reasonable time and finish the
        update_sub, update_super work on GPU later on if need be...

        :param data: Data object for level-0 points
        :param data_list: list of Data objects for level-1+ points
        :param idx: optional, indexing that has been applied on level-0
            data and guides higher levels updating (see NAG.select and
            Data.select with update_super=True)
        :return:
        """
        if idx is None and data_list is None or len(data_list) == 0:
            return NAG([data])

        if idx is None:
            return NAG([data] + data_list)

        if data_list is None or len(data_list) == 0:
            data.super_index = consecutive_cluster(data.super_index)[0]
            return NAG([data])

        data_fake = Data(super_index=data_list[0].sub.to_super_index())
        nag = NAG([data_fake] + data_list)
        nag = nag.select(0, idx)
        data.super_index = nag[0].super_index
        nag._list[0] = data

        return NAG(data_list)

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
