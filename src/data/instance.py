import h5py
import torch
from time import time
from src.data.csr import CSRData, CSRBatch
from src.utils import tensor_idx, is_dense, save_tensor, load_tensor
from torch_scatter import scatter_max, scatter_sum
from torch_geometric.nn.pool.consecutive import consecutive_cluster


__all__ = ['InstanceData', 'InstanceBatch']


class InstanceData(CSRData):
    """Child class of CSRData to simplify some common operations
    dedicated to instance labels clustering.
    """

    def __init__(self, pointers, obj, count, dense=False, **kwargs):
        super().__init__(
            pointers, obj, count, dense=dense, is_index_value=None)

    @staticmethod
    def get_batch_type():
        """Required by CSRBatch.from_csr_list."""
        return InstanceBatch

    @property
    def obj(self):
        return self.values[0]

    @obj.setter
    def obj(self, obj):
        assert obj.device == self.device, \
            f"obj is on {obj.device} while self is on {self.device}"
        self.values[0] = obj
        # if src.is_debug_enabled():
        #     self.debug()

    @property
    def count(self):
        return self.values[1]

    @count.setter
    def count(self, count):
        assert count.device == self.device, \
            f"count is on {count.device} while self is on {self.device}"
        self.values[1] = count
        # if src.is_debug_enabled():
        #     self.debug()

    @property
    def num_clusters(self):
        return self.num_groups

    @property
    def num_overlaps(self):
        return self.num_items

    @property
    def num_obj(self):
        return self.obj.unique().numel()

    @property
    def major(self):
        """Return the obj and count of the majority (ie most frequent)
        instance in each cluster.
        """
        # Compute the cluster index for each overlap (ie each row in
        # self.values)
        cluster_idx = self.indices

        # Search for the obj with the largest count, for each cluster
        count, argmax = scatter_max(self.count, cluster_idx)
        obj = self.obj[argmax]

        return obj, count

    def select(self, idx):
        """Returns a new InstanceData which indexes `self` using entries
        in `idx`. Supports torch and numpy fancy indexing.

        NB: since we store global object ids in `obj`, as opposed to
        maintaining contiguous indices for the instances, we do not need
        to update the `obj` when indexing and can simply use CSRData
        indexing.

        :parameter
        idx: int or 1D torch.LongTensor or numpy.NDArray
            Cluster indices to select from 'self'. Must NOT contain
            duplicates
        """
        # Normal CSRData indexing, creates a new object in memory
        return self[idx]

    def merge(self, idx):
        """Merge clusters based on `idx` and return the result in a new
        InstanceData object.

        :param idx: 1D torch.LongTensor or numpy.NDArray
            Indices of the parent cluster each cluster should be merged
            into. Must have the same size as `self.num_clusters` and
            indices must start at 0 and be contiguous.
        """
        # Make sure the indices are dense
        idx = tensor_idx(idx)
        assert is_dense(idx), f"Expected contiguous indices in [0, max]"

        # Compute the merged cluster index for each item
        cluster_idx = self.indices
        merged_idx = idx[cluster_idx].long()

        # Build a unique merged_idx-obj indices with lexicographic order
        # placing merged_idx first and obj second. This fill facilitate
        # building the pointers downstream
        base = self.obj.long().max() + 1
        merged_obj_idx = merged_idx * base + self.obj.long()

        # Make the indices contiguous in [0, max] to alleviate
        # downstream scatter operations
        merged_obj_idx, perm = consecutive_cluster(merged_obj_idx)
        num_unique_merge_obj = perm.shape[0]

        # Compute the new counts for each obj in the merged data
        count = scatter_sum(self.count, merged_obj_idx)


        # build pointers using merged_idx bincounts (no need to reverse the consecutive_cluster)
        # return new InstanceData






    def __repr__(self):
        info = [
            f"{key}={getattr(self, key)}"
            for key in ['num_clusters', 'num_overlaps', 'num_obj', 'device']]
        return f"{self.__class__.__name__}({', '.join(info)})"

    def save(self, f, fp_dtype=torch.float):
        """Save InstanceData to HDF5 file.

        :param f: h5 file path of h5py.File or h5py.Group
        :param fp_dtype: torch dtype
            Data type to which floating point tensors will be cast
            before saving
        :return:
        """
        if not isinstance(f, (h5py.File, h5py.Group)):
            with h5py.File(f, 'w') as file:
                self.save(file, fp_dtype=fp_dtype)
            return

        save_tensor(self.pointers, f, 'pointers', fp_dtype=fp_dtype)
        save_tensor(self.obj, f, 'obj', fp_dtype=fp_dtype)
        save_tensor(self.count, f, 'count', fp_dtype=fp_dtype)

    @staticmethod
    def load(f, idx=None, verbose=False):
        """Load InstanceData from an HDF5 file. See `InstanceData.save`
        for writing such file. Options allow reading only part of the
        clusters.

        :param f: h5 file path of h5py.File or h5py.Group
        :param idx: int, list, numpy.ndarray, torch.Tensor
            Used to select clusters when reading. Supports fancy
            indexing
        :param verbose: bool
        """
        KEYS = ['pointers', 'obj', 'count']

        if not isinstance(f, (h5py.File, h5py.Group)):
            with h5py.File(f, 'r') as file:
                out = InstanceData.load(file, idx=idx, verbose=verbose)
            return out

        assert all(k in f.keys() for k in KEYS)

        start = time()
        idx = tensor_idx(idx)
        if verbose:
            print(f'InstanceData.load tensor_idx         : {time() - start:0.5f}s')

        if idx is None or idx.shape[0] == 0:
            start = time()
            pointers = load_tensor(f['pointers'])
            obj = load_tensor(f['obj'])
            count = load_tensor(f['count'])
            if verbose:
                print(f'InstanceData.load read all           : {time() - start:0.5f}s')
            start = time()
            out = InstanceData(pointers, obj, count)
            if verbose:
                print(f'InstanceData.load init               : {time() - start:0.5f}s')
            return out

        # Read only pointers start and end indices based on idx
        start = time()
        ptr_start = load_tensor(f['pointers'], idx=idx)
        ptr_end = load_tensor(f['pointers'], idx=idx + 1)
        if verbose:
            print(f'InstanceData.load read ptr       : {time() - start:0.5f}s')

        # Create the new pointers
        start = time()
        pointers = torch.cat([
            torch.zeros(1, dtype=ptr_start.dtype),
            torch.cumsum(ptr_end - ptr_start, 0)])
        if verbose:
            print(f'InstanceData.load new pointers   : {time() - start:0.5f}s')

        # Create the indexing tensor to select and order values.
        # Simply, we could have used a list of slices, but we want to
        # avoid for loops and list concatenations to benefit from torch
        # capabilities.
        start = time()
        sizes = pointers[1:] - pointers[:-1]
        val_idx = torch.arange(pointers[-1])
        val_idx -= torch.arange(pointers[-1] + 1)[
            pointers[:-1]].repeat_interleave(sizes)
        val_idx += ptr_start.repeat_interleave(sizes)
        if verbose:
            print(f'InstanceData.load val_idx        : {time() - start:0.5f}s')

        # Read the obj and count, now we have computed the val_idx
        start = time()
        obj = load_tensor(f['obj'], idx=val_idx)
        count = load_tensor(f['count'], idx=val_idx)
        if verbose:
            print(f'InstanceData.load read values    : {time() - start:0.5f}s')

        # Build the InstanceData object
        start = time()
        out = InstanceData(pointers, obj, count)
        if verbose:
            print(f'InstanceData.load init           : {time() - start:0.5f}s')
        return out


class InstanceBatch(InstanceData, CSRBatch):
    """Wrapper for InstanceData batching."""
    __csr_type__ = InstanceData
