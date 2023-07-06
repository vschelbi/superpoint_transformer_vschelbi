import h5py
import torch
from time import time
from src.data.csr import CSRData, CSRBatch
from src.utils import tensor_idx, is_dense, save_tensor, load_tensor, \
    has_duplicates
from torch_scatter import scatter_max, scatter_sum
from torch_geometric.nn.pool.consecutive import consecutive_cluster


__all__ = ['InstanceData', 'InstanceBatch']


class InstanceData(CSRData):
    """Child class of CSRData to simplify some common operations
    dedicated to instance labels clustering. In particular, this data
    structure stores the cluster-object overlaps: for each cluster (ie
    segment, superpoint, node in the superpoint graph, etc), we store
    all the object instances with which it overlaps. Concretely, for
    each cluster-object pair, we store:
      - `obj`: the object's index
      - `count`: the number of points in the cluster-object overlap
      - `y`: the object's semantic label

    Importantly, each object in the InstanceData is expected to be
    described by a unique index in `obj', regardless of its actual
    semantic class. It is not required for the object instances to be
    contiguous in `[0, obj_max]`, although enforcing it may have
    beneficial downstream effects on memory and I/O times. Finally,
    when two InstanceData are batched in an InstanceBatch, the `obj'
    indices will be updated to avoid collision between the batch items.

    :param pointers: torch.LongTensor
        Pointers to address the data in the associated value tensors.
        `values[Pointers[i]:Pointers[i+1]]` old the values for the ith
        cluster. If `dense=True`, the `pointers` are actually the dense
        indices to be converted to pointer format.
    :param obj: torch.LongTensor
        Object index for each cluster-object pair. Assumes there are
        NO DUPLICATE CLUSTER-OBJECT pairs in the input data, unless
        'dense=True'.
    :param count: torch.LongTensor
        Number of points in the overlap for each cluster-object pair.
    :param y: torch.LongTensor
        Semantic label the object for each cluster-object pair. By
        definition, we assume the objects to be SEMANTICALLY PURE. For
        that reason, we only store a single semantic label for objects,
        as opposed to superpoints, for which we want to maintain a
        histogram of labels.
    :param dense: bool
        If `dense=True`, the `pointers` are actually the dense indices
        to be converted to pointer format. Besides, any duplicate
        cluster-obj pairs will be merged and the corresponding `count`
        will be updated.
    :param kwargs:
        Other kwargs will be ignored.
    """

    def __init__(self, pointers, obj, count, y, dense=False, **kwargs):
        # If the input data is passed in 'dense' format, we merge the
        # potential duplicate cluster-obj pairs before anything else.
        # NB: if dense=True, 'pointers' are not actual pointers but
        # dense cluster indices instead
        if dense:
            # Build indices to uniquely identify each cluster-obj pair
            cluster_obj_idx = pointers * (obj.max() + 1) + obj

            # Make the indices contiguous in [0, max] to alleviate
            # downstream scatter operations. Compute the cluster and obj
            # for each unique cluster_obj_idx index. These will be
            # helpful in building the cluster_idx and obj of the new
            # merged data
            cluster_obj_idx, perm = consecutive_cluster(cluster_obj_idx)
            pointers = pointers[perm]
            obj = obj[perm]
            y = y[perm]

            # Compute the actual count for each cluster-obj pair in the
            # input data
            count = scatter_sum(count, cluster_obj_idx)

        super().__init__(
            pointers, obj, count, y, dense=dense,
            is_index_value=[True, False, False])

    @staticmethod
    def get_batch_type():
        """Required by CSRBatch.from_list."""
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
    def y(self):
        return self.values[2]

    @y.setter
    def y(self, y):
        assert y.device == self.device, \
            f"y is on {y.device} while self is on {self.device}"
        self.values[2] = y
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
        """Return the obj, count, and y of the majority (ie most
        frequent) instance in each cluster.
        """
        # Compute the cluster index for each overlap (ie each row in
        # self.values)
        cluster_idx = self.indices

        # Search for the obj with the largest count, for each cluster
        count, argmax = scatter_max(self.count, cluster_idx)
        obj = self.obj[argmax]
        y = self.y[argmax]

        return obj, count, y

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
        # Make sure each cluster has a merge index and that the merge
        # indices are dense
        idx = tensor_idx(idx)
        assert idx.shape == torch.Size([self.num_clusters]), \
            f"Expected indices of shape {torch.Size([self.num_clusters])}, " \
            f"but received shape {idx.shape} instead"
        assert is_dense(idx), f"Expected contiguous indices in [0, max]"

        # Compute the merged cluster index for each cluster-obj pair
        merged_idx = idx[self.indices].long()

        # Return a new object holding the merged data.
        # NB: specifying 'dense=True' will do all the merging for us
        return InstanceData(
            merged_idx, self.obj, self.count, self.y, dense=True)

    def iou_and_size(self):
        """Compute the Intersection over Union (IoU) and the individual
        size for each cluster-object pair in the data. This is typically
        needed for computing the Average Precision.
        """
        # Prepare the indices for sets A (ie predictions) and B (ie
        # targets). In particular, we want the indices to be contiguous
        # in [0, idx_max], to alleviate scatter operations computation.
        # Since `self.obj` contains potentially-large and non-contiguous
        # global object indices, we update these indices locally
        a_idx = self.indices
        b_idx = consecutive_cluster(self.obj)[0]

        # Compute the size of each set and redistribute to each a-b pair
        a_size = scatter_sum(self.count, a_idx)[a_idx]
        b_size = scatter_sum(self.count, b_idx)[b_idx]

        # If self was created using `self.remove_void()`, use the
        # `self.pair_cropped_count` attribute to account for cropped
        # parts of b
        # TODO: `self.pair_cropped_count` is not accounted for in the
        #  `self.values`. InstanceBatch mechanisms will discard this
        #  value. ie 'pair_cropped_count' will disappear when calling
        #  `InstanceBatch.from_list` or `InstanceBatch.to_list`
        if getattr(self, 'pair_cropped_count', None) is not None:
            b_size += self.pair_cropped_count

        # Compute the IoU
        iou = self.count / (a_size + b_size - self.count)

        return iou, a_size, b_size

    def search_void(self, num_classes):
        """Search for clusters and objects with 'void' semantic labels.

        IMPORTANT:
        By convention, we assume `y ∈ [0, num_classes-1]` ARE ALL
        VALID LABELS (ie not 'void', 'ignored', 'unknown', etc),
        while `y < 0` AND `y >= num_classes` ARE IGNORED LABELS.
        This applies to both `Data.y` and `Data.obj.y`.

        Points with 'void' labels are handled following the procedure
        proposed in:
          - https://arxiv.org/abs/1801.00868
          - https://arxiv.org/abs/1905.01220

        More precisely, we remove from IoU and metrics computation:
          - predictions (ie clusters here) containing more than 50% of
            'void' points
          - targets (ie objects here) containing more than 50% of
            'void' points. In our case, we assume targets to be
            SEMANTICALLY PURE, so we remove a target even if it contains
            a single 'void' point

        To this end, the present function returns:
          - `cluster_mask`: boolean mask of the clusters containing more
            than 50% points with `void` labels
          - `pair_mask`: boolean mask of the cluster-object pairs whose
            object (ie target) has an `void` label
          - `pair_cropped_count`: tensor of cropped target size, for
            each pair. Indeed, blindly removing the predictions with 50%
            or more void points will affect downstream IoU computation.
            To account for this, this, `pair_cropped_count` is intended
            to be used at IoU computation time, when assessing the
            prediction and target sizes

        NB: by construction, removing pairs in `pair_mask` from the
            InstanceData will also remove all target objects containing
            'void' points. Importantly, this assumes, however, that the
            raw instance annotations in the datasets are semantically
            pure: all annotated instance contain points of the same
            class. Said otherwise: IF AN INSTANCE CONTAINS A SINGLE
            'VOID' POINT, THEN ALL OF ITS POINTS ARE 'VOID'.
        """
        # Identify the pairs whose object (ie target instance) is void.
        # For simplicity, we note 'a' for clusters/predictions and 'b'
        # for objects/targets/ground truths
        is_pair_b_void = (self.y < 0) | (self.y >= num_classes)

        # Get the cluster indices, for each cluster-object pair
        pair_a_idx = self.indices

        # Compute the size of each set and redistribute to each a-b pair
        a_size = scatter_sum(self.count, pair_a_idx)

        # Identify the indices of the clusters included in a void pair
        void_a_idx = pair_a_idx[is_pair_b_void].unique()

        # For those clusters specifically, identify those whose total
        # size encompasses more than 50% void points
        void_a_total_size = a_size[void_a_idx]
        void_a_void_size = scatter_sum(
            self.count[is_pair_b_void], pair_a_idx[is_pair_b_void])[void_a_idx]
        void_a_50_plus = (void_a_void_size / void_a_total_size.float()) > 0.5
        void_a_50_plus_idx = void_a_idx[void_a_50_plus]

        # Convert the indices to a boolean mask spanning the clusters
        is_a_void = torch.zeros(
            self.num_clusters, dtype=torch.bool, device=self.device)
        is_a_void[void_a_50_plus_idx] = True

        # Blindly removing the predictions with 50% or more void points
        # will affect downstream IoU computation. To account for this,
        # we search the affected target indices and compute the size of
        # the corresponding crop induced by void prediction removal.
        # Finally, we expand this as a pair-wise tensor, indicating the
        # missing crop size for each pair
        b_idx = consecutive_cluster(self.obj)[0]
        pair_cropped_count = scatter_sum(
            self.count * is_a_void[pair_a_idx], b_idx)[b_idx]

        # Update the pair-wise void mask, to account for the removal of
        # +50%-void predictions
        is_pair_void = is_pair_b_void | is_a_void[pair_a_idx]

        return is_a_void, is_pair_void, pair_cropped_count

    def remove_void(self, num_classes):
        """Return a new InstanceData with void clusters, objects and
        pairs removed.

        IMPORTANT:
        By convention, we assume `y ∈ [0, num_classes-1]` ARE ALL
        VALID LABELS (ie not 'void', 'ignored', 'unknown', etc),
        while `y < 0` AND `y >= num_classes` ARE IGNORED LABELS.
        This applies to both `Data.y` and `Data.obj.y`.

        Points with 'void' labels are handled following the procedure
        proposed in:
          - https://arxiv.org/abs/1801.00868
          - https://arxiv.org/abs/1905.01220

        More precisely:
          - predictions (ie clusters here) containing more than 50% of
            'void' points are removed from the metrics computation
          - targets (ie objects here) containing more than 50% of
            'void' points are removed from the metrics computation
          - the remaining 'void' points are ignored when computing the
            prediction-target (ie cluster-object here) IoUs

        To this end, the present function returns:
          - `instance_data`: a new InstanceData object with all void
            clusters, objects, and pairs removed
          - `non_void_mask`: boolean mask spanning the clusters,
            indicating the clusters that were preserved in the
            `instance_data`. This mask can be used outside of this
            function to subsample cluster-wise information after
            void-removal

        NB: by construction, removing pairs in `pair_mask` from the
            InstanceData will also remove all target objects containing
            'void' points. Importantly, this assumes, however, that the
            raw instance annotations in the datasets are semantically
            pure: all annotated instance contain points of the same
            class. Said otherwise: IF AN INSTANCE CONTAINS A SINGLE
            'VOID' POINT, THEN ALL OF ITS POINTS ARE 'VOID'.
        """
        # Get the masks for indexing void clusters and pairs
        is_cluster_void, is_pair_void, pair_cropped_count = \
            self.search_void(num_classes)

        print(self.obj[is_pair_void].unique())
        print(self.y[is_pair_void].unique())

        # Create a new InstanceData without void data
        idx = self.indices
        idx = idx[~is_pair_void]
        idx = consecutive_cluster(idx)[0]
        obj = self.obj[~is_pair_void]
        count = self.count[~is_pair_void]
        y = self.y[~is_pair_void]
        pair_cropped_count = pair_cropped_count[~is_pair_void]
        instance_data = InstanceData(idx, obj, count, y, dense=True)

        # Save the pair_cropped_count in the new InstanceData. This will
        # be used by `self.iou_and_size()` to cleanly account for the
        # removal of +50%-void predictions
        instance_data.pair_cropped_count = pair_cropped_count

        return instance_data, ~is_cluster_void

    def debug(self):
        super().debug()

        # Make sure there are no duplicate cluster-obj pairs
        cluster_obj_idx = self.indices * (self.obj.max() + 1) + self.obj
        assert not has_duplicates(cluster_obj_idx)

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
        save_tensor(self.y, f, 'y', fp_dtype=fp_dtype)

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
        KEYS = ['pointers', 'obj', 'count', 'y']

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
            y = load_tensor(f['y'])
            if verbose:
                print(f'InstanceData.load read all           : {time() - start:0.5f}s')
            start = time()
            out = InstanceData(pointers, obj, count, y)
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
        y = load_tensor(f['y'], idx=val_idx)
        if verbose:
            print(f'InstanceData.load read values    : {time() - start:0.5f}s')

        # Build the InstanceData object
        start = time()
        out = InstanceData(pointers, obj, count, y)
        if verbose:
            print(f'InstanceData.load init           : {time() - start:0.5f}s')
        return out


class InstanceBatch(InstanceData, CSRBatch):
    """Wrapper for InstanceData batching. Importantly, although
    instance labels in 'obj' will be updated to avoid collisions between
    the different batch items.
    """
    __csr_type__ = InstanceData
