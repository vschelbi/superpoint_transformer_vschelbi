import torch
import copy
from superpoint_transformer.utils import tensor_idx, is_sorted, is_dense


class CSRData:
    """Implements the CSRData format and associated mechanisms in Torch.

    When defining a subclass A of CSRData, it is recommended to create
    an associated CSRBatch subclass by doing the following:
        - ABatch inherits from (A, CSRBatch)
        - A.get_batch_type() returns ABatch
    """

    def __init__(
            self, pointers: torch.LongTensor, *args, dense=False,
            is_index_value=None, debug=False):
        """Initialize the pointers and values.

        Values are passed as args and stored in a list. They are
        expected to all have the same size and support torch tensor
        indexing (i.e. they can be torch tensor or CSRData objects
        themselves).

        If `dense=True`, pointers are treated as a dense tensor of
        indices to be converted into pointer indices.

        Optionally, a list of booleans `is_index_value` can be passed.
        It must be the same size as *args and indicates, for each value,
        whether it holds elements that should be treated as indices when
        stacking CSRData objects into a CSRBatch. If so, the indices
        will be updated wrt the cumulative size of the batched values.
        """
        if dense:
            self.pointers, order = CSRData.indices_to_pointers(pointers)
            args = [a[order] for a in args]
        else:
            self.pointers = pointers
        self.values = [*args] if len(args) > 0 else None
        if is_index_value is None or is_index_value == []:
            self.is_index_value = torch.zeros(
                self.num_values, dtype=torch.bool)
        else:
            self.is_index_value = torch.BoolTensor(is_index_value)
        if debug:
            self.debug()

    def debug(self):
        # assert self.num_groups >= 1, \
        #     "pointer indices must cover at least one group."
        assert self.pointers[0] == 0, \
            "The first pointer element must always be 0."
        assert torch.all(self.pointers[1:] - self.pointers[:-1] >= 0), \
            "pointer indices must be increasing."

        if self.values is not None:
            assert isinstance(self.values, list), \
                "Values must be held in a list."
            assert all([len(v) == self.num_items for v in self.values]), \
                "All value objects must have the same size."
            assert len(self.values[0]) == self.num_items, \
                "pointers must cover the entire range of values."
            for v in self.values:
                if isinstance(v, CSRData):
                    v.debug()

        if self.values is not None and self.is_index_value is not None:
            assert isinstance(self.is_index_value, torch.BoolTensor), \
                "is_index_value must be a torch.BoolTensor."
            assert self.is_index_value.dtype == torch.bool, \
                "is_index_value must be an tensor of booleans."
            assert self.is_index_value.ndim == 1, \
                "is_index_value must be a 1D tensor."
            assert self.is_index_value.shape[0] == self.num_values, \
                "is_index_value size must match the number of value tensors."

    def to(self, device):
        """Move the CSRData to the specified device."""
        out = self.clone()
        out.pointers = out.pointers.to(device)
        for i in range(out.num_values):
            out.values[i] = out.values[i].to(device)
        return out

    def cpu(self):
        """Move the CSRData to the CPU."""
        return self.to('cpu')

    def cuda(self):
        """Move the CSRData to the first available GPU."""
        return self.to('cuda')

    @property
    def device(self):
        return self.pointers.device

    @property
    def num_groups(self):
        return self.pointers.shape[0] - 1

    @property
    def num_values(self):
        return len(self.values) if self.values is not None else 0

    @property
    def num_items(self):
        return self.pointers[-1].item()

    @property
    def sizes(self):
        return self.pointers[1:] - self.pointers[:-1]

    @staticmethod
    def get_batch_type():
        """Required by CSRBatch.from_csr_list."""
        #TODO: CSRBatch
        raise NotImplementedError

    def clone(self):
        """Shallow copy of self. This may cause issues for certain types
        of downstream operations but it saves time and memory. In
        practice, it shouldn't in this project.
        """
        out = copy.copy(self)
        out.pointers = copy.copy(self.pointers)
        out.values = copy.copy(self.values)
        return out

    @staticmethod
    def indices_to_pointers(indices: torch.LongTensor):
        """Convert pre-sorted dense indices to CSR format."""
        device = indices.device
        assert len(indices.shape) == 1, "Only 1D indices are accepted."
        assert indices.shape[0] >= 1, "At least one group index is required."
        assert is_dense(indices), "Indices must be dense"

        # Sort indices if need be
        order = torch.arange(indices.shape[0], device=device)
        if not is_sorted(indices):
            indices, order = indices.sort()

        # Convert sorted indices to pointers
        pointers = torch.cat([
            torch.LongTensor([0]).to(device),
            torch.where(indices[1:] > indices[:-1])[0] + 1,
            torch.LongTensor([indices.shape[0]]).to(device)])

        return pointers, order

    def reindex_groups(
            self, group_indices: torch.LongTensor, order=None,
            num_groups=None):
        """Returns a copy of self with modified pointers to account for
        new groups. Affects the num_groups and the order of groups.
        Injects 0-length pointers where need be.

        By default, pointers are implicitly linked to the group indices
        in range(0, self.num_groups).

        Here we provide new group_indices for the existing pointers,
        with group_indices[i] corresponding to the position of existing
        group i in the new tensor. The indices missing from
        group_indices account for empty groups to be injected.

        The num_groups specifies the number of groups in the new tensor.
        If not provided, it is inferred from the size of group_indices.
        """
        if order is None:
            order = torch.argsort(group_indices)
        csr_new = self[order].insert_empty_groups(
            group_indices[order], num_groups=num_groups)
        return csr_new

    def insert_empty_groups(
            self, group_indices: torch.LongTensor, num_groups=None):
        """Method called when in-place reindexing groups.

        The group_indices are assumed to be sorted and group_indices[i]
        corresponds to the position of existing group i in the new
        tensor. The indices missing from group_indices correspond to
        empty groups to be injected.

        The num_groups specifies the number of groups in the new tensor.
        If not provided, it is inferred from the size of group_indices.
        """
        assert self.num_groups == group_indices.shape[0], \
            "New group indices must correspond to the existing number " \
            "of groups"
        assert is_sorted(group_indices), "New group indices must be sorted."

        if num_groups is not None:
            num_groups = max(group_indices.max() + 1, num_groups)
        else:
            num_groups = group_indices.max() + 1

        starts = torch.cat([
            torch.LongTensor([-1]).to(self.device),
            group_indices.to(self.device)])
        ends = torch.cat([
            group_indices.to(self.device),
            torch.LongTensor([num_groups]).to(self.device)])
        repeats = ends - starts
        self.pointers = self.pointers.repeat_interleave(repeats)

        return self

    @staticmethod
    def index_select_pointers(
            pointers: torch.LongTensor, indices: torch.LongTensor):
        """Index selection of pointers.

        Returns a new pointer tensor with updated pointers, along with
        an index tensor to be used to update any values tensor
        associated with the input pointers.
        """
        assert indices.max() <= pointers.shape[0] - 2
        device = pointers.device

        # Create the new pointers
        pointers_new = torch.cat([
            torch.zeros(1, dtype=pointers.dtype, device=device),
            torch.cumsum(pointers[indices + 1] - pointers[indices], 0)])

        # Create the indexing tensor to select and order values.
        # Simply, we could have used a list of slices but we want to
        # avoid for loops and list concatenations to benefit from torch
        # capabilities.
        sizes = pointers_new[1:] - pointers_new[:-1]
        val_idx = torch.arange(pointers_new[-1]).to(device)
        val_idx -= torch.arange(
            pointers_new[-1] + 1)[pointers_new[:-1]].repeat_interleave(
            sizes).to(device)
        val_idx += pointers[indices].repeat_interleave(sizes).to(device)

        return pointers_new, val_idx

    def __getitem__(self, idx):
        """Indexing CSRData format. Supports Numpy and torch indexing
        mechanisms.

        Return a copy of self with updated pointers and values.
        """
        idx = tensor_idx(idx).to(self.device)

        # Shallow copy self and edit pointers and values. This
        # preserves the class for CSRData subclasses.
        out = self.clone()

        # If idx is empty, return an empty CSRData with empty values
        # of consistent type
        if idx.shape[0] == 0:
            out = self.clone()
            out.pointers = torch.LongTensor([0])
            out.values = [v[[]] for v in self.values]

        else:
            # Select the pointers and prepare the values indexing
            pointers, val_idx = CSRData.index_select_pointers(
                self.pointers, idx)
            out.pointers = pointers
            out.values = [v[val_idx] for v in self.values]

        # out.debug()

        return out

    def __len__(self):
        return self.num_groups

    def __repr__(self):
        info = [
            f"{key}={getattr(self, key)}"
            for key in ['num_groups', 'num_items', 'device']]
        return f"{self.__class__.__name__}({', '.join(info)})"
