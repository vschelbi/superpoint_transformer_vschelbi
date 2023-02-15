import torch
from src.data import Data, NAG
from src.transforms import Transform
from src.utils import tensor_idx
from torch_geometric.nn.pool.consecutive import consecutive_cluster


__all__ = [
    'DataToNAG', 'NAGToData', 'RemoveKeys', 'NAGRemoveKeys', 'AddKeyToX',
    'NAGAddKeyToX', 'NAGSelectByKey', 'SelectColumns', 'NAGSelectColumns',
    'DropoutColumns', 'NAGDropoutColumns']


class DataToNAG(Transform):
    """Convert Data to a single-level NAG."""

    _IN_TYPE = Data
    _OUT_TYPE = NAG

    def _process(self, data):
        return NAG([Data])


class NAGToData(Transform):
    """Convert a single-level NAG to Data."""

    _IN_TYPE = NAG
    _OUT_TYPE = Data

    def _process(self, nag):
        assert nag.num_levels == 1
        return NAG[0]


class RemoveKeys(Transform):
    """Remove attributes of a Data object based on their name.

    :param keys: list(str)
        List of attribute names
    :param strict: bool=False
        If True, will raise an exception if an attribute from key is
        not within the input Data keys
    """

    _NO_REPR = ['strict']

    def __init__(self, keys=[], strict=False):
        self.keys = keys
        self.strict = strict

    def _process(self, data):
        keys = set(data.keys)
        for k in self.keys:
            if k not in keys and self.strict:
                raise Exception(f"key: {k} is not within Data keys: {keys}")
        for k in self.keys:
            delattr(data, k)
        return data


class NAGRemoveKeys(Transform):
    """Remove attributes of a NAG object based on their name.

    :param level: int or str
        Level at which to remove attributes. Can be an int or a str. If
        the latter, 'all' will apply on all levels, 'i+' will apply on
        level-i and above, 'i-' will apply on level-i and below
    :param keys: list(str)
        List of attribute names
    :param strict: bool=False
        If True, will raise an exception if an attribute from key is
        not within the input Data keys
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG
    _NO_REPR = ['strict']

    def __init__(self, level='all', keys=[], strict=False):
        assert isinstance(level, (int, str))
        self.level = level
        self.keys = keys
        self.strict = strict

    def _process(self, nag):

        level_keys = [[]] * nag.num_levels
        if isinstance(self.level, int):
            level_keys[self.level] = self.keys
        elif self.level == 'all':
            level_keys = [self.keys] * nag.num_levels
        elif self.level[-1] == '+':
            i = int(self.level[:-1])
            level_keys[i:] = [self.keys] * (nag.num_levels - i)
        elif self.level[-1] == '-':
            i = int(self.level[:-1])
            level_keys[:i] = [self.keys] * i
        else:
            raise ValueError(f'Unsupported level={self.level}')

        transforms = [RemoveKeys(keys=k) for k in level_keys]

        for i_level in range(nag.num_levels):
            nag._list[i_level] = transforms[i_level](nag._list[i_level])

        return nag


class AddKeyToX(Transform):
    """Get attributes from their keys and concatenate them to x.

    :param keys: str or list(str)
        The feature concatenated to x
    :param strict: bool, optional
        Whether we want to raise an error if a key is not found
    :param delete_after: bool, optional
        Whether the Data attributes should be removed once added to x
    """

    _NO_REPR = ['strict']

    def __init__(self, keys=None, strict=True, delete_after=True):
        self.keys = keys
        self.strict = strict
        self.delete_after = delete_after

    def _process_single_key(self, data, key):
        # Read existing features and the attribute of interest
        feat = getattr(data, key, None)
        x = getattr(data, "x", None)

        # Skip if the attribute is None
        if feat is None:
            if self.strict:
                raise Exception(f"Data should contain the attribute {key}")
            else:
                return data

        # Remove the attribute from the Data, if required
        if self.delete_after:
            delattr(data, key)

        # In case Data has no features yet
        if x is None:
            if self.strict and data.num_nodes != feat.shape[0]:
                raise Exception("We expected to have an attribute x")
            if feat.dim() == 1:
                feat = feat.unsqueeze(-1)
            data.x = feat
            return data

        # Make sure shapes match
        if x.shape[0] != feat.shape[0]:
            raise Exception(
                f"The tensor x and {key} can't be concatenated, x: "
                f"{x.shape[0]}, feat: {feat.shape[0]}")

        # Concatenate x and feat
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if feat.dim() == 1:
            feat = feat.unsqueeze(-1)
        data.x = torch.cat([x, feat], dim=-1)

        return data

    def _process(self, data):
        if self.keys is None or len(self.keys) == 0:
            return data

        for key in self.keys:
            data = self._process_single_key(data, key)

        return data


class NAGAddKeyToX(Transform):
    """Get attributes from their keys and concatenate them to x.

    :param level: int or str
        Level at which to remove attributes. Can be an int or a str. If
        the latter, 'all' will apply on all levels, 'i+' will apply on
        level-i and above, 'i-' will apply on level-i and below
    :param keys: str or list(str)
        The feature concatenated to x
    :param strict: bool, optional
        Whether we want to raise an error if a key is not found
    :param delete_after: bool, optional
        Whether the Data attributes should be removed once added to x
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG
    _NO_REPR = ['strict']

    def __init__(self, level='all', keys=None, strict=True, delete_after=True):
        self.level = level
        self.keys = keys
        self.strict = strict
        self.delete_after = delete_after

    def _process(self, nag):

        level_keys = [[]] * nag.num_levels
        if isinstance(self.level, int):
            level_keys[self.level] = self.keys
        elif self.level == 'all':
            level_keys = [self.keys] * nag.num_levels
        elif self.level[-1] == '+':
            i = int(self.level[:-1])
            level_keys[i:] = [self.keys] * (nag.num_levels - i)
        elif self.level[-1] == '-':
            i = int(self.level[:-1])
            level_keys[:i] = [self.keys] * i
        else:
            raise ValueError(f'Unsupported level={self.level}')

        transforms = [
            AddKeyToX(
                keys=k, strict=self.strict, delete_after=self.delete_after)
            for k in level_keys]

        for i_level in range(nag.num_levels):
            nag._list[i_level] = transforms[i_level](nag._list[i_level])

        return nag


class NAGSelectByKey(Transform):
    """Select the i-level nodes based on a key. The corresponding key is
    expected to exist in the i-level attributes and should hold a 1D
    boolean mask.

    :param key: str
        Key attribute expected to be found in the input NAG's `level`.
        The `key` attribute should carry a 1D boolean mask over the
        `level` nodes
    :param level: int
        NAG level based on which to operate the selection
    :param negation: bool
        Whether the mask or its complementary should be used
    :param strict: bool, optional
        Whether we want to raise an error if the key is not found or if
        it does not carry a 1D boolean mask
    :param delete_after: bool, optional
        Whether the `key` attribute should be removed after selection
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG
    _NO_REPR = ['strict']

    def __init__(
            self, key=None, level=0, negation=False, strict=True,
            delete_after=True):
        assert key is not None
        self.key = key
        self.level = level
        self.negation = negation
        self.strict = strict
        self.delete_after = delete_after

    def _process(self, nag):
        # Ensure the key exists
        if self.key not in nag[self.level].keys:
            if self.strict:
                raise ValueError(
                    f'Input NAG does not have `{self.key}` attribute at '
                    f'level `{self.level}`')
            return nag

        # Read the mask
        mask = nag[self.level][self.key]

        # Ensure the mask is a boolean tensor
        dtype = mask.dtype
        if dtype != torch.bool:
            if self.strict:
                raise ValueError(
                    f'`{self.key}` attribute has dtype={dtype} but '
                    f'dtype=torch.bool was expected')
            return nag

        # Ensure the mask size matches
        expected_size = torch.Size((nag[self.level].num_nodes,))
        actual_size = mask.shape
        if expected_size != actual_size:
            if self.strict:
                raise ValueError(
                    f'`{self.key}` attribute has shape={actual_size} but '
                    f'shape={expected_size} was expected')
            return nag

        # Call NAG.select using the mask on the `level` nodes
        mask = ~mask if self.negation else mask
        nag = nag.select(self.level, torch.where(mask)[0])

        # Remove the key if need be
        if self.delete_after:
            nag[self.level][self.key] = None

        return nag


class SelectColumns(Transform):
    """Select columns of an attribute based on their indices.

    :param key: str
        The Data attribute whose columns should be selected
    :param idx: int, Tensor or list
        The indices of the edge features to keep. If None, this
        transform will have no effect and edge features will be left
        untouched
    """

    def __init__(self, key=None, idx=None):
        assert key is not None, f"A Data key must be specified"
        self.key = key
        self.idx = tensor_idx(idx) if idx is not None else None

    def _process(self, data):
        if self.idx is None or getattr(data, self.key, None) is None:
            return data
        data[self.key] = data[self.key][:, self.idx.to(device=data.device)]
        return data


class NAGSelectColumns(Transform):
    """Select columns of an attribute based on their indices.

    :param level: int or str
        Level at which to select attributes. Can be an int or a str. If
        the latter, 'all' will apply on all levels, 'i+' will apply on
        level-i and above, 'i-' will apply on level-i and below
    :param key: str
        The Data attribute whose columns should be selected
    :param idx: int, Tensor or list
        The indices of the edge features to keep. If None, this
        transform will have no effect and edge features will be left
        untouched
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, level='all', key=None, idx=None):
        self.level = level
        self.key = key
        self.idx = idx

    def _process(self, nag):

        level_idx = [None] * nag.num_levels
        if isinstance(self.level, int):
            level_idx[self.level] = self.idx
        elif self.level == 'all':
            level_idx = [self.idx] * nag.num_levels
        elif self.level[-1] == '+':
            i = int(self.level[:-1])
            level_idx[i:] = [self.idx] * (nag.num_levels - i)
        elif self.level[-1] == '-':
            i = int(self.level[:-1])
            level_idx[:i] = [self.idx] * i
        else:
            raise ValueError(f'Unsupported level={self.level}')

        transforms = [SelectColumns(key=self.key, idx=idx) for idx in level_idx]

        for i_level in range(nag.num_levels):
            nag._list[i_level] = transforms[i_level](nag._list[i_level])

        return nag


class DropoutColumns(Transform):
    """Randomly set a Data attribute column to 0.

    :param p: float
        Probability of a column to be dropped
    :param key: str
        The Data attribute whose columns should be selected
    :param group: Tensor or list
        The indices by which columns should be grouped. If two columns
        have the same group index, they will be always be dropped
        together
    """

    def __init__(self, p=0.5, key=None, group=None):
        assert key is not None, f"A Data key must be specified"
        self.p = p
        self.key = key
        self.group = tensor_idx(group).tolist() if group is not None else None

    def _process(self, data):
        device = data.device

        # Skip dropout if the attribute is not present in the input Data
        if getattr(data, self.key, None) is None:
            return data

        # Recover the Data attribute of interest
        attr = data[self.key]
        if attr.dim() == 1:
            attr = attr.view(-1, 1)
        num_col = attr.shape[1]

        # Prepare the column indexing. In case an index was passed, the
        # columns sharing the same index will undergo the same dropout
        group = torch.tensor(self.group, device=device) if self.group is not None \
            else torch.arange(num_col, device=device)
        group = consecutive_cluster(group)[0]
        assert group.shape[0] == num_col

        # Compute a boolean mask across the columns, indicating whether
        # they should (not) be dropped
        mask = torch.rand(attr.shape[1], device=data.device) > self.p
        attr *= mask[group].float()

        # Restore the Data attribute
        data[self.key] = attr

        return data


class NAGDropoutColumns(Transform):
    """Randomly set a Data attribute column to 0.

    :param level: int or str
        Level at which to drop columns. Can be an int or a str. If
        the latter, 'all' will apply on all levels, 'i+' will apply on
        level-i and above, 'i-' will apply on level-i and below
    :param p: float
        Probability of a column to be dropped
    :param key: str
        The Data attribute whose columns should be selected
    :param group: Tensor or list
        The indices by which columns should be grouped. If two columns
        have the same group index, they will be always be dropped
        together
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, level='all', p=0.5, key=None, group=None):
        self.level = level
        self.p = p
        self.key = key
        self.group = group

    def _process(self, nag):

        level_p = [0] * nag.num_levels
        if isinstance(self.level, int):
            level_p[self.level] = self.p
        elif self.level == 'all':
            level_p = [self.p] * nag.num_levels
        elif self.level[-1] == '+':
            i = int(self.level[:-1])
            level_p[i:] = [self.p] * (nag.num_levels - i)
        elif self.level[-1] == '-':
            i = int(self.level[:-1])
            level_p[:i] = [self.p] * i
        else:
            raise ValueError(f'Unsupported level={self.level}')

        transforms = [
            DropoutColumns(p=p, key=self.key, group=self.group)
            for p in level_p]

        for i_level in range(nag.num_levels):
            nag._list[i_level] = transforms[i_level](nag._list[i_level])

        return nag
