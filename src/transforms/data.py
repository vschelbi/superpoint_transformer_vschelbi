import torch
from src.data import Data, NAG
from src.transforms import Transform


__all__ = [
    'DataToNAG', 'NAGToData', 'RemoveKeys', 'NAGRemoveKeys', 'AddKeyToX',
    'NAGAddKeyToX', 'NAGSelectByKey']


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
