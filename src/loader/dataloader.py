from torch.utils.data import DataLoader as TorchDataLoader


__all__ = ['DataLoader']


class DataLoader(TorchDataLoader):
    """Same as torch DataLoader except that the default behaviour for
    `collate_fn=None` is a simple identity. (ie the DataLoader will
    return a list of elements by default). This approach is meant to
    move the CPU-hungry NAG.from_nag_list (in particular, the level-0
    Data.from_nag_list) to GPU. This would ideally be taken care of by
    a pytorch lightning callback.

    Use `collate_fn=NAG.from_data_list` if you want the CPU to do this
    operation.
    """
    def __init__(self, *args, collate_fn=None, **kwargs):
        if collate_fn is None:
            collate_fn = lambda batch_list: batch_list
        super().__init__(*args, collate_fn=collate_fn, **kwargs)
