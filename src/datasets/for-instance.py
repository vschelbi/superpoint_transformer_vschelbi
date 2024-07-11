from src.datasets import BaseDataset


########################################################################
#                                FOR-instance                          #
########################################################################
class FORinstance(BaseDataset):
    """ FOR-instance dataset.

    Dataset link: https://paperswithcode.com/dataset/for-instance

    Parameters
    ----------
    root: str
        Root directory where the dataset is stored.
    stage : {'train', 'val', 'test', 'trainval'}, optional
    transform : `callable`, optional
        transform function operating on data.
    pre_transform : `callable`, optional
        pre_transform function operating on data.
    pre_filter : `callable`, optional
        pre_filter function operating on data.
    on_device_transform: `callable`, optional
        on_device_transform function operating on data, in the
        'on_after_batch_transfer' hook. This is where GPU-based
        augmentations should be, as well as any Transform you do not
        want to run in CPU-based DataLoaders
    """

