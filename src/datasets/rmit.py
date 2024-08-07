import os
import logging
from src.datasets import FORinstance
from src.datasets.forinstance_config import *

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)

__all__ = ['RMIT', 'MiniRMIT']


########################################################################
#                                 RMIT                                 #
########################################################################

class RMIT(FORinstance):
    """RMIT dataset, extracted from the FOR-Instance dataset.

    Dataset link: https://paperswithcode.com/dataset/FOR-Instance

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

    @property
    def all_base_cloud_ids(self):
        """Dictionary holding lists of paths to the clouds, for each
        stage.

        The following structure is expected:
            `{'train': [...], 'val': [...], 'test': [...]}`
        """
        return {
            split: [x for x in plots if x.startswith('RMIT')]
            for split, plots in TILES.items()}


########################################################################
#                               MiniRMIT                               #
########################################################################

class MiniRMIT(RMIT):
    """A mini version of RMIT with only a few plots for experimentation.
    """
    _NUM_MINI = 2

    @property
    def all_cloud_ids(self):
        return {k: v[:self._NUM_MINI] for k, v in super().all_cloud_ids.items()}

    @property
    def data_subdir_name(self):
        return self.__class__.__bases__[0].__name__.lower()

    # We have to include this method, otherwise the parent class skips
    # processing
    def process(self):
        super().process()

    # We have to include this method, otherwise the parent class skips
    # processing
    def download(self):
        super().download()
