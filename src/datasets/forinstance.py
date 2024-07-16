import os
import sys
import torch
import shutil
import logging
import os.path as osp
import laspy
from src.datasets import BaseDataset
from src.data import Data, InstanceData
from src.datasets.forinstance_config import *
from torch_geometric.data import extract_tar
from torch_geometric.nn.pool.consecutive import consecutive_cluster


DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)


# Occasional Dataloader issues with DALES on some machines. Hack to
# solve this:
# https://stackoverflow.com/questions/73125231/pytorch-dataloaders-bad-file-descriptor-and-eof-for-workers0
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')


__all__ = ['FORinstance', 'MiniFORinstance']

########################################################################
#                                 Utils                                #
########################################################################
def read_FORinstance_plot(
        filepath, xyz=True, intensity=True, semantic=True, instance=True, 
        remap=True, max_intensity=600):
    """Read a FORinstance file saved as LAS.

    :param filepath: str
        Absolute path to the LAS file
    :param xyz: bool
        Whether XYZ coordinates should be saved in the output Data.pos
    :param intensity: bool
        Whether intensity should be saved in the output Data.rgb
    :param semantic: bool
        Whether semantic labels should be saved in the output Data.y
    :param instance: bool
        Whether instance labels should be saved in the output Data.obj
    :param remap: bool
        Whether semantic labels should be mapped from their FORinstance ID
        to their train ID
    :param max_intensity: float
        Maximum value used to clip intensity signal before normalizing 
        to [0, 1]
    """
    data = Data()
    key = 'testing'


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

    # TODO 















########################################################################
#                              MiniDALES                               #
########################################################################

class MiniFORinstance(FORinstance):
    """A mini version of FOR-instance with only a few windows for
    experimentation.
    """
    _NUM_MINI = 2

    # TODO