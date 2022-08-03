import os
import os.path as osp
import numpy as np
import torch
from plyfile import PlyData
from torch.utils.data import Sampler
import logging
from sklearn.neighbors import KDTree
from tqdm.auto import tqdm as tq
from random import shuffle
from datetime import datetime

from superpoint_transformer.data import Data
# import superpoint_transformer.core.data_transform as cT
# from superpoint_transformer.datasets.base_dataset import BaseDataset
from superpoint_transformer.datasets.kitti360_config import *
# from superpoint_transformer.utils.download import run_command

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)


########################################################################
#                                 Utils                                #
########################################################################

def read_kitti360_window(
        filepath, xyz=True, rgb=True, semantic=True, instance=False,
        remap=False):
    data = Data()
    with open(filepath, "rb") as f:
        window = PlyData.read(f)
        attributes = [p.name for p in window['vertex'].properties]

        if xyz:
            data.pos = torch.stack([
                torch.FloatTensor(window["vertex"][axis])
                for axis in ["x", "y", "z"]], dim=-1)

        if rgb:
            data.rgb = torch.stack([
                torch.FloatTensor(window["vertex"][axis])
                for axis in ["red", "green", "blue"]], dim=-1) / 255

        if semantic and 'semantic' in attributes:
            y = torch.LongTensor(window["vertex"]['semantic'])
            data.y = torch.from_numpy(ID2TRAINID)[y] if remap else y

        if instance and 'instance' in attributes:
            data.instance = torch.LongTensor(window["vertex"]['instance'])

    return data

