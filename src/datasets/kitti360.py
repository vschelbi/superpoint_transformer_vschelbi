import os
import torch
import logging
from plyfile import PlyData
from src.datasets import BaseDataset, MiniDataset
from src.data import Data
from src.datasets.kitti360_config import *
from src.utils.download import run_command

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)


__all__ = ['KITTI360', 'MiniKITTI360']


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


########################################################################
#                               KITTI360                               #
########################################################################

class KITTI360(BaseDataset):
    """KITTI360 dataset.

    Dataset website: http://www.cvlibs.net/datasets/kitti-360/

    Parameters
    ----------
    root : `str`
        Root directory where the dataset should be saved.
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
    def class_names(self):
        """List of string names for dataset classes. This list may be
        one-item larger than `self.num_classes` if the last label
        corresponds to 'unlabelled' or 'ignored' indices, indicated as
        `-1` in the dataset labels.
        """
        raise CLASS_NAMES

    @property
    def num_classes(self):
        """Number of classes in the dataset. May be one-item smaller
        than `self.class_names`, to account for the last class name
        being optionally used for 'unlabelled' or 'ignored' classes,
        indicated as `-1` in the dataset labels.
        """
        raise KITTI360_NUM_CLASSES

    @property
    def all_clouds(self):
        """Dictionary holding lists of paths to the clouds, for each
        stage.

        The following structure is expected:
            `{'train': [...], 'val': [...], 'test': [...]}`
        """
        return WINDOWS

    def download_dataset(self):
        """Download the KITTI-360 dataset.
        """
        # Location of the KITTI-360 download shell scripts
        here = osp.dirname(osp.abspath(__file__))
        scripts_dir = osp.join(here, '../../scripts')

        # Accumulated 3D point clouds with annotations
        if not all(osp.exists(osp.join(self.raw_dir, x))
                   for x in self.raw_file_names_3d):
            if self.stage != 'test':
                msg = 'Accumulated Point Clouds for Train & Val (12G)'
            else:
                msg = 'Accumulated Point Clouds for Test (1.2G)'
            self.download_message(msg)
            script = osp.join(scripts_dir, 'download_kitti360_3d_semantics.sh')
            run_command([f'{script} {self.raw_dir} {self.stage}'])

    def read_single_raw_cloud(self, cloud_path):
        """Read a single raw cloud and return a Data object, ready to
        be passed to `self.pre_transform`.
        """
        # Extract useful information from <path>
        stage, hash_dir, sequence_name, cloud_name = \
            osp.splitext(cloud_path)[0].split('/')[-4:]

        # Read the raw cloud data
        raw_cloud_path = osp.join(
            self.raw_dir, 'data_3d_semantics', sequence_name, 'static',
            cloud_name + '.ply')
        data = read_kitti360_window(
            raw_cloud_path, semantic=True, instance=False, remap=True)

        return data

    @property
    def raw_file_structure(self):
        return f"""
    {self.root}/
        └── raw/
            └── data_3d_semantics/
                └── 2013_05_28_drive_{{seq:0>4}}_sync/
                    └── static/
                        └── {{start_frame:0>10}}_{{end_frame:0>10}}.ply
            """

    def cloud_to_relative_raw_path(self, cloud):
        """Given a cloud name as stored in `self.clouds`, return the
        path (relative to `self.raw_dir`) of the corresponding raw
        cloud.
        """
        return osp.join(
            'data_3d_semantics', cloud.split('/')[0], 'static',
            cloud.split('/')[1] + '.ply')

    def processed_to_raw_path(self, processed_path):
        """Return the raw cloud path corresponding to the input
        processed path.
        """
        # Extract useful information from <path>
        stage, hash_dir, sequence_name, cloud_name = \
            osp.splitext(processed_path)[0].split('/')[-4:]

        # Read the raw cloud data
        raw_path = osp.join(
            self.raw_dir, 'data_3d_semantics', sequence_name, 'static',
            cloud_name + '.ply')

        return raw_path








########################################################################
#                         MiniKITTI360Cylinder                         #
########################################################################

class MiniKITTI360(MiniDataset, KITTI360):
    """A mini version of KITTI360 with only a few windows for
    experimentation.
    """
    _NUM_MINI = 2
