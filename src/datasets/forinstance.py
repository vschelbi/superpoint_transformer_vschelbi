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
from torch_geometric.data import extract_zip
from torch_geometric.nn.pool.consecutive import consecutive_cluster


DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)

# TODO check if this is necessary also for FORinstance:
# Occasional Dataloader issues on some machines. Hack to solve this:
# https://stackoverflow.com/questions/73125231/pytorch-dataloaders-bad-file-descriptor-and-eof-for-workers0
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')


__all__ = ['FORinstance', 'MiniFORinstance']

########################################################################
#                                 Utils                                #
########################################################################

def read_FORinstance_plot(filepath, xyz=True, intensity=True, 
                           semantic=True, instance=True, remap=True, 
                           max_intensity=None):
    """
    Read a FORinstance plot from a LAS file and return the data object.

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
    las = laspy.read(filepath)

    if xyz:
        pos = torch.stack([
            torch.as_tensor(np.array(las[axis]))
            for axis in ["X", "Y", "Z"]], dim=-1)
        pos *= las.header.scale
        pos_offset = pos[0]
        data.pos = (pos - pos_offset).float()
        data.pos_offset = pos_offset
    
    intensity_remaped = True
    if intensity:
        data.intensity = torch.FloatTensor(
            las['intensity'].astype('float32')
        )
        if intensity_remaped:
            if max_intensity is None:
                max_intensity = data.intensity.max()
            data.intensity = data.intensity.clip(
                min=0, max=max_intensity) / max_intensity

    if semantic:
        y = torch.LongTensor(np.array(las['classification']))
        data.y = torch.from_numpy(ID2TRAINID)[y] if remap else y

    if instance:
        idx = torch.arange(data.num_points)
        obj = torch.LongTensor(np.array(las['treeID']))
        
        y = torch.LongTensor(np.array(las['classification']))
        y = torch.from_numpy(ID2TRAINID)[y] if remap else y

        if remap:
            ground_mask = (obj == 0) & (y == 0)
            low_veg_mask = (obj == 0) & (y == 1)
            if low_veg_mask.any() or ground_mask.any():
                ground_instance_label = obj.max().item() + 1
                # for separate ground and low vegetation classes:
                # ground_instance_label + 1
                low_veg_instance_label = ground_instance_label
                obj[ground_mask] = ground_instance_label
                obj[low_veg_mask] = low_veg_instance_label

        obj = consecutive_cluster(obj)[0]
        count = torch.ones_like(obj)

        data.obj = InstanceData(idx, obj, count, y, dense=True)
    
    return data


########################################################################
#                             FOR-Instance                             #
########################################################################

class FORinstance(BaseDataset):
    """FOR-Instance dataset.

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

    _download_url = DOWNLOAD_URL
    _las_name = LAS_ZIP_NAME
    _unzip_name = LAS_UNZIP_NAME

    @property
    def class_names(self):
        """List of string names for dataset classes. This list must be
        one-item larger than `self.num_classes`, with the last label
        corresponding to 'void', 'unlabelled', 'ignored' classes,
        indicated as `y=self.num_classes` in the dataset labels.
        """
        return CLASS_NAMES

    @property
    def num_classes(self):
        """Number of classes in the dataset. Must be one-item smaller
        than `self.class_names`, to account for the last class name
        being used for 'void', 'unlabelled', 'ignored' classes,
        indicated as `y=self.num_classes` in the dataset labels.
        """
        return FORInstance_NUM_CLASSES
    
    @property
    def stuff_classes(self):
        """List of 'stuff' labels for INSTANCE and PANOPTIC
        SEGMENTATION (setting this is NOT REQUIRED FOR SEMANTIC
        SEGMENTATION alone). By definition, 'stuff' labels are labels in
        `[0, self.num_classes-1]` which are not 'thing' labels.

        In instance segmentation, 'stuff' classes are not taken into
        account in performance metrics computation.

        In panoptic segmentation, 'stuff' classes are taken into account
        in performance metrics computation. Besides, each cloud/scene
        can only have at most one instance of each 'stuff' class.

        IMPORTANT:
        By convention, we assume `y ∈ [0, self.num_classes-1]` ARE ALL
        VALID LABELS (i.e. not 'ignored', 'void', 'unknown', etc), while
        `y < 0` AND `y >= self.num_classes` ARE VOID LABELS.
        """
        return STUFF_CLASSES

    @property
    def class_colors(self):
        """Colors for visualization, if not None, must have the same
        length as `self.num_classes`. If None, the visualizer will use
        the label values in the data to generate random colors.
        """
        return CLASS_COLORS
    
    @property
    def all_base_cloud_ids(self):
        """Dictionary holding lists of paths to the clouds, for each
        stage.

        The following structure is expected:
            `{'train': [...], 'val': [...], 'test': [...]}`
        """
        return TILES
    
    def download_dataset(self):
        """Download the FOR-Instance dataset."""
        if not osp.exists(osp.join(self.root, self._las_name)):
            log.error(
                f"\FOR-Instance does not support automatic download.\n"
                f"Please download the dataset from {self._download_url} "
                f"and save it to {self.root}/ directory and re-run.\n"
                f"The dataset will automatically be unzipped into the "
                f"following structure:\n"
                f"{self.raw_file_structure}"
            )
            sys.exit(1)
    
        # Unzip the file and rename it into the 'root/raw/' directory
        extract_zip(osp.join(self.root, self._las_name), self.root)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(self.root, self._unzip_name), self.raw_dir)

    def read_single_raw_cloud(self, raw_cloud_path):
        """Read a single raw cloud and return a `Data` object, ready to
        be passed to `self.pre_transform`.

        This `Data` object should contain the following attributes:
          - `pos`: point coordinates
          - `y`: OPTIONAL point semantic label
          - `obj`: OPTIONAL `InstanceData` object with instance labels
          - `rgb`: OPTIONAL point color
          - `intensity`: OPTIONAL point LiDAR intensity

        IMPORTANT:
        By convention, we assume `y ∈ [0, self.num_classes-1]` ARE ALL
        VALID LABELS (i.e. not 'ignored', 'void', 'unknown', etc),
        while `y < 0` AND `y >= self.num_classes` ARE VOID LABELS.
        This applies to both `Data.y` and `Data.obj.y`.
        """
        return read_FORinstance_plot(
            raw_cloud_path, intensity=True, semantic=True, instance=True, 
            remap=True)

    @property
    def raw_file_structure(self):
        """Return a string representing the expected raw file structure
        for the dataset. This string is used in the download
        instructions when the dataset is not found in the root
        directory.
        """
        return (
            f"{self.root}/\n"
            f" └── raw/\n"
            f"     ├── CULS/\n"
            f"     │   ├── plot_1_annotated.las\n"
            f"     │   ├── plot_2_annotated.las\n"
            f"     │   ├── ...\n"
            f"     ├── NIBIO/\n"
            f"     │   ├── plot_1_annotated.las\n"
            f"     │   ├── plot_2_annotated.las\n"
            f"     │   ├── ...\n"
            f"     ├── RMIT/\n"
            f"     │   ├── train.las\n"
            f"     │   ├── test.las\n"
            f"     ├── SCION/\n"
            f"     │   ├── plot_31_annotated.las\n"
            f"     │   ├── plot_35_annotated.las\n"
            f"     │   ├── ...\n"
            f"     └── TUWIEN/\n"
            f"         ├── train.las\n"
            f"         ├── test.las\n"
        )

    def id_to_relative_raw_path(self, id):
        """Given a cloud id as stored in `self.cloud_ids`, return the
        path (relative to `self.raw_dir`) of the corresponding raw
        cloud.
        """
        return self.id_to_base_id(id) + '.las'

    def processed_to_raw_path(self, processed_path):
        """Given a processed cloud path from `self.processed_paths`,
        return the absolute path to the corresponding raw cloud.
        """
        hash_dir, area, cloud_id = \
            osp.splitext(processed_path)[0].split(os.sep)[-3:]
        base_cloud_id = self.id_to_base_id(cloud_id)
        raw_path = osp.join(self.raw_dir, area, base_cloud_id) + '.las'

        return raw_path


########################################################################
#                              MiniFORinstance                         #
########################################################################

class MiniFORinstance(FORinstance):
    """A mini version of FOR-Instance with only a few plots for
    experimentation.
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
