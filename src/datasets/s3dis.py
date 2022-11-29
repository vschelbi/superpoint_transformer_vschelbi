import os
import os.path as osp
import glob
import torch
import gdown
import shutil
import logging
import pandas as pd
from src.datasets import BaseDataset, MiniDataset
from src.data import Data, Batch
from src.datasets.s3dis_config import *
from src.utils.download import run_command
from torch_geometric.data import extract_zip

DIR = osp.dirname(osp.realpath(__file__))
log = logging.getLogger(__name__)


__all__ = ['S3DIS', 'MiniS3DIS']


########################################################################
#                                 Utils                                #
########################################################################

def read_s3dis_area(
        area_dir, xyz=True, rgb=True, semantic=True, instance=False,
        is_val=True, verbose=False):
    """Read all S3DIS object-wise annotations in a given Area directory.
    All room-wise data are accumulated into a single cloud.

    :param area_dir: str
        Absolute path to the Area directory, eg: '/some/path/Area_1'
    :param xyz: bool
        Whether XYZ coordinates should be saved in the output Data.pos
    :param rgb: bool
        Whether RGB colors should be saved in the output Data.rgb
    :param semantic: bool
        Whether semantic labels should be saved in the output Data.y
    :param instance: bool
        Whether instance labels should be saved in the output Data.y
    :param is_val: bool
        Whether the output `Batch.is_val` should carry a boolean label
        indicating whether they belong to the Area validation split
    :param verbose: bool
        Verbosity
    :return:
        Batch of accumulated points clouds
    """
    # List the object-wise annotation files in the room
    room_directories = sorted(
        [x for x in glob.glob(osp.join(area_dir, '*')) if osp.isdir(x)])

    #TODO: clean up the multiprocessing. In particular: support
    # `read_s3dis_room` kwargs using partial of something
    import multiprocessing
    with multiprocessing.get_context("spawn").Pool(processes=4) as pool:
        data_list = pool.map(read_s3dis_room, room_directories)

    batch = Batch.from_data_list(data_list)

#    # Read all rooms in the Area and concatenate point clouds in a Batch
#    batch = Batch.from_data_list([
#        read_s3dis_room(
#            r, xyz=xyz, rgb=rgb, semantic=semantic, instance=instance,
#            is_val=is_val, verbose=verbose)
#        for r in room_directories])

    # Convert from Batch to Data
    data_dict = batch.to_dict()
    del data_dict['batch']
    del data_dict['ptr']
    data = Data(**data_dict)

    return data


def read_s3dis_room(
        room_dir, xyz=True, rgb=True, semantic=True, instance=False,
        is_val=True, verbose=False):
    """Read all S3DIS object-wise annotations in a given room directory.

    :param room_dir: str
        Absolute path to the room directory, eg:
        '/some/path/Area_1/office_1'
    :param xyz: bool
        Whether XYZ coordinates should be saved in the output `Data.pos`
    :param rgb: bool
        Whether RGB colors should be saved in the output `Data.rgb`
    :param semantic: bool
        Whether semantic labels should be saved in the output `Data.y`
    :param instance: bool
        Whether instance labels should be saved in the output `Data.y`
    :param is_val: bool
        Whether the output `Data.is_val` should carry a boolean label
        indicating whether they belong to their Area validation split
    :param verbose: bool
        Verbosity
    :return: Data
    """
    if verbose:
        log.debug(f"Reading room: {room_dir}")

    # Initialize accumulators for xyz, RGB, semantic label and instance
    # label
    xyz_list = [] if xyz else None
    rgb_list = [] if rgb else None
    y_list = [] if semantic else None
    o_list = [] if instance else None

    # List the object-wise annotation files in the room
    objects = sorted(glob.glob(osp.join(room_dir, 'Annotations', '*.txt')))
    for i_object, path in enumerate(objects):
        object_name = osp.splitext(osp.basename(path))[0]
        if verbose:
            log.debug(f"Reading object {i_object}: {object_name}")

        # Remove the trailing number in the object name to isolate the
        # object class (eg 'chair_24' -> 'chair')
        object_class = object_name.split('_')[0]

        # Convert object class string to int label. Note that by default
        # if an unknown class is read, it will be treated as 'clutter'.
        # This is necessary because an unknown 'staris' class can be
        # found in some rooms
        label = OBJECT_LABEL.get(object_class, OBJECT_LABEL['clutter'])
        points = pd.read_csv(path, sep=' ', header=None).values

        if xyz:
            xyz_list.append(
                np.ascontiguousarray(points[:, 0:3], dtype='float32'))

        if rgb:
            try:
                rgb_list.append(
                    np.ascontiguousarray(points[:, 3:6], dtype='uint8'))
            except ValueError:
                rgb_list.append(np.zeros((points.shape[0], 3), dtype='uint8'))
                log.warning(f"WARN - corrupted rgb data for file {path}")

        if semantic:
            y_list.append(np.full(points.shape[0], label, dtype='int64'))

        if instance:
            o_list.append(np.full(points.shape[0], i_object, dtype='int64'))

    # Concatenate and convert to torch
    xyz_data = torch.from_numpy(np.concatenate(xyz_list, 0)) if xyz else None
    rgb_data = torch.from_numpy(np.concatenate(rgb_list, 0)) if rgb else None
    y_data = torch.from_numpy(np.concatenate(y_list, 0)) if semantic else None
    o_data = torch.from_numpy(np.concatenate(o_list, 0)) if instance else None

    # Store into a Data object
    data = Data(pos=xyz_data, rgb=rgb_data, y=y_data, o=o_data)

    # Add is_val attribute if need be
    if is_val:
        data.is_val = torch.ones(data.num_nodes, dtype=torch.bool) * (
                osp.basename(room_dir) in VALIDATION_ROOMS)

    return data


########################################################################
#                               S3DIS                               #
########################################################################

class S3DIS(BaseDataset):
    """S3DIS dataset.

    Dataset website: http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    root : `str`
        Root directory where the dataset should be saved.
    fold : `int`
        Integer in [1, ..., 6] indicating the Test Area
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

    def __init__(self, *args, fold=5, **kwargs):
        self.fold = fold
        super().__init__(*args, val_mixed_in_train=True, **kwargs)

    @property
    def class_names(self):
        """List of string names for dataset classes. This list may be
        one-item larger than `self.num_classes` if the last label
        corresponds to 'unlabelled' or 'ignored' indices, indicated as
        `-1` in the dataset labels.
        """
        return CLASS_NAMES

    @property
    def num_classes(self):
        """Number of classes in the dataset. May be one-item smaller
        than `self.class_names`, to account for the last class name
        being optionally used for 'unlabelled' or 'ignored' classes,
        indicated as `-1` in the dataset labels.
        """
        return S3DIS_NUM_CLASSES

    @property
    def all_cloud_ids(self):
        """Dictionary holding lists of clouds ids, for each
        stage.

        The following structure is expected:
            `{'train': [...], 'val': [...], 'test': [...]}`
        """
        return {
            'train': [f'Area_{i}' for i in range(1, 7) if i != self.fold],
            'val': [f'Area_{i}' for i in range(1, 7) if i != self.fold],
            'test': [f'Area_{self.fold}']}

    def download_dataset(self):
        """Download the S3DIS dataset.
        """
        # Download the whole dataset as a single zip file
        if not osp.exists(osp.join(self.root, ZIP_NAME)):
            self.download_zip()

        # Unzip the file and rename it into the `root/raw/` directory. This
        # directory contains the raw Area folders from the zip
        extract_zip(osp.join(self.root, ZIP_NAME), self.root)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(self.root, UNZIP_NAME), self.raw_dir)

        # Patch some erroneous values in the dataset
        cmd = f"patch -ruN -p0 -d  {self.raw_dir} < {PATCH_FILE}"
        run_command(cmd)

    def download_zip(self):
        """Download the S3DIS dataset as a single zip file.
        """
        log.info(
            f"Please, register yourself by filling up the form at {FORM_URL}")
        log.info("***")
        log.info(
            "Press any key to continue, or CTRL-C to exit. By continuing, "
            "you confirm having filled up the form.")
        input("")
        gdown.download(DOWNLOAD_URL, osp.join(self.root, ZIP_NAME), quiet=False)

    def read_single_raw_cloud(self, raw_cloud_path):
        """Read a single raw cloud and return a Data object, ready to
        be passed to `self.pre_transform`.
        """
        return read_s3dis_area(
            raw_cloud_path, xyz=True, rgb=True, semantic=True, instance=False,
            is_val=True, verbose=False)

    @property
    def raw_file_structure(self):
        return f"""
    {self.root}/
        └── {ZIP_NAME}
        └── raw/
            └── Area_{{i_area:1>6}}/
                └── ...
            """

    def id_to_relative_raw_path(self, id):
        """Given a cloud id as stored in `self.cloud_ids`, return the
        path (relative to `self.raw_dir`) of the corresponding raw
        cloud.
        """
        return id


########################################################################
#                              MiniS3DIS                               #
########################################################################

class MiniS3DIS(MiniDataset, S3DIS):
    """A mini version of S3DIS with only 2 areas per stage for
    experimentation.
    """
    _NUM_MINI = 2
