import glob
import torch
import logging
import pandas as pd
import os.path as osp
from src.datasets import BaseDataset, MiniDataset
from src.data import Data, Batch
from src.datasets.s3dis_config import *
from src.utils.download import run_command

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
    room_directories = sorted(glob.glob(osp.join(area_dir, '*')))
    return Batch.from_data_list([
        read_s3dis_room(
            r, xyz=xyz, rgb=rgb, semantic=semantic, instance=instance,
            is_val=is_val, verbose=verbose)
        for r in room_directories])


def read_s3dis_room(
        room_dir, xyz=True, rgb=True, semantic=True, instance=False,
        is_val=False, verbose=False):
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
    # Initialize accumulators for xyz, RGB, semantic label and instance
    # label
    xyz_list = [] if xyz else None
    rgb_list = [] if rgb else None
    y_list = [] if semantic else None
    o_list = [] if instance else None

    # List the object-wise annotation files in the room
    objects = sorted(glob.glob(osp.join(room_dir, 'Annotations', '*.txt')))
    for i_object, path in objects:
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
            y_list.append(np.full(N, label, dtype='int64'))
            
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


def read_s3dis_format(
        train_file, room_name, label_out=True, verbose=False, debug=False):
    """Read room data from S3DIS raw format and return a Data object.
    """

    room_type = room_name.split("_")[0]
    room_label = ROOM_TYPES[room_type]
    raw_path = osp.join(train_file, "{}.txt".format(room_name))
    if debug:
        reader = pd.read_csv(raw_path, delimiter="\n")
        RECOMMENDED = 6
        for idx, row in enumerate(reader.values):
            row = row[0].split(" ")
            if len(row) != RECOMMENDED:
                log.info("1: {} row {}: {}".format(raw_path, idx, row))

            try:
                for r in row:
                    r = float(r)
            except:
                log.info("2: {} row {}: {}".format(raw_path, idx, row))

        return True
    else:
        room_ver = pd.read_csv(raw_path, sep=" ", header=None).values
        xyz = np.ascontiguousarray(room_ver[:, 0:3], dtype="float32")
        try:
            rgb = np.ascontiguousarray(room_ver[:, 3:6], dtype="uint8")
        except ValueError:
            rgb = np.zeros((room_ver.shape[0], 3), dtype="uint8")
            log.warning("WARN - corrupted rgb data for file %s" % raw_path)
        if not label_out:
            return xyz, rgb
        n_ver = len(room_ver)
        del room_ver
        nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(xyz)
        semantic_labels = np.zeros((n_ver,), dtype="int64")
        room_label = np.asarray([room_label])
        instance_labels = np.zeros((n_ver,), dtype="int64")
        objects = glob.glob(osp.join(train_file, "Annotations/*.txt"))
        i_object = 1
        for single_object in objects:
            object_name = osp.splitext(osp.basename(single_object))[0]
            if verbose:
                log.debug("adding object " + str(i_object) + " : " + object_name)
            object_class = object_name.split("_")[0]
            object_label = object_name_to_label(object_class)
            obj_ver = pd.read_csv(single_object, sep=" ", header=None).values
            _, obj_ind = nn.kneighbors(obj_ver[:, 0:3])
            semantic_labels[obj_ind] = object_label
            instance_labels[obj_ind] = i_object
            i_object = i_object + 1

        return (
            torch.from_numpy(xyz),
            torch.from_numpy(rgb),
            torch.from_numpy(semantic_labels),
            torch.from_numpy(instance_labels),
            torch.from_numpy(room_label),
        )

def read_s3dis_area(
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
#                               S3DIS                               #
########################################################################

class S3DIS(BaseDataset):
    """S3DIS dataset.

    Dataset website: http://buildingparser.stanford.edu/dataset.html

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
        raise S3DIS_NUM_CLASSES

    @property
    def all_clouds(self):
        """Dictionary holding lists of paths to the clouds, for each
        stage.

        The following structure is expected:
            `{'train': [...], 'val': [...], 'test': [...]}`
        """
        # TODO this will be problematic for 'trainval' in S3DIS. Not a
        #  simple list concatenation for S3DIS...
        return {train:f'Area_{i}'}

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
            └── {ZIP_NAME}
            └── Area_{{i_area:1>6}}/
                └── ...
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
