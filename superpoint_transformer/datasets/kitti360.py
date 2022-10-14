import os
import os.path as osp
import torch
from plyfile import PlyData
import logging
from tqdm.auto import tqdm as tq
from datetime import datetime

from torch_geometric.data import InMemoryDataset
from superpoint_transformer.data import Data, NAG
from superpoint_transformer.datasets.kitti360_config import CVLIBS_URL, \
    KITTI360_NUM_CLASSES, WINDOWS, SEQUENCES, ID2TRAINID
from superpoint_transformer.utils.download import run_command

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


########################################################################
#                               KITTI360                               #
########################################################################

class KITTI360(InMemoryDataset):
    """KITTI360 dataset.

    Dataset website: http://www.cvlibs.net/datasets/kitti-360/

    Parameters
    ----------
    root : `str`
        Root directory where the dataset should be saved.
    split : {'train', 'val', 'test', 'trainval'}, optional
    transform : `callable`, optional
        transform function operating on data.
    pre_transform : `callable`, optional
        pre_transform function operating on data.
    pre_filter : `callable`, optional
        pre_filter function operating on data.
    """
    num_classes = KITTI360_NUM_CLASSES
    _WINDOWS = WINDOWS
    _SEQUENCES = SEQUENCES

    def __init__(
            self, root, split="train", transform=None, pre_transform=None,
            pre_filter=None, x32=True, y_to_csr=True):

        # Initialization with downloading and all preprocessing
        super().__init__(root, transform, pre_transform, pre_filter)

        self._split = split
        self.x32 = x32
        self.y_to_csr = y_to_csr

    @property
    def split(self):
        return self._split

    @property
    def has_labels(self):
        """Self-explanatory attribute needed for BaseDataset."""
        return self.split != 'test'

    @property
    def windows(self):
        """Filenames of the dataset windows."""
        if self.split == 'trainval':
            return self._WINDOWS['train'] + self._WINDOWS['val']
        return self._WINDOWS[self.split]

    @property
    def raw_file_structure(self):
        return """
    root_dir/
        └── raw/
            └── data_3d_semantics/
                └── 2013_05_28_drive_{seq:0>4}_sync/
                    └── static/
                        └── {start_frame:0>10}_{end_frame:0>10}.ply
            """

    @property
    def raw_file_names(self):
        """The file paths to find in order to skip the download."""
        return self.raw_file_names_3d

    @property
    def raw_file_names_3d(self):
        """Some of the file paths to find in order to skip the download.
        Those are not directly specified inside of self.raw_file_names
        in case self.raw_file_names would need to be extended (eg with
        3D bounding boxes files).
        """
        return [
            osp.join('data_3d_semantics', x.split('/')[0], 'static',
                x.split('/')[1] + '.ply')
            for x in self.windows]

    @property
    def processed_file_names(self):
        """The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing
        """
        # For 'trainval', we use files from 'train' and 'val' to save
        # memory
        if self.split == 'trainval':
            return [
                osp.join(s, '3d', f'{w}.pt')
                for s in ('train', 'val')
                for w in self._WINDOWS[s]]
        return [osp.join(self.split, '3d', f'{w}.pt') for w in self.windows]

    @property
    def submission_dir(self):
        """Submissions are saved in the `submissions` folder, in the
        same hierarchy as `raw` and `processed` directories. Each
        submission has a sub-directory of its own, named based on the
        date and time of creation.
        """
        submissions_dir = osp.join(self.root, "submissions")
        date = '-'.join([
            f'{getattr(datetime.now(), x)}'
            for x in ['year', 'month', 'day']])
        time = '-'.join([
            f'{getattr(datetime.now(), x)}'
            for x in ['hour', 'minute', 'second']])
        submission_name = f'{date}_{time}'
        path = osp.join(submissions_dir, submission_name)
        return path

    def download(self):
        self.download_warning()

        # Location of the KITTI-360 download shell scripts
        here = osp.dirname(osp.abspath(__file__))
        scripts_dir = osp.join(here, '../../../scripts/datasets')

        # Accumulated 3D point clouds with annotations
        if not all(osp.exists(osp.join(self.raw_dir, x))
                   for x in self.raw_file_names_3d):
            if self.split != 'test':
                msg = 'Accumulated Point Clouds for Train & Val (12G)'
            else:
                msg = 'Accumulated Point Clouds for Test (1.2G)'
            self.download_message(msg)
            script = osp.join(scripts_dir, 'download_kitti360_3d_semantics.sh')
            run_command([f'{script} {self.raw_dir} {self.split}'])

    def download_warning(self):
        # Warning message for the user about to download
        print(
            f"WARNING: You are about to download KITTI-360 data from: "
            f"{CVLIBS_URL}")
        print("Files will be organized in the following structure:")
        print(self.raw_file_structure)
        print("")
        print("Press any key to continue, or CTRL-C to exit.")
        input("")
        print("")

    def download_message(self, msg):
        print(f'Downloading "{msg}" to {self.raw_dir}...')

    def process(self):
        for p in tq(self.processed_paths):
            self._process_window(p)

    def _process_window(self, window_path):
        """Internal method called by `self.process` to preprocess a
        single KITTI360 window of 3D points.
        """
        # If required files exist, skip processing
        if osp.exists(window_path):
            return

        # Extract useful information from <path>
        split, modality, sequence_name, window_name = \
            osp.splitext(window_path)[0].split('/')[-4:]

        # Create necessary parent folders if need be
        os.makedirs(osp.dirname(window_path), exist_ok=True)

        # Read the raw window data
        raw_window_path = osp.join(
            self.raw_dir, 'data_3d_semantics', sequence_name, 'static',
            window_name + '.ply')
        data = read_kitti360_window(
            raw_window_path, semantic=True, instance=False, remap=True)

        # Apply pre_transform
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Save pre_transformed data to the processed dir/<path>
        data.save(window_path, x32=self.x32, y_to_csr=self.y_to_csr)

    def __len__(self):
        """Number of windows in the dataset."""
        return len(self.windows)

    def __getitem__(self, idx):
        """Load a preprocessed NAG from disk and apply `self.transform`
        if any.
        """
        nag = NAG.load(self.processed_paths[idx])
        nag = nag if self.transform is None else self.transform(nag)
        return nag


########################################################################
#                         MiniKITTI360Cylinder                         #
########################################################################

class MiniKITTI360(KITTI360):
    """A mini version of KITTI360 with only a few windows for
    experimentation.
    """
    _WINDOWS = {k: v[:2] for k, v in WINDOWS.items()}

    # We have to include this method, otherwise the parent class skips
    # processing
    def process(self):
        super().process()

    # We have to include this method, otherwise the parent class skips
    # processing
    def download(self):
        super().download()
