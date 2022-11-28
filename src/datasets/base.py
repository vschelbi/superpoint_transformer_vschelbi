import os
import os.path as osp
import torch
import logging
import hashlib
from datetime import datetime
from tqdm.auto import tqdm as tq
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.dataset import _repr
from src.data import NAG
from src.transforms import RemoveKeys

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)


__all__ = ['BaseDataset', 'MiniDataset']


########################################################################
#                             BaseDataset                              #
########################################################################

class BaseDataset(InMemoryDataset):
    """Base class for datasets.

    Child classes must overwrite the following:

    ```
    MyDataset(BaseDataset):

        def class_names(self):
            pass

        def num_classes(self):
            pass

        def all_clouds(self):
            pass

        def download_dataset(self):
            pass

        def read_single_raw_cloud(self):
            pass

        def processed_to_raw_path(self):
            pass

        def raw_file_structure(self) (optional):
            # Optional: only if your raw or processed file structure
            # differs from the default
            pass

        def cloud_to_relative_raw_path(self):
            # Optional: only if your raw or processed file structure
            # differs from the default
            pass

        def processed_to_raw_path(self):
            # Optional: only if your raw or processed file structure
            # differs from the default
            pass
    ```


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
    _LEVEL0_SAVE_KEYS = ['pos', 'x', 'rgb', 'y', 'node_size', 'super_index']
    _LEVEL0_LOAD_KEYS = ['pos', 'x', 'y', 'node_size', 'super_index']

    def __init__(
            self, root, stage='train', transform=None, pre_transform=None,
            pre_filter=None, on_device_transform=None, x32=True, y_to_csr=True):

        assert stage in ['train', 'val', 'trainval', 'test']

        # Set these attributes before calling parent `__init__` because
        # some attributes will be needed in parent `download` and
        # `process` methods
        self._stage = stage
        self.x32 = x32
        self.y_to_csr = y_to_csr
        self.on_device_transform = on_device_transform

        # Initialization with downloading and all preprocessing
        root = osp.join(root, self.data_subdir_name)
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def class_names(self):
        """List of string names for dataset classes. This list may be
        one-item larger than `self.num_classes` if the last label
        corresponds to 'unlabelled' or 'ignored' indices, indicated as
        `-1` in the dataset labels.
        """
        raise NotImplementedError

    @property
    def num_classes(self):
        """Number of classes in the dataset. May be one-item smaller
        than `self.class_names`, to account for the last class name
        being optionally used for 'unlabelled' or 'ignored' classes,
        indicated as `-1` in the dataset labels.
        """
        raise NotImplementedError

    @property
    def data_subdir_name(self):
        return self.__class__.__name__.lower()

    @property
    def stage(self):
        """Dataset stage. Expected to be 'train', 'val', 'trainval',
        or 'test'
        """
        return self._stage

    @property
    def all_clouds(self):
        """Dictionary holding lists of paths to the clouds, for each
        stage.

        The following structure is expected:
            `{'train': [...], 'val': [...], 'test': [...]}`
        """
        raise NotImplementedError

    @property
    def clouds(self):
        """Filenames of the dataset clouds, based on its `stage`.
        """
        if self.stage == 'trainval':
            return self.all_clouds['train'] + self.all_clouds['val']
        return self.all_clouds[self.stage]

    @property
    def raw_file_structure(self):
        """String to describe to the user the file structure of your
        dataset, at download time.
        """
        return None

    @property
    def raw_file_names(self):
        """The file paths to find in order to skip the download."""
        return self.raw_file_names_3d

    @property
    def raw_file_names_3d(self):
        """Some of the file paths to find in order to skip the download.
        Those are not directly specified inside of `self.raw_file_names`
        in case `self.raw_file_names` would need to be extended (eg with
        3D bounding boxes files).
        """
        return [self.cloud_to_relative_raw_path(x) for x in self.clouds]

    def cloud_to_relative_raw_path(self, cloud):
        """Given a cloud name as stored in `self.clouds`, return the
        path (relative to `self.raw_dir`) of the corresponding raw
        cloud.
        """
        return osp.join(cloud + '.ply')

    @property
    def pre_transform_hash(self):
        """Produce a unique but stable hash based on the dataset's
        `pre_transform` attributes (as exposed by `_repr`).
        """
        if self.pre_transform is None:
            return 'no_pre_transform'
        return hashlib.md5(_repr(self.pre_transform).encode()).hexdigest()

    @property
    def processed_file_names(self):
        """The name of the files to find in the `self.processed_dir`
        folder in order to skip the processing
        """
        # For 'trainval', we use files from 'train' and 'val' to save
        # memory
        # TODO better deal with trainval and val for S3DIS
        if self.stage == 'trainval':
            return [
                osp.join(s, self.pre_transform_hash, f'{w}.h5')
                for s in ('train', 'val')
                for w in self.all_clouds[s]]
        return [
            osp.join(self.stage, self.pre_transform_hash, f'{w}.h5')
            for w in self.clouds]

    def processed_to_raw_path(self, processed_path):
        """Given a processed cloud path from `self.processed_paths`,
        return the absolute path to the corresponding raw cloud.

        Overwrite this method if your raw data does not follow the
        default structure.
        """
        # Extract useful information from <path>
        stage, hash_dir, cloud_name = \
            osp.splitext(processed_path)[0].split('/')[-3:]

        # Read the raw cloud data
        raw_ext = self.raw_file_names_3d[0].splitext[1]
        raw_path = osp.join(self.raw_dir, cloud_name + raw_ext)

        return raw_path

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
        self.download_dataset()

    def download_dataset(self):
        """Download the dataset data. Modify this method to implement
        your own `BaseDataset` child class.
        """
        raise NotImplementedError

    def download_warning(self):
        # Warning message for the user about to download
        print(
            f"WARNING: You are about to download {self.__class__.__name__} "
            f"data.")
        if self.raw_file_structure is not None:
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
            self._process_single_cloud(p)

    def _process_single_cloud(self, cloud_path):
        """Internal method called by `self.process` to preprocess a
        single cloud of 3D points.
        """
        # If required files exist, skip processing
        if osp.exists(cloud_path):
            return

        # Create necessary parent folders if need be
        os.makedirs(osp.dirname(cloud_path), exist_ok=True)

        # Read the raw cloud corresponding to the final processed
        # `cloud_path` and convert it to a Data object
        raw_path = self.processed_to_raw_path(cloud_path)
        data = self.read_single_raw_cloud(raw_path)

        # TODO: this is dirty, may cause subsequent collisions with
        #  self.num_classes ?
        if getattr(data, 'y', None) is not None:
            data.y[data.y == -1] = self.num_classes

        # Apply pre_transform
        if self.pre_transform is not None:
            nag = self.pre_transform(data)
        else:
            nag = NAG([data])

        # To save some disk space, we discard some level-0 attributes
        level0_keys = set(nag[0].keys) - set(self._LEVEL0_SAVE_KEYS)
        nag._list[0] = RemoveKeys(keys=level0_keys)(nag[0])

        # Save pre_transformed data to the processed dir/<path>
        # TODO: is you do not throw away level-0 neighbors, make sure
        #  that they contain no '-1' empty neighborhoods, because if
        #  you load them for batching, the pyg reindexing mechanism will
        #  break indices will not index update
        nag.save(cloud_path, x32=self.x32, y_to_csr=self.y_to_csr)

    def read_single_raw_cloud(self, path):
        """Read a single raw cloud and return a Data object, ready to
        be passed to `self.pre_transform`.
        """
        raise NotImplementedError

    def get_class_weight(self, smooth='sqrt'):
        """Compute class weights based on the labels distribution in the
        dataset. Optionally a 'smooth' function may be passed to
        smoothen the weights statistics.
        """
        assert smooth in [None, 'sqrt', 'log']

        # Read the first NAG just to know how many levels we have in the
        # preprocessed NAGs.
        nag = self[0]
        low = nag.num_levels - 1

        # Make sure the dataset has labels
        if nag[0].y is None:
            raise ValueError(
                f'{self} does not have labels to compute class weights on.')
        del nag

        # To be as fast as possible, we read only the last level of each
        # NAG, and accumulate the class counts from the label histograms
        counts = torch.zeros(self.num_classes)
        for i in range(len(self)):
            y = NAG.load(self.processed_paths[i], low=low, keys_low=['y'])[0].y
            counts += y.sum(dim=0)[:self.num_classes]

        # Compute the class weights. Optionally, a 'smooth' function may
        # be applied to smoothen the weights statistics
        if smooth == 'sqrt':
            counts = counts.sqrt()
        if smooth == 'log':
            counts = counts.log()

        weights = 1 / (counts + 1)
        weights /= weights.sum()

        return weights

    def __len__(self):
        """Number of clouds in the dataset."""
        return len(self.clouds)

    def __getitem__(self, idx):
        """Load a preprocessed NAG from disk and apply `self.transform`
        if any.
        """
        nag = NAG.load(
            self.processed_paths[idx], keys_low=self._LEVEL0_LOAD_KEYS)
        nag = nag if self.transform is None else self.transform(nag)
        return nag


########################################################################
#                         MiniS3DISCylinder                         #
########################################################################

class MiniDataset:
    """A class to make a BaseDataset smaller with only 2 clouds from
    each stage, for experimentation. This class is useless by itself,
    it should be used for multiple inheritance on a BaseDataset class.

    For instance:

    ```
    MyDataset(MiniDataset, BaseDataset):
        ...
    ```
    """
    _NUM_MINI = 2

    @property
    def all_clouds(self):
        return {k: v[:self._NUM_MINI] for k, v in super().all_clouds.items()}

    # We have to include this method, otherwise the parent class skips
    # processing
    def process(self):
        super().process()

    # We have to include this method, otherwise the parent class skips
    # processing
    def download(self):
        super().download()
